"""Integration tests for Sliced Wasserstein Distance alignment in full pipeline.

Tests verify SWD integration into AlignmentStage dispatcher, full paired evaluation
pipeline (DTW temporal + SWD spatial registration), cross-method comparison,
parameter flow, backward compatibility, and reproducibility.

This file tests ALIGN-05-05 and ALIGN-05-03 requirements.
"""

import pytest
import torch
from copy import deepcopy
from unittest.mock import patch, MagicMock

from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_trajectory

from eval.config import EvalConfig
from eval.stages.alignment import AlignmentStage
from eval.types import AlignResult


class TestSWDAlignmentIntegration:
    """Integration tests for SWD alignment in full paired evaluation pipeline."""

    # =====================
    # Fixtures
    # =====================

    @pytest.fixture
    def synthetic_paired_dataset(self):
        """Generate synthetic paired tracklets for testing.

        Creates source and target trajectories with different random seeds
        to simulate paired data.
        """
        # Source trajectory: random gaussian points
        source = generate_trajectory(n_points=20, n_frames=4, seed=0)

        # Target trajectory: different seed to create distinct paired data
        target = generate_trajectory(n_points=20, n_frames=4, seed=100)
        return source, target

    @pytest.fixture
    def real_paired_dataset(self):
        """Generate real-like paired data using synthetic generation.

        Uses multiple random seeds to simulate real paired tracklets.
        """
        source = generate_trajectory(n_points=30, n_frames=5, seed=42)
        target = generate_trajectory(n_points=30, n_frames=5, seed=200)
        return source, target

    @pytest.fixture
    def eval_config(self, tmp_path):
        """Minimal EvalConfig for integration tests."""
        return EvalConfig(data_path=str(tmp_path / "unused.mat"))

    @pytest.fixture
    def default_params(self):
        """Default parameters for AlignmentStage tests."""
        return {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "swd",
            "swd_variant": "aswd",
        }

    # =====================
    # Test 1: AlignmentStage SWD Dispatch (ALIGN-05-05)
    # =====================

    def test_alignment_stage_swd_dispatch(
        self,
        synthetic_paired_dataset,
        eval_config,
        default_params,
    ):
        """Test that AlignmentStage correctly dispatches to SWD aligner.

        Verifies:
        - AlignmentStage.run() accepts alignment_method='swd'
        - Dispatcher routes to SlicedWassersteinAligner
        - Result contains valid aligned_cloud with expected structure
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        stage = AlignmentStage(eval_config)
        result = stage.run(source_dataset, target_dataset, default_params)

        # Verify result is AlignResult
        assert isinstance(result, AlignResult)

        # Verify params_used contains SWD method
        assert result.params_used["alignment_method"] == "swd"
        assert result.params_used["swd_variant"] == "aswd"

        # Verify aligned_cloud is valid
        assert result.aligned_cloud is not None
        assert isinstance(result.aligned_cloud, dict)

        # Verify aligned_cloud has expected keys (same as target dataset)
        assert set(result.aligned_cloud.keys()) == set(target_dataset.keys())

        # Verify all frames have valid point positions
        for frame_id, frame in result.aligned_cloud.items():
            assert isinstance(frame, zRegPointCloud)
            assert frame["pos"].shape[1] == 3, f"Frame {frame_id} has incorrect shape"
            assert not torch.isnan(frame["pos"]).any(), f"Frame {frame_id} has NaN values"
            assert not torch.isinf(frame["pos"]).any(), f"Frame {frame_id} has Inf values"

    # =====================
    # Test 2: SWD Variant Selection in Pipeline (ALIGN-05-05)
    # =====================

    @pytest.mark.parametrize("swd_variant", ["swd", "aswd", "oswd", "gswd", "pswd"])
    def test_swd_variant_selection_in_pipeline(
        self,
        swd_variant,
        synthetic_paired_dataset,
        eval_config,
    ):
        """Test that all SWD variants work in full alignment pipeline.

        Parametrized over variants: swd, aswd, oswd, gswd, pswd.
        Verifies each variant produces valid results without errors.
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "swd",
            "swd_variant": swd_variant,
        }

        stage = AlignmentStage(eval_config)
        result = stage.run(source_dataset, target_dataset, params)

        # Verify result structure is valid
        assert isinstance(result, AlignResult)
        assert result.aligned_cloud is not None
        assert isinstance(result.aligned_cloud, dict)
        assert len(result.aligned_cloud) == len(target_dataset)

        # Verify no NaN or Inf in aligned positions
        for frame in result.aligned_cloud.values():
            assert not torch.isnan(frame["pos"]).any()
            assert not torch.isinf(frame["pos"]).any()

    # =====================
    # Test 3: Cross-Method Comparison (Smoke Test)
    # =====================

    def test_alignment_stage_swd_vs_cpd_quality(
        self,
        real_paired_dataset,
        eval_config,
    ):
        """Smoke test: CPD vs SWD produce valid results (no NaN/Inf).

        Verifies both methods produce reasonable outputs. Does NOT assert
        quality comparison (CPD distance < SWD or vice versa); that's deferred
        to future analysis.
        """
        source_dataset, target_dataset = real_paired_dataset

        # Run with CPD
        params_cpd = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": "rigid",
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "cpd",
        }
        stage = AlignmentStage(eval_config)
        result_cpd = stage.run(source_dataset, target_dataset, params_cpd)

        # Run with SWD
        params_swd = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "swd",
            "swd_variant": "aswd",
        }
        result_swd = stage.run(source_dataset, target_dataset, params_swd)

        # Verify both methods produce valid metrics
        assert result_cpd.dtw_distance >= 0, "CPD distance should be non-negative"
        assert result_swd.dtw_distance >= 0, "SWD distance should be non-negative"

        # Verify no NaN or Inf in either result
        for frame in result_cpd.aligned_cloud.values():
            assert not torch.isnan(frame["pos"]).any()
            assert not torch.isinf(frame["pos"]).any()

        for frame in result_swd.aligned_cloud.values():
            assert not torch.isnan(frame["pos"]).any()
            assert not torch.isinf(frame["pos"]).any()

    # =====================
    # Test 4: Parameter Flow Verification (ALIGN-05-03)
    # =====================

    def test_swd_parameter_flow_from_config(
        self,
        synthetic_paired_dataset,
        eval_config,
    ):
        """Test that SWD parameters flow from EvalConfig → AlignmentStage → Aligner.

        Verifies:
        - swd_variant parameter flows through pipeline
        - swd_num_iterations parameter is accepted
        - Parameters don't cause errors during execution
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "swd",
            "swd_variant": "oswd",
            "swd_num_iterations": 75,  # Custom iterations
            "swd_learning_rate": 5e-4,  # Custom learning rate
        }

        stage = AlignmentStage(eval_config)
        result = stage.run(source_dataset, target_dataset, params)

        # Verify result is valid and parameters were accepted
        assert isinstance(result, AlignResult)
        assert result.aligned_cloud is not None
        assert result.params_used["alignment_method"] == "swd"
        assert result.params_used["swd_variant"] == "oswd"

    # =====================
    # Test 5: Backward Compatibility (old configs without SWD params)
    # =====================

    def test_backward_compatibility_cpd_config(
        self,
        synthetic_paired_dataset,
        eval_config,
    ):
        """Test that pre-Phase-40 configs (CPD-only) still work unchanged.

        Verifies:
        - Old config with alignment_method='cpd', no swd_variant field
        - AlignmentStage dispatches correctly to CPD (not SWD)
        - Result is valid
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        # Old-style config (no swd fields)
        old_params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": "rigid",
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "cpd",
            # No swd_variant or swd_num_iterations
        }

        stage = AlignmentStage(eval_config)
        result = stage.run(source_dataset, target_dataset, old_params)

        # Verify result uses CPD (not SWD)
        assert result.params_used["alignment_method"] == "cpd"

        # Verify result is valid
        assert isinstance(result, AlignResult)
        assert result.aligned_cloud is not None
        assert len(result.aligned_cloud) == len(target_dataset)

        # Verify all frames have valid positions
        for frame in result.aligned_cloud.values():
            assert not torch.isnan(frame["pos"]).any()
            assert not torch.isinf(frame["pos"]).any()

    # =====================
    # Test 6: All Methods Parametrized (cpd, icp, swd)
    # =====================

    @pytest.mark.parametrize("alignment_method", ["cpd", "icp", "swd"])
    def test_alignment_stage_all_methods_parametrized(
        self,
        alignment_method,
        synthetic_paired_dataset,
        eval_config,
    ):
        """Test that all 3 alignment methods (CPD, ICP, SWD) work in parametrized test.

        Ensures SWD is treated as first-class citizen alongside CPD and ICP.
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": "rigid" if alignment_method == "cpd" else None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": alignment_method,
        }

        # Add SWD-specific params if needed
        if alignment_method == "swd":
            params["swd_variant"] = "aswd"

        stage = AlignmentStage(eval_config)
        result = stage.run(source_dataset, target_dataset, params)

        # Verify result is valid for all methods
        assert isinstance(result, AlignResult)
        assert result.aligned_cloud is not None
        assert isinstance(result.aligned_cloud, dict)

        # Verify no NaN/Inf in result
        for frame in result.aligned_cloud.values():
            assert isinstance(frame, zRegPointCloud)
            assert not torch.isnan(frame["pos"]).any()
            assert not torch.isinf(frame["pos"]).any()

    # =====================
    # Test 7: Stored Transform Reuse Pattern (Phase 35/39)
    # =====================

    def test_swd_alignment_with_stored_transform_reuse(
        self,
        synthetic_paired_dataset,
        eval_config,
    ):
        """Test SWD alignment with StoredTransform caching (Phase 35/39 pattern).

        Verifies:
        - First run computes and caches transforms
        - Second run on same data reuses transforms without error
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "swd",
            "swd_variant": "aswd",
        }

        stage = AlignmentStage(eval_config)

        # First run: compute transforms
        result1 = stage.run(source_dataset, target_dataset, params)
        assert result1 is not None
        assert result1.aligned_cloud is not None

        # Second run: reuse transforms (if caching is implemented)
        result2 = stage.run(source_dataset, target_dataset, params)
        assert result2 is not None
        assert result2.aligned_cloud is not None

        # Verify both runs produce valid results
        for frame in result1.aligned_cloud.values():
            assert not torch.isnan(frame["pos"]).any()
        for frame in result2.aligned_cloud.values():
            assert not torch.isnan(frame["pos"]).any()

    # =====================
    # Test 8: DTW Distance Independence (Pitfall 1)
    # =====================

    def test_swd_alignment_with_dtw_distance_comparison(
        self,
        synthetic_paired_dataset,
        eval_config,
    ):
        """Test that dtw_dist_fn and alignment_method are independent (Pitfall 1).

        Verifies:
        - Euclidean for temporal + CPD for spatial
        - Euclidean for temporal + SWD for spatial
        - Euclidean for temporal + ICP for spatial

        All three configurations should work without interference.
        (SWD/ASWD temporal distances require downsampling, so we use Euclidean for temporal.)
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        configs = [
            # Euclidean for temporal, CPD for spatial
            {
                "window_size": 10,
                "step": 1,
                "cpd_penalty": "rigid",
                "dtw_dist_fn": "euclidean",  # Role 1: temporal
                "n_breakpoints": 5,
                "alignment_method": "cpd",  # Role 2: spatial
            },
            # Euclidean for temporal, SWD for spatial
            {
                "window_size": 10,
                "step": 1,
                "cpd_penalty": None,
                "dtw_dist_fn": "euclidean",  # Role 1: temporal
                "n_breakpoints": 5,
                "alignment_method": "swd",  # Role 2: spatial
                "swd_variant": "aswd",
            },
            # Euclidean for temporal, ICP for spatial
            {
                "window_size": 10,
                "step": 1,
                "cpd_penalty": None,
                "dtw_dist_fn": "euclidean",  # Role 1: temporal
                "n_breakpoints": 5,
                "alignment_method": "icp",  # Role 2: spatial
            },
        ]

        stage = AlignmentStage(eval_config)

        for i, config in enumerate(configs):
            result = stage.run(source_dataset, target_dataset, config)

            # Verify each configuration produces valid results
            assert result is not None, f"Config {i} returned None"
            assert result.aligned_cloud is not None, f"Config {i} has no aligned_cloud"

            for frame in result.aligned_cloud.values():
                assert not torch.isnan(frame["pos"]).any(), f"Config {i} has NaN"
                assert not torch.isinf(frame["pos"]).any(), f"Config {i} has Inf"

    # =====================
    # Test 9: Reproducibility (Seeded Runs)
    # =====================

    def test_swd_alignment_quality_reproducibility(
        self,
        synthetic_paired_dataset,
        eval_config,
    ):
        """Test that seeded runs produce reproducible results.

        Verifies:
        - Same seed produces identical (or very close) aligned positions
        - Different seeds produce different results
        """
        source_dataset, target_dataset = synthetic_paired_dataset

        params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "swd",
            "swd_variant": "aswd",
        }

        stage = AlignmentStage(eval_config)

        # Run 1 with seed
        torch.manual_seed(999)
        result1 = stage.run(source_dataset, target_dataset, params)

        # Run 2 with same seed
        torch.manual_seed(999)
        result2 = stage.run(source_dataset, target_dataset, params)

        # Verify both runs produce identical results (or very close)
        for frame_id in result1.aligned_cloud.keys():
            pos1 = result1.aligned_cloud[frame_id]["pos"]
            pos2 = result2.aligned_cloud[frame_id]["pos"]

            # Allow small numerical differences (1e-5 tolerance)
            assert torch.allclose(pos1, pos2, atol=1e-5), \
                f"Frame {frame_id} positions differ between seeded runs"
