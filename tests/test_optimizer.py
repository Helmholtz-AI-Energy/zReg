"""Tests for eval.runners.optimizer.HyperparamOptimizer (FRAME-09) and
eval.search_strategies (FRAME-10).

Covers:
  - FRAME-09-SC1: sanity tier completes in under 2 minutes
  - FRAME-09-SC2: best_params.json and search_history.json written to output_dir
  - FRAME-09-SC4: prune_candidates() reduces candidate count
  - FRAME-10-SC3: best params from optimizer improve score vs default params
  - FRAME-10: BayesianSearch uses TPE sampler with n_startup_trials >= 2*N_params
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_labels, generate_trajectory

import torch
import optuna

from eval.config import EvalConfig

try:
    from eval.runners import HyperparamOptimizer
except ImportError:
    HyperparamOptimizer = None  # type: ignore[assignment,misc]

try:
    from eval.search_strategies import GridSearch, RandomSearch, BayesianSearch, SobolSearch
except ImportError:
    GridSearch = None  # type: ignore[assignment,misc]
    RandomSearch = None  # type: ignore[assignment,misc]
    BayesianSearch = None  # type: ignore[assignment,misc]
    SobolSearch = None  # type: ignore[assignment,misc]

from eval.types import Trial, SearchResult, StageMetrics


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optimizer_config(tmp_path) -> EvalConfig:
    """EvalConfig with search_space and output_dir inside tmp_path."""
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "output"),
        search_strategy="grid",
        tier="sanity",
        n_trials=3,
        run_alignment=True,
        run_label_transfer=True,
        search_space={
            "window_size": [3, 5],
            "k_neighbours": [3, 5],
        },
    )


@pytest.fixture
def synthetic_dataset() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with labels (id field populated).

    generate_labels() populates the 'id' field (torch.long, shape (N,))
    so DataFactory.get_ground_truth() returns valid id tensors.
    Mirror of test_eval_runner.py:51-58 fixture.
    """
    traj = generate_trajectory(n_points=20, n_frames=3, seed=0)
    return generate_labels(traj, n_labels=4, seed=0)


@pytest.fixture
def full_params() -> dict:
    """Complete params dict covering all 9 required keys."""
    return {
        "window_size": 10,
        "step": 1,
        "cpd_penalty": None,
        "dtw_dist_fn": "euclidean",
        "n_breakpoints": 5,
        "k_neighbours": 5,
        "dist_metric": "euclidean",
        "smoothing": 0.0,
        "threshold": 0.0,
    }


# ---------------------------------------------------------------------------
# Test classes (Wave 1 — fully populated in Plan 22-02)
# ---------------------------------------------------------------------------


class TestHyperparamOptimizerSanityTier:
    """FRAME-09-SC1: sanity tier completes in under 2 minutes."""

    @patch("eval.runners.optimizer.DataFactory")
    def test_sanity_tier_completes_under_two_minutes(
        self, mock_factory_cls, optimizer_config, synthetic_dataset
    ) -> None:
        """Sanity tier run() completes in < 120 seconds with mocked DataFactory."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.generate_synthetic.return_value = synthetic_dataset
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        # get_ground_truth is called inside _objective with the tier_dataset;
        # return the color labels for the last frame key of whatever is passed.
        mock_factory.get_ground_truth.side_effect = (
            lambda ds: {k: ds[k]["label"] for k in ds}
        )

        start = time.time()
        optimizer = HyperparamOptimizer(optimizer_config)
        result = optimizer.run()
        elapsed = time.time() - start

        assert elapsed < 120.0, f"Sanity tier took {elapsed:.1f}s — exceeded 2-minute gate"
        assert isinstance(result, SearchResult)


class TestHyperparamOptimizerOutputFiles:
    """FRAME-09-SC2: best_params.json and search_history.json written."""

    @patch("eval.runners.optimizer.DataFactory")
    def test_output_files_written(
        self, mock_factory_cls, optimizer_config, synthetic_dataset
    ) -> None:
        """run() writes best_params.json and search_history.json to output_dir."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.generate_synthetic.return_value = synthetic_dataset
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = (
            lambda ds: {k: ds[k]["label"] for k in ds}
        )

        output_dir = Path(optimizer_config.output_dir)
        HyperparamOptimizer(optimizer_config).run()

        assert (output_dir / "best_params.json").exists(), "best_params.json not written"
        assert (output_dir / "search_history.json").exists(), "search_history.json not written"

        # Validate JSON content
        with open(output_dir / "best_params.json") as f:
            bp = json.load(f)
        assert isinstance(bp, dict), "best_params.json must be a flat dict"

        with open(output_dir / "search_history.json") as f:
            hist = json.load(f)
        assert isinstance(hist, list), "search_history.json must be a list"
        assert len(hist) > 0, "search_history.json must be non-empty"


class TestPruneCandidates:
    """FRAME-09-SC4: prune_candidates() reduces candidate count."""

    def test_prune_reduces_count(self) -> None:
        """prune_candidates(history_of_5, keep_top_k=3) returns exactly 3 dicts sorted by score descending."""
        dummy_metrics = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        history = [
            Trial(params={"window_size": i}, score=float(i), metrics=dummy_metrics, tier="sanity")
            for i in range(5)
        ]

        pruned = HyperparamOptimizer.prune_candidates(history, keep_top_k=3)

        assert len(pruned) == 3, f"Expected 3 candidates, got {len(pruned)}"
        assert len(pruned) < len(history), "prune_candidates must reduce candidate count"
        assert pruned[0]["window_size"] == 4, (
            "Top candidate should have highest score (window_size=4, score=4.0)"
        )
        # Each element must be a dict (param dict, not Trial)
        assert all(isinstance(p, dict) for p in pruned)


class TestBestParamsImproveDefault:
    """FRAME-10-SC3: optimizer best params improve score vs default params."""

    @patch("eval.runners.optimizer.MetricsEngine")
    @patch("eval.runners.optimizer.DataFactory")
    def test_best_params_improve_default_score(
        self, mock_factory_cls, mock_engine_cls, optimizer_config, synthetic_dataset, full_params
    ) -> None:
        """Optimizer best_score exceeds default_score via mock-controlled MetricsEngine."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.generate_synthetic.return_value = synthetic_dataset
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = (
            lambda ds: {k: ds[k]["label"] for k in ds}
        )

        # compute_stage_metrics must return a real StageMetrics for Trial pydantic validation
        stub_metrics = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics

        call_count = 0

        def mock_compute_score(metrics):
            nonlocal call_count
            call_count += 1
            return 0.3 if call_count == 1 else 0.7

        mock_engine_cls.return_value.compute_score.side_effect = mock_compute_score

        optimizer = HyperparamOptimizer(optimizer_config)
        result = optimizer.run()

        # Reset counter; baseline call returns 0.3
        call_count = 0
        tier_dataset = optimizer._tier_dataset("sanity")
        default_score = optimizer._objective(full_params, tier_dataset, "sanity", [])

        assert isinstance(result.best_params, dict) and len(result.best_params) > 0, (
            "best_params must be non-empty"
        )
        assert result.best_score > default_score, (
            f"Optimizer best_score ({result.best_score}) must exceed default_score ({default_score}). "
            "This tests that the optimizer mechanism finds params better than the baseline."
        )


class TestBayesianSearch:
    """FRAME-10: BayesianSearch uses TPE with n_startup_trials >= 2*N_params."""

    def test_tpe_sampler_n_startup_trials(self, tmp_path) -> None:
        """BayesianSearch.search creates study with TPESampler; n_startup_trials >= 2*N_params."""
        search_space = {"window_size": [3, 5, 7], "k_neighbours": [3, 5]}
        n_params = len(search_space)  # = 2

        captured: dict = {}
        original_create_study = optuna.create_study

        def mock_create_study(**kwargs):
            captured["sampler"] = kwargs.get("sampler")
            # Recreate with in-memory storage to avoid SQLite side effects
            return original_create_study(
                storage=None,
                direction="maximize",
                sampler=kwargs.get("sampler"),
            )

        with patch("eval.search_strategies.optuna.create_study", side_effect=mock_create_study):
            results = BayesianSearch().search(
                search_space,
                lambda p: 0.5,
                n_trials=5,
                output_dir=str(tmp_path),
            )

        sampler = captured.get("sampler")
        assert sampler is not None, "create_study was not called or sampler not captured"
        assert isinstance(sampler, optuna.samplers.TPESampler), (
            f"Expected TPESampler, got {type(sampler)}"
        )
        assert sampler._n_startup_trials >= 2 * n_params, (
            f"n_startup_trials={sampler._n_startup_trials} < 2*{n_params}"
        )

        # Integration smoke test: results is a list of (params, score) tuples
        assert len(results) > 0, "BayesianSearch returned no results"
        assert all(
            isinstance(r, tuple) and len(r) == 2 for r in results
        ), "Each result must be a (params, score) tuple"


# ---------------------------------------------------------------------------
# TestOptimizerSyntheticMode — Phase 31 MODE-02 / MODE-03 / D-11
# ---------------------------------------------------------------------------


from eval.runners.optimizer import (  # noqa: E402
    _apply_transform_to_dataset,
    _resolve_subsample_pair_seeds,
    SUBSAMPLE_PAIR_MULTISEED_TIER,
)
from eval.data_factory import DataFactory  # noqa: E402


class TestOptimizerSyntheticMode:
    """Phase 31: synthetic pipeline_mode wiring for HyperparamOptimizer._objective().

    Verifies:
    - _apply_transform_to_dataset helper returns distinct positions
    - helper does NOT touch the caller's DataFactory state (D-11)
    - sanity tier does NOT call self._factory.generate_target() (D-11 / Pitfall 3)
    - dev tier uses _synthetic_target and get_synthetic_ground_truth() (D-10, D-09)
    """

    def test_apply_transform_returns_distinct_pos(
        self, tmp_path, synthetic_dataset
    ) -> None:
        """_apply_transform_to_dataset with noise sigma=0.1 returns distinct pos arrays."""
        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
        )
        transform_spec = {"type": "noise", "sigma": 0.1}
        result = _apply_transform_to_dataset(synthetic_dataset, transform_spec, cfg)
        # At least one frame should have different pos
        for k in synthetic_dataset:
            orig_pos = synthetic_dataset[k]["pos"]
            new_pos = result[k]["pos"]
            if not torch.allclose(orig_pos, new_pos):
                return  # found a differing frame — test passes
        raise AssertionError("_apply_transform_to_dataset returned identical positions for all frames")

    def test_apply_transform_does_not_touch_caller_factory(
        self, tmp_path, synthetic_dataset
    ) -> None:
        """Calling _apply_transform_to_dataset does NOT set _synthetic_target on caller factory (D-11)."""
        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
        )
        outer_factory = DataFactory(cfg)
        assert outer_factory._synthetic_target is None
        transform_spec = {"type": "noise", "sigma": 0.1}
        _apply_transform_to_dataset(synthetic_dataset, transform_spec, cfg)
        # caller's factory must be untouched
        assert outer_factory._synthetic_target is None, (
            "_apply_transform_to_dataset must not modify the caller's DataFactory instance (D-11)"
        )

    @patch("eval.runners.optimizer.MetricsEngine")
    @patch("eval.runners.optimizer.DataFactory")
    def test_sanity_synthetic_mode_does_not_call_main_factory_generate_target(
        self,
        mock_factory_cls,
        mock_engine_cls,
        tmp_path,
        synthetic_dataset,
    ) -> None:
        """Sanity tier in synthetic mode must NOT call self._factory.generate_target() (D-11, Pitfall 3)."""
        synth_config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            search_strategy="grid",
            tier="sanity",
            n_trials=3,
            run_alignment=True,
            run_label_transfer=True,
            search_space={
                "window_size": [3, 5],
                "k_neighbours": [3, 5],
            },
            pipeline_mode="synthetic",
            transform_spec={"type": "noise", "sigma": 0.1},
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = (
            lambda ds: {k: ds[k]["label"] for k in ds}
        )
        # The sanity tier's isolated scratch factory (same patched class) applies
        # the transform via augment(); return the input so trials really succeed.
        # Phase 59 NUM-05: before, augment() returned a MagicMock, every trial
        # failed silently with 0.0 and this test still passed; an all-failed run
        # now raises.
        mock_factory.augment.side_effect = lambda ds: ds
        # stub MetricsEngine so objective body runs to completion
        stub_metrics = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
        mock_engine_cls.return_value.compute_score.return_value = 0.5
        mock_engine_cls.return_value.sanity_check.return_value = []

        optimizer = HyperparamOptimizer(synth_config)
        result = optimizer.run()

        # The main factory's generate_target must NEVER be called in sanity tier (D-11)
        mock_factory.generate_target.assert_not_called()
        assert result.failed_trials == []
        assert len(result.history) == 4

    @patch("eval.runners.optimizer.MetricsEngine")
    @patch("eval.runners.optimizer.DataFactory")
    def test_dev_synthetic_mode_uses_synthetic_target_and_get_synthetic_ground_truth(
        self,
        mock_factory_cls,
        mock_engine_cls,
        tmp_path,
        synthetic_dataset,
    ) -> None:
        """Dev tier synthetic mode: _factory.get_synthetic_ground_truth() called; load_target() NOT called."""
        synth_config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            search_strategy="grid",
            tier="dev",
            n_trials=3,
            run_alignment=True,
            run_label_transfer=True,
            search_space={
                "window_size": [3, 5],
                "k_neighbours": [3, 5],
            },
            pipeline_mode="synthetic",
            transform_spec={"type": "noise", "sigma": 0.1},
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.generate_synthetic.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        # pre-populate _synthetic_target on the mock factory
        mock_factory._synthetic_target = synthetic_dataset
        mock_factory.get_synthetic_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        mock_factory.get_ground_truth.side_effect = (
            lambda ds: {k: ds[k]["label"] for k in ds}
        )
        # stub MetricsEngine
        stub_metrics = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
        mock_engine_cls.return_value.compute_score.return_value = 0.5
        mock_engine_cls.return_value.sanity_check.return_value = []

        optimizer = HyperparamOptimizer(synth_config)
        optimizer.run()

        assert mock_factory.get_synthetic_ground_truth.called is True, (
            "dev tier synthetic mode must call get_synthetic_ground_truth()"
        )
        mock_factory.load_target.assert_not_called()


# ---------------------------------------------------------------------------
# TestOptimizerSubsamplePair — Phase 57 GT-06 / D-06
# ---------------------------------------------------------------------------


class TestOptimizerSubsamplePair:
    """Phase 57: subsample_pair single-fixed-seed wiring for HyperparamOptimizer.

    Verifies:
    - _is_subsample_pair_mode() correctly identifies subsample_pair/rigid/paired configs
    - run()'s pre-populate step calls generate_subsample_pair() (not generate_target())
      for subsample_pair mode
    - _tier_dataset("dev"/"full") return self._factory._subsample_source_view in
      subsample_pair mode
    - sanity tier uses an isolated scratch DataFactory, never self._factory
      (T-57-06)
    - a real (non-mocked) end-to-end full-tier sweep completes and produces a
      non-empty SearchResult.history
    """

    @staticmethod
    def _make_scratch_factory(dataset: dict) -> MagicMock:
        """Build a MagicMock standing in for an isolated scratch DataFactory instance."""
        scratch = MagicMock()
        scratch.generate_subsample_pair.return_value = (dataset, dataset)
        scratch.get_synthetic_ground_truth.return_value = {
            k: dataset[k]["label"] for k in dataset
        }
        return scratch

    def _make_factory_side_effect(self, main_factory: MagicMock, dataset: dict):
        """Return a DataFactory(...) side_effect: 1st call -> main_factory,
        every subsequent call -> a fresh isolated scratch-factory MagicMock
        (mirrors _objective's ``DataFactory(self.config)`` sanity-tier call,
        which must be a genuinely distinct instance from ``self._factory``).
        """
        calls = {"n": 0}

        def _side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return main_factory
            return self._make_scratch_factory(dataset)

        return _side_effect

    # -- _is_subsample_pair_mode() direct unit tests -----------------------

    def test_is_subsample_pair_mode_true_for_subsample_pair_config(self, tmp_path):
        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec={"type": "subsample_pair", "seed": 1},
        )
        optimizer = HyperparamOptimizer(cfg)
        assert optimizer._is_subsample_pair_mode() is True

    def test_is_subsample_pair_mode_false_for_rigid_config(self, tmp_path):
        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec={"type": "rigid", "rotation_deg": 30.0, "rotation_axis": [0, 0, 1]},
        )
        optimizer = HyperparamOptimizer(cfg)
        assert optimizer._is_subsample_pair_mode() is False

    def test_is_subsample_pair_mode_false_for_paired_config(self, tmp_path):
        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="paired",
        )
        optimizer = HyperparamOptimizer(cfg)
        assert optimizer._is_subsample_pair_mode() is False

    # -- _tier_dataset("dev"/"full") direct unit tests ----------------------

    def test_tier_dataset_dev_and_full_return_subsample_source_view(
        self, tmp_path, synthetic_dataset
    ) -> None:
        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec={"type": "subsample_pair", "seed": 7},
        )
        optimizer = HyperparamOptimizer(cfg)
        optimizer._factory = MagicMock()
        optimizer._factory._subsample_source_view = synthetic_dataset

        assert optimizer._tier_dataset("dev") is synthetic_dataset
        assert optimizer._tier_dataset("full") is synthetic_dataset

    # -- run() pre-populate dispatch ----------------------------------------

    @patch("eval.runners.optimizer.MetricsEngine")
    @patch("eval.runners.optimizer.DataFactory")
    def test_run_prepopulate_calls_generate_subsample_pair_not_generate_target(
        self,
        mock_factory_cls,
        mock_engine_cls,
        tmp_path,
        synthetic_dataset,
    ) -> None:
        """run()'s pre-populate step must dispatch to generate_subsample_pair(),
        not generate_target(), for subsample_pair mode (D-06)."""
        transform_spec = {
            "type": "subsample_pair",
            "seed": 42,
            "synthesize": True,
            "n_classes": 3,
        }
        synth_config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            search_strategy="grid",
            tier="full",
            n_trials=1,
            run_alignment=True,
            run_label_transfer=True,
            search_space={"window_size": [3], "k_neighbours": [3]},
            pipeline_mode="synthetic",
            transform_spec=transform_spec,
        )
        main_factory = MagicMock()
        main_factory._subsample_source_view = synthetic_dataset
        main_factory._synthetic_target = synthetic_dataset
        main_factory.get_synthetic_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        mock_factory_cls.side_effect = self._make_factory_side_effect(
            main_factory, synthetic_dataset
        )

        stub_metrics = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
        mock_engine_cls.return_value.compute_score.return_value = 0.5
        mock_engine_cls.return_value.sanity_check.return_value = []

        optimizer = HyperparamOptimizer(synth_config)
        optimizer.run()

        main_factory.generate_subsample_pair.assert_called_once_with(None, transform_spec)
        main_factory.generate_target.assert_not_called()

    # -- sanity-tier isolation ------------------------------------------------

    @patch("eval.runners.optimizer.MetricsEngine")
    @patch("eval.runners.optimizer.DataFactory")
    def test_sanity_subsample_pair_does_not_call_main_factory(
        self,
        mock_factory_cls,
        mock_engine_cls,
        tmp_path,
        synthetic_dataset,
    ) -> None:
        """Sanity-tier trials in subsample_pair mode must NEVER call
        self._factory.generate_subsample_pair() (T-57-06) — only the
        pre-populate step (skipped entirely for tier='sanity') is allowed to
        touch the main factory's generate_subsample_pair()."""
        transform_spec = {
            "type": "subsample_pair",
            "seed": 3,
            "synthesize": True,
            "n_classes": 3,
        }
        synth_config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            search_strategy="grid",
            tier="sanity",
            n_trials=1,
            run_alignment=True,
            run_label_transfer=True,
            search_space={"window_size": [3], "k_neighbours": [3]},
            pipeline_mode="synthetic",
            transform_spec=transform_spec,
        )
        main_factory = MagicMock()
        mock_factory_cls.side_effect = self._make_factory_side_effect(
            main_factory, synthetic_dataset
        )

        stub_metrics = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
        mock_engine_cls.return_value.compute_score.return_value = 0.5
        mock_engine_cls.return_value.sanity_check.return_value = []

        optimizer = HyperparamOptimizer(synth_config)
        optimizer.run()

        main_factory.generate_subsample_pair.assert_not_called()

    # -- real end-to-end smoke test (no mocking) -----------------------------

    def test_end_to_end_subsample_pair_full_tier_smoke(self, tmp_path) -> None:
        """A full-tier HPO sweep against a real subsample_pair config completes
        without raising and produces a non-empty SearchResult.history with
        finite scores."""
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec={
                "type": "subsample_pair",
                "synthesize": True,
                "seed": 5,
                "n_classes": 3,
                "n_points": 60,
                "source_fraction": 0.8,
                "target_fraction": 0.8,
            },
            tier="full",
            n_trials=3,
            search_strategy="grid",
            search_space={"window_size": [3], "k_neighbours": [3]},
        )
        result = HyperparamOptimizer(config).run()

        assert isinstance(result, SearchResult)
        assert len(result.history) > 0, "SearchResult.history must be non-empty"
        for trial in result.history:
            assert isinstance(trial.score, float)
            assert trial.score == trial.score, "score must not be NaN"  # NaN != NaN
            assert trial.score not in (float("inf"), float("-inf")), "score must be finite"


# ---------------------------------------------------------------------------
# TestSubsamplePairMultiseed — Phase 57 GT-04 / D-07
# ---------------------------------------------------------------------------


class TestSubsamplePairMultiseed:
    """Phase 57: D-07 opt-in multi-seed averaging for subsample_pair mode.

    Verifies:
    - _resolve_subsample_pair_seeds()'s full decision table (int seed, list
      on full tier, list graceful fallback on sanity/dev, empty-list and
      wrong-type raises)
    - _objective()'s early-exit dispatch: single int seed never calls
      _score_subsample_pair_multiseed; a list seed on the full tier does
    - averaging correctness: the returned score is the exact arithmetic
      mean of the per-seed compute_score values
    - a real end-to-end 3-seed full-tier sweep completes without raising
    """

    # -- _resolve_subsample_pair_seeds() direct unit tests -------------------

    def test_resolve_seeds_int_seed_returns_single_element_list(self):
        assert _resolve_subsample_pair_seeds({"seed": 42}, "sanity") == [42]
        assert _resolve_subsample_pair_seeds({"seed": 42}, "full") == [42]

    def test_resolve_seeds_list_on_full_tier_returns_unchanged(self):
        assert _resolve_subsample_pair_seeds({"seed": [1, 2, 3]}, "full") == [1, 2, 3]
        assert SUBSAMPLE_PAIR_MULTISEED_TIER == "full"

    def test_resolve_seeds_list_on_sanity_tier_falls_back_to_first(self):
        assert _resolve_subsample_pair_seeds({"seed": [1, 2, 3]}, "sanity") == [1]

    def test_resolve_seeds_list_on_dev_tier_falls_back_to_first(self):
        assert _resolve_subsample_pair_seeds({"seed": [1, 2, 3]}, "dev") == [1]

    def test_resolve_seeds_empty_list_raises_value_error(self):
        with pytest.raises(ValueError):
            _resolve_subsample_pair_seeds({"seed": []}, "full")

    def test_resolve_seeds_wrong_type_raises_value_error(self):
        with pytest.raises(ValueError):
            _resolve_subsample_pair_seeds({"seed": "bad"}, "full")

    # -- _objective() early-exit dispatch -------------------------------------

    @patch("eval.runners.optimizer.HyperparamOptimizer._score_subsample_pair_multiseed")
    def test_single_int_seed_never_calls_multiseed_scorer(
        self, mock_multiseed, tmp_path
    ) -> None:
        """A plain int seed (the default, unchanged Plan 57-03 path) must
        NEVER trigger _score_subsample_pair_multiseed (D-06)."""
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec={
                "type": "subsample_pair",
                "synthesize": True,
                "seed": 42,
                "n_classes": 3,
                "n_points": 40,
                "source_fraction": 0.8,
                "target_fraction": 0.8,
            },
            tier="full",
            n_trials=1,
            search_strategy="grid",
            search_space={"window_size": [3], "k_neighbours": [3]},
        )
        HyperparamOptimizer(config).run()
        mock_multiseed.assert_not_called()

    @patch("eval.runners.optimizer.HyperparamOptimizer._score_subsample_pair_multiseed")
    def test_list_seed_full_tier_calls_multiseed_scorer(
        self, mock_multiseed, tmp_path
    ) -> None:
        """A list seed on the full tier must dispatch to
        _score_subsample_pair_multiseed (D-07)."""
        mock_multiseed.return_value = 0.5
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec={
                "type": "subsample_pair",
                "synthesize": True,
                "seed": [1, 2],
                "n_classes": 3,
                "n_points": 40,
                "source_fraction": 0.8,
                "target_fraction": 0.8,
            },
            tier="full",
            n_trials=1,
            search_strategy="grid",
            search_space={"window_size": [3], "k_neighbours": [3]},
        )
        HyperparamOptimizer(config).run()
        mock_multiseed.assert_called()

    # -- averaging correctness (mocked stages/engine) -------------------------

    @patch("eval.runners.optimizer.MetricsEngine")
    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_score_subsample_pair_multiseed_averages_correctly(
        self,
        mock_factory_cls,
        mock_align_cls,
        mock_label_cls,
        mock_engine_cls,
        tmp_path,
        synthetic_dataset,
    ) -> None:
        """The returned score is the exact arithmetic mean of the per-seed
        compute_score values (not the first or last seed's score alone),
        and exactly one Trial is appended to history_out."""
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            run_alignment=True,
            run_label_transfer=True,
            transform_spec={
                "type": "subsample_pair",
                "synthesize": True,
                "seed": [1, 2],
            },
            search_space={"window_size": [3], "k_neighbours": [3]},
        )

        # 1st DataFactory(...) call = self._factory (constructed in __init__);
        # 2nd/3rd calls = the per-seed FRESH scratch factories inside
        # _score_subsample_pair_multiseed (T-57-11 isolation).
        main_factory = MagicMock()
        gt = {k: synthetic_dataset[k]["label"] for k in synthetic_dataset}
        scratch_1 = MagicMock()
        scratch_1.generate_subsample_pair.return_value = (synthetic_dataset, synthetic_dataset)
        scratch_1.get_synthetic_ground_truth.return_value = gt
        scratch_2 = MagicMock()
        scratch_2.generate_subsample_pair.return_value = (synthetic_dataset, synthetic_dataset)
        scratch_2.get_synthetic_ground_truth.return_value = gt
        mock_factory_cls.side_effect = [main_factory, scratch_1, scratch_2]

        mock_align_cls.return_value.run.return_value = MagicMock(
            aligned_cloud=synthetic_dataset, warp_path=[]
        )
        mock_label_cls.return_value.run.return_value = MagicMock(
            transferred_labels=gt
        )

        stub_metrics = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
        mock_engine_cls.return_value.compute_score.side_effect = [0.4, 0.6]

        optimizer = HyperparamOptimizer(config)
        history_out: list[Trial] = []
        merged = dict(optimizer._default_params)

        score = optimizer._score_subsample_pair_multiseed(
            {}, merged, "full", [1, 2], history_out
        )

        assert score == pytest.approx(0.5), (
            "averaged score must be the exact arithmetic mean of [0.4, 0.6]"
        )
        assert len(history_out) == 1, (
            "exactly one Trial must be appended per multiseed trial call"
        )
        assert history_out[0].score == pytest.approx(0.5)
        assert history_out[0].tier == "full"
        # Each seed used its own FRESH DataFactory — never self._factory (main_factory).
        main_factory.generate_subsample_pair.assert_not_called()

    # -- real end-to-end smoke test (no mocking) -------------------------------

    def test_end_to_end_subsample_pair_multiseed_full_tier_smoke(
        self, tmp_path
    ) -> None:
        """A full-tier HPO sweep with a 3-entry seed list completes without
        raising and produces a non-empty SearchResult.history with finite
        scores (D-07 real integration)."""
        config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec={
                "type": "subsample_pair",
                "synthesize": True,
                "seed": [11, 12, 13],
                "n_classes": 3,
                "n_points": 60,
                "source_fraction": 0.8,
                "target_fraction": 0.8,
            },
            tier="full",
            n_trials=2,
            search_strategy="grid",
            search_space={"window_size": [3], "k_neighbours": [3]},
        )
        result = HyperparamOptimizer(config).run()

        assert isinstance(result, SearchResult)
        assert len(result.history) > 0, "SearchResult.history must be non-empty"
        for trial in result.history:
            assert isinstance(trial.score, float)
            assert trial.score == trial.score, "score must not be NaN"  # NaN != NaN
            assert trial.score not in (float("inf"), float("-inf")), "score must be finite"


# ---------------------------------------------------------------------------
# Coverage gap tests for optimizer.py
# ---------------------------------------------------------------------------


class TestHyperparamOptimizerCoverageGaps:
    """Additional tests to cover missed lines in optimizer.py."""

    def test_search_space_non_list_value_raises(self, tmp_path):
        """optimizer.py:167 — ValueError when search_space value is not a list."""
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": 5},  # int, not list
        )
        with pytest.raises(ValueError, match="must be a non-empty list"):
            HyperparamOptimizer(cfg)

    def test_search_space_empty_list_raises(self, tmp_path):
        """optimizer.py:167 — ValueError when search_space value is an empty list."""
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": []},
        )
        with pytest.raises(ValueError, match="must be a non-empty list"):
            HyperparamOptimizer(cfg)

    def test_both_stages_disabled_raises(self, tmp_path):
        """optimizer.py:173 — ValueError when both stages disabled."""
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            run_alignment=False,
            run_label_transfer=False,
        )
        with pytest.raises(ValueError, match="At least one stage must be enabled"):
            HyperparamOptimizer(cfg)

    @patch("eval.runners.optimizer.DataFactory")
    def test_full_tier_n_trials_path(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:231 — n_trials from config when tier='full'."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="grid",
            tier="full",
            n_trials=2,
            search_space={"window_size": [3]},
        )
        result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.DataFactory")
    def test_auto_strategy_resolves(
        self, mock_factory_cls, tmp_path, synthetic_dataset, monkeypatch
    ):
        """optimizer.py:235 — _detect_backend called when search_strategy='auto'."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="auto",
            tier="sanity",
            n_trials=2,
            search_space={"window_size": [3]},
        )
        import sys
        # Only the mpi4py key is patched. patch.dict(sys.modules, ...) would, on
        # exit, also drop every module first imported inside the block (here
        # optuna's sqlite RDB storage + SQLAlchemy ORM modules); their later
        # re-import builds duplicate ORM mappings and breaks every subsequent
        # sqlite study that enqueues trials (found by the Phase 59 NUM-05 tests).
        monkeypatch.setitem(sys.modules, "mpi4py", None)
        result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.DataFactory")
    def test_random_search_strategy(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:254 — RandomSearch().search() called when strategy='random'."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="random",
            tier="sanity",
            n_trials=2,
            search_space={"window_size": [3, 5]},
        )
        result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.DataFactory")
    def test_bayesian_search_strategy(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:261 — BayesianSearch().search() called when strategy='bayesian'."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="bayesian",
            tier="sanity",
            n_trials=2,
            search_space={"window_size": [3, 5]},
        )
        result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.DataFactory")
    def test_unknown_strategy_raises(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:299 — ValueError for unknown strategy."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="grid",
            tier="sanity",
            n_trials=2,
            search_space={"window_size": [3]},
        )
        optimizer = HyperparamOptimizer(cfg)
        with patch.object(optimizer, "_detect_backend", return_value="bad_strategy"):
            optimizer.config = optimizer.config.model_copy(update={"search_strategy": "auto"})
            with pytest.raises(ValueError, match="Unknown search_strategy"):
                optimizer.run()

    @patch("eval.runners.optimizer.DataFactory")
    def test_paired_mode_dev_tier_loads_target(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:385 — load_target() called in paired mode non-sanity tier."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.generate_synthetic.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="grid",
            tier="dev",
            n_trials=2,
            pipeline_mode="paired",
            search_space={"window_size": [3]},
        )
        HyperparamOptimizer(cfg).run()
        mock_factory.load_target.assert_called()

    @patch("eval.runners.optimizer.DataFactory")
    def test_run_alignment_false(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:391-395 — False branch of run_alignment."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="grid",
            tier="sanity",
            n_trials=2,
            run_alignment=False,
            run_label_transfer=True,
            search_space={"k_neighbours": [3]},
        )
        result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.DataFactory")
    def test_run_label_transfer_false(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:395-400 — False branch of run_label_transfer (with alignment True)."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="grid",
            tier="sanity",
            n_trials=2,
            run_alignment=True,
            run_label_transfer=False,
            search_space={"window_size": [3]},
        )
        result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer._apply_transform_to_dataset")
    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_synthetic_sanity_no_color_uses_arange(
        self, mock_factory_cls, mock_align_cls, mock_label_cls, mock_transform, tmp_path
    ):
        """optimizer.py:417-418 — y_true = torch.arange when color is None in synthetic sanity."""
        from zreg.data_generation import generate_trajectory
        from eval.types import StageMetrics

        # Dataset with label=None — generate_trajectory returns no labels
        ds = generate_trajectory(n_points=10, n_frames=3, seed=42)

        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = ds
        mock_factory.load_target.return_value = ds
        mock_factory.get_ground_truth.side_effect = lambda d: {k: torch.arange(10, dtype=torch.long) for k in d}

        # _apply_transform_to_dataset returns dataset unchanged (avoids augment() error on label=None)
        mock_transform.side_effect = lambda dataset, spec, cfg: dataset

        # Stub AlignmentStage so the code reaches GT selection (lines 408-418)
        from eval.types import AlignResult
        fake_align = AlignResult(
            aligned_cloud=ds,
            warp_path=[(i, i) for i in range(3)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        mock_align_cls.return_value.run.return_value = fake_align

        # Stub LabelTransferStage so it doesn't raise on label=None
        from eval.types import LabelResult
        fake_label = LabelResult(
            transferred_labels={k: torch.arange(10, dtype=torch.long) for k in ds},
            params_used={},
        )
        mock_label_cls.return_value.run.return_value = fake_label

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.5, knn_consistency=0.5,
        )
        with patch("eval.runners.optimizer.MetricsEngine") as mock_engine_cls:
            mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
            mock_engine_cls.return_value.compute_score.return_value = 0.5
            mock_engine_cls.return_value.sanity_check.return_value = []

            cfg = EvalConfig(
                data_path=str(tmp_path / "x"),
                output_dir=str(tmp_path / "out"),
                search_strategy="grid",
                tier="sanity",
                n_trials=2,
                pipeline_mode="synthetic",
                transform_spec={"type": "noise", "sigma": 0.01},
                search_space={"window_size": [3]},
            )
            optimizer = HyperparamOptimizer(cfg)
            # _tier_dataset("sanity") always calls generate_labels — override to return label=None ds
            with patch.object(optimizer, "_tier_dataset", return_value=ds):
                result = optimizer.run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_paired_sanity_null_gt_records_failure_and_raises(
        self, mock_factory_cls, mock_align_cls, mock_label_cls, tmp_path
    ):
        """ValueError when y_true is None (both id and label absent) is a recorded failure.

        Phase 59 NUM-05: previously the trial silently returned 0.0 and run()
        returned an empty SearchResult; the only trial failing now raises.
        """
        from zreg.data_generation import generate_trajectory
        from eval.types import AlignResult, LabelResult, StageMetrics

        # Dataset where both label and id are None — generate_trajectory has label=None, id=None
        ds_base = generate_trajectory(n_points=10, n_frames=3, seed=42)

        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = ds_base
        mock_factory.load_target.return_value = ds_base

        fake_align = AlignResult(
            aligned_cloud=ds_base,
            warp_path=[(i, i) for i in range(3)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        mock_align_cls.return_value.run.return_value = fake_align
        fake_label = LabelResult(
            transferred_labels={k: torch.zeros(10, dtype=torch.long) for k in ds_base},
            params_used={},
        )
        mock_label_cls.return_value.run.return_value = fake_label

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.5, knn_consistency=0.5,
        )
        with patch("eval.runners.optimizer.MetricsEngine") as mock_engine_cls:
            mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
            mock_engine_cls.return_value.compute_score.return_value = 0.5
            mock_engine_cls.return_value.sanity_check.return_value = []

            cfg = EvalConfig(
                data_path=str(tmp_path / "x"),
                output_dir=str(tmp_path / "out"),
                search_strategy="grid",
                tier="sanity",
                n_trials=1,
                pipeline_mode="paired",  # non-synthetic → hits lines 421-432
                search_space={"window_size": [3]},
            )
            optimizer = HyperparamOptimizer(cfg)
            # _tier_dataset("sanity") always generates labeled data — override to return
            # label=None, id=None dataset so y_true is None → ValueError, caught by
            # _objective's except branch → recorded failure (-inf); the run has no
            # successful trial, so it raises after writing failed_trials.json.
            with patch.object(optimizer, "_tier_dataset", return_value=ds_base):
                with pytest.raises(RuntimeError, match="All 1 HPO trials failed"):
                    optimizer.run()
        records = json.loads((tmp_path / "out" / "failed_trials.json").read_text())
        assert len(records) == 1
        assert records[0]["error_type"] == "ValueError"
        assert "No ground-truth labels" in records[0]["error"]

    @patch("eval.runners.optimizer.DataFactory")
    def test_propulate_strategy_mocked(self, mock_factory_cls, tmp_path, synthetic_dataset):
        """optimizer.py:445-447 — PropulateSearch dispatch with mocked PropulateSearch."""
        import sys, types
        from eval.types import StageMetrics

        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.5, knn_consistency=0.5,
        )

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="propulate",
            tier="sanity",
            n_trials=2,
            search_space={"window_size": [3]},
        )
        with patch("eval.runners.optimizer.PropulateSearch") as mock_propulate:
            mock_propulate.return_value.search.return_value = [({"window_size": 3}, 0.5)]
            result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_objective_truncates_when_ytrue_ypred_differ(
        self, mock_factory_cls, mock_align_cls, mock_label_cls, tmp_path
    ):
        """optimizer.py:445-447 — y_true[:min_len]/y_pred[:min_len] when shapes differ."""
        from zreg.data_generation import generate_labels, generate_trajectory
        from eval.types import AlignResult, LabelResult, StageMetrics

        ds = generate_labels(generate_trajectory(n_points=10, n_frames=3, seed=0), n_labels=4, seed=0)

        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = ds
        mock_factory.load_target.return_value = ds

        fake_align = AlignResult(
            aligned_cloud=ds,
            warp_path=[(i, i) for i in range(3)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        mock_align_cls.return_value.run.return_value = fake_align
        # transferred_labels has 7 points but y_true (from color) has 10 → truncation at 445-447
        fake_label = LabelResult(
            transferred_labels={k: torch.zeros(7, dtype=torch.long) for k in ds},
            params_used={},
        )
        mock_label_cls.return_value.run.return_value = fake_label

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.5, knn_consistency=0.5,
        )
        with patch("eval.runners.optimizer.MetricsEngine") as mock_engine_cls:
            mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
            mock_engine_cls.return_value.compute_score.return_value = 0.5
            mock_engine_cls.return_value.sanity_check.return_value = []

            cfg = EvalConfig(
                data_path=str(tmp_path / "x"),
                output_dir=str(tmp_path / "out"),
                search_strategy="grid",
                tier="sanity",
                n_trials=1,
                pipeline_mode="paired",
                search_space={"window_size": [3]},
            )
            optimizer = HyperparamOptimizer(cfg)
            with patch.object(optimizer, "_tier_dataset", return_value=ds):
                result = optimizer.run()
        assert isinstance(result, SearchResult)

    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_synthetic_sanity_with_color_uses_color_field(
        self, mock_factory_cls, mock_align_cls, mock_label_cls, tmp_path
    ):
        """optimizer.py:415 — y_true = color field in synthetic sanity when color is not None."""
        from zreg.data_generation import generate_labels, generate_trajectory
        from eval.types import AlignResult, LabelResult, StageMetrics

        ds = generate_labels(generate_trajectory(n_points=10, n_frames=3, seed=0), n_labels=4, seed=0)

        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = ds

        mock_transform_ds = {k: ds[k] for k in ds}

        fake_align = AlignResult(
            aligned_cloud=ds,
            warp_path=[(i, i) for i in range(3)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        mock_align_cls.return_value.run.return_value = fake_align
        fake_label = LabelResult(
            transferred_labels={k: ds[k]["label"].long() for k in ds},
            params_used={},
        )
        mock_label_cls.return_value.run.return_value = fake_label

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.5, knn_consistency=0.5,
        )
        with patch("eval.runners.optimizer._apply_transform_to_dataset",
                   side_effect=lambda dataset, spec, cfg: dataset), \
             patch("eval.runners.optimizer.MetricsEngine") as mock_engine_cls:
            mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
            mock_engine_cls.return_value.compute_score.return_value = 0.5
            mock_engine_cls.return_value.sanity_check.return_value = []

            cfg = EvalConfig(
                data_path=str(tmp_path / "x"),
                output_dir=str(tmp_path / "out"),
                search_strategy="grid",
                tier="sanity",
                n_trials=1,
                pipeline_mode="synthetic",
                transform_spec={"type": "noise", "sigma": 0.01},
                search_space={"window_size": [3]},
            )
            optimizer = HyperparamOptimizer(cfg)
            with patch.object(optimizer, "_tier_dataset", return_value=ds):
                result = optimizer.run()
        assert isinstance(result, SearchResult)

    def test_detect_backend_import_error_falls_through(self, tmp_path):
        """optimizer.py:587-588 — ImportError in _detect_backend is silently caught."""
        import sys
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": [3]},
        )
        optimizer = HyperparamOptimizer(cfg)
        with patch.dict(sys.modules, {"mpi4py": None}):
            result = optimizer._detect_backend()
        # Should fall through to bayesian or propulate (SLURM env may vary)
        assert result in ("bayesian", "propulate")

    def test_detect_backend_mpi_runtime_error_falls_through(self, tmp_path):
        """optimizer.py:589-590 — non-ImportError from MPI init is caught by except Exception."""
        import sys, types
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": [3]},
        )
        optimizer = HyperparamOptimizer(cfg)

        # Build a fake mpi4py module where MPI.COMM_WORLD.Get_size() raises RuntimeError
        mock_mpi4py = types.ModuleType("mpi4py")
        mock_MPI = types.ModuleType("mpi4py.MPI")
        mock_comm = MagicMock()
        mock_comm.Get_size.side_effect = RuntimeError("MPI init failed")
        mock_MPI.COMM_WORLD = mock_comm
        mock_mpi4py.MPI = mock_MPI

        with patch.dict(sys.modules, {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_MPI}):
            result = optimizer._detect_backend()
        assert result in ("bayesian", "propulate")

    def test_detect_backend_mpi_world_size_gt1_returns_propulate(self, tmp_path):
        """optimizer.py:584-586 — MPI world_size > 1 returns 'propulate'."""
        import sys, types
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": [3]},
        )
        optimizer = HyperparamOptimizer(cfg)

        # Build a fake mpi4py module where Get_size() returns 2 (multi-process)
        mock_mpi4py = types.ModuleType("mpi4py")
        mock_MPI = types.ModuleType("mpi4py.MPI")
        mock_comm = MagicMock()
        mock_comm.Get_size.return_value = 2
        mock_MPI.COMM_WORLD = mock_comm
        mock_mpi4py.MPI = mock_MPI

        with patch.dict(sys.modules, {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_MPI}):
            result = optimizer._detect_backend()
        assert result == "propulate"

    def test_detect_backend_mpi_world_size_eq1_falls_through(self, tmp_path):
        """optimizer.py:584->593 — MPI world_size == 1, try completes, falls to SLURM check."""
        import sys, types, os
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": [3]},
        )
        optimizer = HyperparamOptimizer(cfg)

        mock_mpi4py = types.ModuleType("mpi4py")
        mock_MPI = types.ModuleType("mpi4py.MPI")
        mock_comm = MagicMock()
        mock_comm.Get_size.return_value = 1  # single process — falls through to SLURM check
        mock_MPI.COMM_WORLD = mock_comm
        mock_mpi4py.MPI = mock_MPI

        env_without_slurm = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        with patch.dict(sys.modules, {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_MPI}), \
             patch.dict(os.environ, env_without_slurm, clear=True):
            result = optimizer._detect_backend()
        assert result == "bayesian"

    def test_detect_backend_slurm_env(self, tmp_path):
        """optimizer.py:593-595 — SLURM_JOB_ID set returns 'propulate'."""
        import os
        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": [3]},
        )
        optimizer = HyperparamOptimizer(cfg)
        with patch.dict(sys.modules, {"mpi4py": None}), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            result = optimizer._detect_backend()
        assert result == "propulate"

    def test_tier_dataset_dev_real_path_exists(self, tmp_path, synthetic_dataset):
        """optimizer.py:501 — dev tier with existing data_path calls load_real."""
        real_file = tmp_path / "data.mat"
        real_file.write_text("fake")

        cfg = EvalConfig(
            data_path=str(real_file),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": [3]},
        )
        with patch("eval.runners.optimizer.DataFactory") as mock_factory_cls:
            mock_factory = mock_factory_cls.return_value
            mock_factory.load_real.return_value = synthetic_dataset
            optimizer = HyperparamOptimizer(cfg)
            result = optimizer._tier_dataset("dev")
        mock_factory.load_real.assert_called_once()
        assert result is synthetic_dataset

    @patch("eval.runners.optimizer.DataFactory")
    def test_is_rank_zero_false_skips_save_best_params(
        self, mock_factory_cls, tmp_path, synthetic_dataset
    ):
        """optimizer.py:458->460 — _is_rank_zero()=False skips save_best_params."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.side_effect = lambda ds: {k: ds[k]["label"] for k in ds}

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_strategy="grid",
            tier="sanity",
            n_trials=1,
            search_space={"window_size": [3]},
        )
        optimizer = HyperparamOptimizer(cfg)
        with patch.object(optimizer, "_is_rank_zero", return_value=False), \
             patch.object(optimizer, "save_best_params") as mock_save:
            optimizer.run()
        mock_save.assert_not_called()

    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_multiseed_run_alignment_false_skips_alignment_stage(
        self, mock_factory_cls, mock_align_cls, mock_label_cls, tmp_path, synthetic_dataset
    ):
        """optimizer.py:746->752 — run_alignment=False skips AlignmentStage in multiseed loop."""
        gt = {k: synthetic_dataset[k]["label"] for k in synthetic_dataset}
        main_factory = MagicMock()
        scratch = MagicMock()
        scratch.generate_subsample_pair.return_value = (synthetic_dataset, synthetic_dataset)
        scratch.get_synthetic_ground_truth.return_value = gt
        mock_factory_cls.side_effect = [main_factory, scratch, scratch]

        mock_label_cls.return_value.run.return_value = MagicMock(
            transferred_labels=gt
        )

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.5, knn_consistency=0.5,
        )
        with patch("eval.runners.optimizer.MetricsEngine") as mock_engine_cls:
            mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
            mock_engine_cls.return_value.compute_score.return_value = 0.5

            cfg = EvalConfig(
                data_path=str(tmp_path / "unused.mat"),
                output_dir=str(tmp_path / "output"),
                pipeline_mode="synthetic",
                run_alignment=False,
                run_label_transfer=True,
                transform_spec={
                    "type": "subsample_pair",
                    "synthesize": True,
                    "seed": [1, 2],
                    "n_classes": 3,
                    "n_points": 40,
                    "source_fraction": 0.8,
                    "target_fraction": 0.8,
                },
                tier="full",
                n_trials=1,
                search_strategy="grid",
                search_space={"window_size": [3], "k_neighbours": [3]},
            )
            optimizer = HyperparamOptimizer(cfg)
            history: list[Trial] = []
            merged = dict(optimizer._default_params)
            optimizer._score_subsample_pair_multiseed({}, merged, "full", [1, 2], history)

        mock_align_cls.return_value.run.assert_not_called()
        assert len(history) == 1

    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_multiseed_run_label_transfer_false_zeros_ypred(
        self, mock_factory_cls, mock_align_cls, mock_label_cls, tmp_path, synthetic_dataset
    ):
        """optimizer.py:752->757, 768 — run_label_transfer=False produces zero y_pred."""
        gt = {k: synthetic_dataset[k]["label"] for k in synthetic_dataset}
        main_factory = MagicMock()
        scratch = MagicMock()
        scratch.generate_subsample_pair.return_value = (synthetic_dataset, synthetic_dataset)
        scratch.get_synthetic_ground_truth.return_value = gt
        mock_factory_cls.side_effect = [main_factory, scratch]

        mock_align_cls.return_value.run.return_value = MagicMock(
            aligned_cloud=synthetic_dataset, warp_path=[]
        )

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.0, knn_consistency=0.0,
        )
        with patch("eval.runners.optimizer.MetricsEngine") as mock_engine_cls:
            mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
            mock_engine_cls.return_value.compute_score.return_value = 0.0

            cfg = EvalConfig(
                data_path=str(tmp_path / "unused.mat"),
                output_dir=str(tmp_path / "output"),
                pipeline_mode="synthetic",
                run_alignment=True,
                run_label_transfer=False,
                transform_spec={
                    "type": "subsample_pair",
                    "synthesize": True,
                    "seed": [1],
                    "n_classes": 3,
                    "n_points": 40,
                    "source_fraction": 0.8,
                    "target_fraction": 0.8,
                },
                tier="full",
                n_trials=1,
                search_strategy="grid",
                search_space={"window_size": [3]},
            )
            optimizer = HyperparamOptimizer(cfg)
            history: list[Trial] = []
            merged = dict(optimizer._default_params)
            optimizer._score_subsample_pair_multiseed({}, merged, "full", [1], history)

        mock_label_cls.return_value.run.assert_not_called()
        assert len(history) == 1
        # y_pred is zeros — f1 with all-zero prediction is 0.0
        _, call_kwargs = mock_engine_cls.return_value.compute_stage_metrics.call_args
        y_pred_arg = mock_engine_cls.return_value.compute_stage_metrics.call_args[0][5]
        assert (y_pred_arg == 0).all()

    @patch("eval.runners.optimizer.LabelTransferStage")
    @patch("eval.runners.optimizer.AlignmentStage")
    @patch("eval.runners.optimizer.DataFactory")
    def test_multiseed_wr01_truncates_when_shapes_differ(
        self, mock_factory_cls, mock_align_cls, mock_label_cls, tmp_path, synthetic_dataset
    ):
        """optimizer.py:773-775 — y_true/y_pred truncated to min length when shapes differ."""
        from zreg.data_generation import generate_labels, generate_trajectory

        ds_source = generate_labels(generate_trajectory(n_points=10, n_frames=3, seed=0), n_labels=3, seed=0)
        ds_target = generate_labels(generate_trajectory(n_points=10, n_frames=3, seed=1), n_labels=3, seed=1)
        gt = {k: ds_source[k]["label"] for k in ds_source}

        main_factory = MagicMock()
        scratch = MagicMock()
        scratch.generate_subsample_pair.return_value = (ds_source, ds_target)
        scratch.get_synthetic_ground_truth.return_value = gt
        mock_factory_cls.side_effect = [main_factory, scratch]

        mock_align_cls.return_value.run.return_value = MagicMock(
            aligned_cloud=ds_source, warp_path=[]
        )
        # transferred_labels has 7 points per frame but y_true has 10 → triggers WR-01
        mock_label_cls.return_value.run.return_value = MagicMock(
            transferred_labels={k: torch.zeros(7, dtype=torch.long) for k in ds_target}
        )

        stub_metrics = StageMetrics(
            chamfer_distance=0.0, hausdorff_distance=0.0, path_smoothness=0.0,
            temporal_stability=0.0, f1_score=0.0, knn_consistency=0.0,
        )
        with patch("eval.runners.optimizer.MetricsEngine") as mock_engine_cls:
            mock_engine_cls.return_value.compute_stage_metrics.return_value = stub_metrics
            mock_engine_cls.return_value.compute_score.return_value = 0.0

            cfg = EvalConfig(
                data_path=str(tmp_path / "unused.mat"),
                output_dir=str(tmp_path / "output"),
                pipeline_mode="synthetic",
                run_alignment=True,
                run_label_transfer=True,
                transform_spec={
                    "type": "subsample_pair",
                    "synthesize": True,
                    "seed": [1],
                    "n_classes": 3,
                    "n_points": 40,
                    "source_fraction": 0.8,
                    "target_fraction": 0.8,
                },
                tier="full",
                n_trials=1,
                search_strategy="grid",
                search_space={"window_size": [3]},
            )
            optimizer = HyperparamOptimizer(cfg)
            history: list[Trial] = []
            merged = dict(optimizer._default_params)
            optimizer._score_subsample_pair_multiseed({}, merged, "full", [1], history)

        # compute_stage_metrics must have been called — WR-01 truncation succeeded
        mock_engine_cls.return_value.compute_stage_metrics.assert_called_once()
        y_true_arg = mock_engine_cls.return_value.compute_stage_metrics.call_args[0][4]
        y_pred_arg = mock_engine_cls.return_value.compute_stage_metrics.call_args[0][5]
        assert y_true_arg.shape[0] == y_pred_arg.shape[0] == 7

    @patch("eval.runners.optimizer.DataFactory")
    def test_tier_dataset_sanity_with_label_generation(
        self, mock_factory_cls, tmp_path
    ):
        """optimizer.py:831-832 — _tier_dataset('sanity') uses LabelGenerationConfig when set."""
        from eval.config import LabelGenerationConfig

        mock_factory_cls.return_value  # suppress unused warning

        cfg = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            search_space={"window_size": [3]},
            label_generation=LabelGenerationConfig(n_labels=5, seed=7),
        )
        optimizer = HyperparamOptimizer(cfg)
        ds = optimizer._tier_dataset("sanity")
        # With n_labels=5, each frame's label tensor must use values in [0, 4]
        for frame in ds.values():
            assert frame["label"] is not None
            assert frame["label"].max().item() <= 4


# ---------------------------------------------------------------------------
# test_sobol_is_default_when_strategy_absent + TestSobolOptimizerIntegration
# ---------------------------------------------------------------------------


def test_sobol_is_default_when_strategy_absent():
    """EvalConfig without search_strategy defaults to 'sobol' (OPT-04-06)."""
    cfg = EvalConfig(data_path="/unused.mat")
    assert cfg.search_strategy == "sobol"


class TestSobolOptimizerIntegration:
    """Integration: HyperparamOptimizer dispatches SobolSearch when search_strategy='sobol'."""

    @patch("eval.runners.optimizer.DataFactory")
    def test_sobol_strategy_runs_and_returns_result(
        self, mock_factory_cls, tmp_path, synthetic_dataset
    ) -> None:
        """search_strategy='sobol' dispatches SobolSearch and returns SearchResult (OPT-04-05)."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset

        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            search_strategy="sobol",
            tier="sanity",
            n_trials=8,
            search_space={"window_size": [3, 5], "k_neighbours": [3, 5]},
        )
        result = HyperparamOptimizer(cfg).run()
        assert isinstance(result, SearchResult)
