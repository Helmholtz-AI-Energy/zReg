"""Tests for eval.runners.eval_runner.EvaluationRunner (FRAME-07).

Covers Phase 21 FRAME-07 gate criteria:
  - FRAME-07-G1: run() with fixed params returns complete EvalReport with
    all 6 metric fields, non-empty aggregated_metrics, per_dataset["dataset"]
  - FRAME-07-G2: 1-frame dataset triggers non-empty sanity_flags containing
    "single-frame" substring
  - FRAME-07-G3: eval_report.json written to output_dir; valid JSON with
    expected top-level keys

Additional coverage:
  - D-06 ValueError when both stage flags are False
  - D-01 shallow copy of params
  - Conditional stage skipping (run_alignment=False / run_label_transfer=False)
"""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_labels, generate_trajectory

import torch

from eval.config import EvalConfig
from eval.metrics import MetricsEngine
from eval.runners import EvaluationRunner
from eval.types import AlignResult, EvalReport, LabelResult, StageMetrics


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_config(tmp_path) -> EvalConfig:
    """EvalConfig with both stages enabled and output_dir inside tmp_path."""
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def synthetic_dataset() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with color labels for runner tests.

    generate_labels() populates the 'id' field (torch.long, shape (N,))
    so DataFactory.get_ground_truth() returns valid id tensors.
    """
    traj = generate_trajectory(n_points=20, n_frames=3, seed=0)
    return generate_labels(traj, n_classes=4, seed=0)


@pytest.fixture
def single_frame_dataset() -> dict[int, zRegPointCloud]:
    """1-frame dataset that triggers MetricsEngine.sanity_check 'single-frame' flag."""
    traj = generate_trajectory(n_points=20, n_frames=1, seed=0)
    return generate_labels(traj, n_classes=4, seed=0)


@pytest.fixture
def full_params() -> dict:
    """Valid params containing all 9 keys (5 alignment + 4 label-transfer)."""
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


@pytest.fixture
def fake_align_result(synthetic_dataset) -> AlignResult:
    """AlignResult constructed without DTW execution (for fast unit tests)."""
    return AlignResult(
        aligned_cloud=synthetic_dataset,
        warp_path=[(0, 0), (1, 1), (2, 2)],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={
            "window_size": 10,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
        },
    )


@pytest.fixture
def fake_label_result(synthetic_dataset) -> LabelResult:
    """LabelResult constructed without kNN execution (for fast unit tests)."""
    return LabelResult(
        transferred_labels={k: torch.zeros(20, dtype=torch.long) for k in synthetic_dataset},
        params_used={
            "k_neighbours": 5,
            "dist_metric": "euclidean",
            "smoothing": 0.0,
            "threshold": 0.0,
        },
    )


# ---------------------------------------------------------------------------
# TestEvaluationRunnerInit
# ---------------------------------------------------------------------------


class TestEvaluationRunnerInit:
    """Constructor attribute storage and D-06 fail-fast guard."""

    def test_init_with_valid_config_and_params(self, eval_config, full_params) -> None:
        """Constructor stores config, params (shallow copy), and engine."""
        runner = EvaluationRunner(eval_config, full_params)
        assert runner.config is eval_config
        assert runner.params == full_params
        assert runner.params is not full_params  # shallow copy identity check
        assert isinstance(runner.engine, MetricsEngine)

    def test_init_raises_when_both_stages_disabled(self, tmp_path) -> None:
        """D-06: ValueError raised before any attribute stored."""
        bad_config = EvalConfig(
            data_path=str(tmp_path / "x"),
            run_alignment=False,
            run_label_transfer=False,
            output_dir=str(tmp_path / "output"),
        )
        with pytest.raises(ValueError, match="At least one stage must be enabled"):
            EvaluationRunner(bad_config, {})

    def test_init_signature_is_config_params(self) -> None:
        """__init__ signature has exactly [self, config, params]."""
        sig = inspect.signature(EvaluationRunner.__init__)
        assert list(sig.parameters.keys()) == ["self", "config", "params"]


# ---------------------------------------------------------------------------
# TestEvaluationRunnerRunFixedParams — FRAME-07-G1
# ---------------------------------------------------------------------------


class TestEvaluationRunnerRunFixedParams:
    """FRAME-07-G1: run() with fixed params returns complete EvalReport."""

    @patch("eval.runners.eval_runner.DataFactory")
    def test_run_returns_eval_report(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """run() returns an EvalReport instance."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        # generate_labels populates 'color' (torch.long) not 'id' (None)
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        report = runner.run()
        assert isinstance(report, EvalReport)

    @patch("eval.runners.eval_runner.DataFactory")
    def test_run_metrics_is_stage_metrics(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """report.metrics is a StageMetrics instance."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        report = runner.run()
        assert isinstance(report.metrics, StageMetrics)

    @patch("eval.runners.eval_runner.DataFactory")
    def test_run_all_six_metric_fields_populated(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """All 6 StageMetrics float fields are present and are floats."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        report = runner.run()
        for field_name in (
            "chamfer_distance",
            "hausdorff_distance",
            "path_smoothness",
            "temporal_stability",
            "f1_score",
            "knn_consistency",
        ):
            assert isinstance(getattr(report.metrics, field_name), float), (
                f"{field_name} is not float: {getattr(report.metrics, field_name)!r}"
            )

    @patch("eval.runners.eval_runner.DataFactory")
    def test_run_aggregated_metrics_non_empty(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """report.aggregated_metrics is a non-empty dict."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        report = runner.run()
        assert report.aggregated_metrics
        assert len(report.aggregated_metrics) > 0

    @patch("eval.runners.eval_runner.DataFactory")
    def test_run_per_dataset_contains_dataset_key(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """report.per_dataset contains the 'dataset' key."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        report = runner.run()
        assert "dataset" in report.per_dataset


# ---------------------------------------------------------------------------
# TestEvaluationRunnerSanityFlags — FRAME-07-G2
# ---------------------------------------------------------------------------


class TestEvaluationRunnerSanityFlags:
    """FRAME-07-G2: 1-frame dataset triggers non-empty sanity_flags."""

    @patch("eval.runners.eval_runner.DataFactory")
    def test_single_frame_dataset_produces_sanity_flag(
        self, mock_factory_cls, eval_config, full_params, single_frame_dataset
    ) -> None:
        """1-frame dataset: report.sanity_flags is non-empty and contains 'single-frame'."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = single_frame_dataset
        mock_factory.load_target.return_value = single_frame_dataset
        # generate_labels populates 'color' (torch.long) not 'id' (None)
        mock_factory.get_ground_truth.return_value = {
            0: single_frame_dataset[0]["label"]
        }
        runner = EvaluationRunner(eval_config, full_params)
        report = runner.run()
        assert len(report.sanity_flags) > 0
        assert any("single-frame" in flag for flag in report.sanity_flags), (
            f"Expected 'single-frame' substring in sanity_flags, got: {report.sanity_flags!r}"
        )


# ---------------------------------------------------------------------------
# TestEvaluationRunnerSaveReport — FRAME-07-G3
# ---------------------------------------------------------------------------


class TestEvaluationRunnerSaveReport:
    """FRAME-07-G3: eval_report.json written to output_dir with correct content."""

    @patch("eval.runners.eval_runner.DataFactory")
    def test_save_report_writes_eval_report_json(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """eval_report.json exists after run()."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        runner.run()
        assert (Path(eval_config.output_dir) / "eval_report.json").exists()

    @patch("eval.runners.eval_runner.DataFactory")
    def test_eval_report_json_is_valid_json(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """eval_report.json parses as valid JSON dict."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        runner.run()
        with open(Path(eval_config.output_dir) / "eval_report.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    @patch("eval.runners.eval_runner.DataFactory")
    def test_eval_report_json_has_expected_keys(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset
    ) -> None:
        """eval_report.json has all 6 top-level expected keys."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        runner.run()
        with open(Path(eval_config.output_dir) / "eval_report.json") as f:
            data = json.load(f)
        expected_keys = {
            "params", "metrics", "aggregated_metrics",
            "per_dataset", "plot_paths", "trajectory_paths", "sanity_flags",
        }
        assert expected_keys.issubset(data.keys()), (
            f"Missing keys: {expected_keys - data.keys()}"
        )

    @patch("eval.runners.eval_runner.DataFactory")
    def test_save_report_returns_written_path(
        self, mock_factory_cls, eval_config, full_params, synthetic_dataset, tmp_path
    ) -> None:
        """save_report() returns Path to eval_report.json and file exists."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        runner.run()
        # Call save_report directly with a fresh dir
        out_dir = tmp_path / "direct_save"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build a minimal valid report via run() output
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        report = runner.run()
        returned_path = runner.save_report(report, out_dir)
        assert returned_path == out_dir / "eval_report.json"
        assert returned_path.exists()


# ---------------------------------------------------------------------------
# TestEvaluationRunnerConditionalStages
# ---------------------------------------------------------------------------


class TestEvaluationRunnerConditionalStages:
    """Conditional stage execution: run_alignment=False, run_label_transfer=False."""

    @patch("eval.runners.eval_runner.DataFactory")
    @patch("eval.runners.eval_runner.AlignmentStage")
    def test_alignment_disabled_skips_alignment_stage(
        self,
        mock_alignment_cls,
        mock_factory_cls,
        tmp_path,
        full_params,
        synthetic_dataset,
    ) -> None:
        """run_alignment=False: AlignmentStage.run is never called."""
        eval_config_no_align = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "output"),
            run_alignment=False,
            run_label_transfer=True,
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config_no_align, full_params)
        runner.run()
        mock_alignment_cls.assert_not_called()

    @patch("eval.runners.eval_runner.DataFactory")
    @patch("eval.runners.eval_runner.LabelTransferStage")
    def test_label_transfer_disabled_skips_label_stage(
        self,
        mock_label_cls,
        mock_factory_cls,
        tmp_path,
        full_params,
        synthetic_dataset,
    ) -> None:
        """run_label_transfer=False: LabelTransferStage.run is never called."""
        eval_config_no_label = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "output"),
            run_alignment=True,
            run_label_transfer=False,
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        # AlignmentStage still runs — mock it minimally so no real DTW
        with patch("eval.runners.eval_runner.AlignmentStage") as mock_align_cls:
            mock_align_instance = mock_align_cls.return_value
            mock_align_instance.run.return_value = AlignResult(
                aligned_cloud=synthetic_dataset,
                warp_path=[(0, 0), (1, 1), (2, 2)],
                dtw_distance=0.0,
                n_changepoints=0,
                params_used=dict(full_params),
            )
            runner = EvaluationRunner(eval_config_no_label, full_params)
            runner.run()
        mock_label_cls.assert_not_called()

    def test_both_disabled_raises_in_init(self, tmp_path) -> None:
        """Both stages disabled: ValueError raised in __init__."""
        with pytest.raises(ValueError, match="At least one stage must be enabled"):
            EvaluationRunner(
                EvalConfig(
                    data_path=str(tmp_path / "x"),
                    output_dir=str(tmp_path / "output"),
                    run_alignment=False,
                    run_label_transfer=False,
                ),
                {},
            )


# ---------------------------------------------------------------------------
# TestEvaluationRunnerSavePlots — FRAME-07 plot-path integration
# ---------------------------------------------------------------------------


class TestEvaluationRunnerSavePlots:
    """FRAME-07 plot-path integration: save_plots=True populates report.plot_paths."""

    @patch("eval.runners.eval_runner.DataFactory")
    def test_save_plots_true_populates_plot_paths(
        self,
        mock_factory_cls,
        eval_config,
        full_params,
        synthetic_dataset,
    ) -> None:
        """save_plots=True (default): report.plot_paths has at least 2 entries."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        report = runner.run()
        assert len(report.plot_paths) >= 2

    @patch("eval.runners.eval_runner.DataFactory")
    def test_save_plots_true_creates_alignment_trajectory_pdf(
        self,
        mock_factory_cls,
        eval_config,
        full_params,
        synthetic_dataset,
    ) -> None:
        """save_plots=True with run_alignment=True: alignment source/aligned/superposed PDFs exist.

        plot_trajectory (D-10) produces alignment_source_trajectory.pdf,
        alignment_aligned_trajectory.pdf, and alignment_superposed_trajectory.pdf
        in the output directory (VIZ-02 refactor — 3 figures when no target given).
        """
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        runner.run()
        assert (Path(eval_config.output_dir) / "alignment_source_trajectory.pdf").exists()

    @patch("eval.runners.eval_runner.DataFactory")
    def test_save_plots_true_creates_metrics_summary_pdf(
        self,
        mock_factory_cls,
        eval_config,
        full_params,
        synthetic_dataset,
    ) -> None:
        """save_plots=True: metrics_summary.pdf exists on disk."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        runner.run()
        assert (Path(eval_config.output_dir) / "metrics_summary.pdf").exists()

    @patch("eval.runners.eval_runner.DataFactory")
    def test_save_plots_false_leaves_plot_paths_empty(
        self,
        mock_factory_cls,
        tmp_path,
        full_params,
        synthetic_dataset,
    ) -> None:
        """save_plots=False: report.plot_paths == []."""
        eval_config_no_plots = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            save_plots=False,
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config_no_plots, full_params)
        report = runner.run()
        assert report.plot_paths == []

    @patch("eval.runners.eval_runner.DataFactory")
    def test_run_alignment_false_omits_alignment_trajectory_pdf(
        self,
        mock_factory_cls,
        tmp_path,
        full_params,
        synthetic_dataset,
    ) -> None:
        """run_alignment=False: alignment_trajectory.pdf absent; metrics_summary.pdf present."""
        eval_config_no_align = EvalConfig(
            data_path=str(tmp_path / "x"),
            output_dir=str(tmp_path / "out"),
            run_alignment=False,
            run_label_transfer=True,
            save_plots=True,
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config_no_align, full_params)
        report = runner.run()
        assert not any("alignment_source_trajectory.pdf" in p for p in report.plot_paths)
        assert any("metrics_summary.pdf" in p for p in report.plot_paths)


# ---------------------------------------------------------------------------
# TestEvaluationRunnerSyntheticMode — Phase 31 MODE-02 / MODE-03
# ---------------------------------------------------------------------------


class TestEvaluationRunnerSyntheticMode:
    """Phase 31: synthetic pipeline_mode wiring for EvaluationRunner.

    Verifies that:
    - run() in synthetic mode calls factory.generate_target() (not load_target())
    - _run_single() in synthetic mode calls factory.get_synthetic_ground_truth()
      (not get_ground_truth())
    - Paired mode is unchanged (regression guard)
    """

    @patch("eval.runners.eval_runner.DataFactory")
    def test_synthetic_mode_calls_generate_target(
        self,
        mock_factory_cls,
        tmp_path,
        full_params,
        synthetic_dataset,
    ) -> None:
        """Synthetic mode: generate_target called once with (source, transform_spec);
        load_target is NOT called (MODE-02 wiring)."""
        transform_spec = {"type": "noise", "sigma": 0.1}
        synth_config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec=transform_spec,
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.generate_target.return_value = synthetic_dataset
        mock_factory.get_synthetic_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(synth_config, full_params)
        runner.run()
        mock_factory.generate_target.assert_called_once_with(synthetic_dataset, transform_spec)
        mock_factory.load_target.assert_not_called()

    @patch("eval.runners.eval_runner.DataFactory")
    def test_synthetic_mode_calls_get_synthetic_ground_truth(
        self,
        mock_factory_cls,
        tmp_path,
        full_params,
        synthetic_dataset,
    ) -> None:
        """Synthetic mode: get_synthetic_ground_truth called at least once;
        get_ground_truth is NOT called (MODE-03 GT path)."""
        transform_spec = {"type": "noise", "sigma": 0.1}
        synth_config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec=transform_spec,
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.generate_target.return_value = synthetic_dataset
        mock_factory.get_synthetic_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(synth_config, full_params)
        runner.run()
        assert mock_factory.get_synthetic_ground_truth.called is True
        assert mock_factory.get_ground_truth.called is False

    @patch("eval.runners.eval_runner.DataFactory")
    def test_paired_mode_unchanged(
        self,
        mock_factory_cls,
        eval_config,
        full_params,
        synthetic_dataset,
    ) -> None:
        """Regression guard: paired mode calls load_target(), never generate_target()."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset
        mock_factory.get_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(eval_config, full_params)
        runner.run()
        assert mock_factory.load_target.called is True
        assert mock_factory.generate_target.called is False

    @patch("eval.runners.eval_runner.DataFactory")
    def test_synthetic_mode_run_returns_eval_report(
        self,
        mock_factory_cls,
        tmp_path,
        full_params,
        synthetic_dataset,
    ) -> None:
        """Synthetic mode: run() returns a populated EvalReport (smoke test)."""
        transform_spec = {"type": "noise", "sigma": 0.1}
        synth_config = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "output"),
            pipeline_mode="synthetic",
            transform_spec=transform_spec,
        )
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.generate_target.return_value = synthetic_dataset
        mock_factory.get_synthetic_ground_truth.return_value = {
            k: synthetic_dataset[k]["label"] for k in synthetic_dataset
        }
        runner = EvaluationRunner(synth_config, full_params)
        report = runner.run()
        assert isinstance(report, EvalReport)


# ---------------------------------------------------------------------------
# Coverage gap tests for eval_runner.py
# ---------------------------------------------------------------------------


class TestEvaluationRunnerCoverageGaps:
    """Additional tests to cover missed lines in eval_runner.py."""

    def test_run_single_raises_when_factory_none(self, eval_config, full_params, synthetic_dataset):
        """eval_runner.py:301 — RuntimeError when factory is None and _run_single called directly."""
        runner = EvaluationRunner(eval_config, full_params)
        # factory is None at init time before run() is called
        assert runner.factory is None
        with pytest.raises(RuntimeError, match="factory not initialised"):
            runner._run_single(synthetic_dataset, synthetic_dataset, full_params)

    @patch("eval.runners.eval_runner.DataFactory")
    def test_run_single_truncates_y_true_y_pred_when_different_size(
        self,
        mock_factory_cls,
        eval_config,
        full_params,
        synthetic_dataset,
    ) -> None:
        """eval_runner.py:348-350 — truncation when y_true.shape[0] != y_pred.shape[0]."""
        mock_factory = mock_factory_cls.return_value
        mock_factory.load_real.return_value = synthetic_dataset
        mock_factory.load_target.return_value = synthetic_dataset

        # y_true has 3 pts (from gt), y_pred has 8 pts (from transferred_labels)
        # knn target frame has 8 points so knn labels (8) match knn points (8)
        # The truncation lines 348-350 trim y_true/y_pred to min(3,8)=3
        n_target = 8
        y_true_3 = torch.zeros(3, dtype=torch.long)
        y_pred_n = torch.zeros(n_target, dtype=torch.long)

        mock_factory.get_ground_truth.return_value = {
            k: y_true_3 for k in synthetic_dataset
        }

        # Target dataset with n_target points so knn check passes
        target_ds = {
            k: zRegPointCloud(
                pos=torch.randn(n_target, 3),
                label=torch.zeros(n_target, dtype=torch.long),
                id=torch.arange(n_target),
            )
            for k in synthetic_dataset
        }
        mock_factory.load_target.return_value = target_ds

        fake_label_result = LabelResult(
            transferred_labels={k: y_pred_n for k in synthetic_dataset},
            params_used=dict(full_params),
        )

        with patch("eval.runners.eval_runner.AlignmentStage") as mock_align_cls, \
             patch("eval.runners.eval_runner.LabelTransferStage") as mock_label_cls:
            mock_align_cls.return_value.run.return_value = AlignResult(
                aligned_cloud=synthetic_dataset,
                warp_path=[(0, 0), (1, 1), (2, 2)],
                dtw_distance=0.0,
                n_changepoints=0,
                params_used=dict(full_params),
            )
            mock_label_cls.return_value.run.return_value = fake_label_result

            runner = EvaluationRunner(eval_config, full_params)
            report = runner.run()
        assert isinstance(report, EvalReport)
