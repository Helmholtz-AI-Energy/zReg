"""Tests for eval.stages.label_transfer.LabelTransferStage.

Covers Phase 20 FRAME-06 gate criteria:
  - LabelTransferStage importable standalone (gate 1)
  - REQUIRED_PARAMS tuple (gate 3)
  - validate_params guards including bool exclusion (gate 4 / WR-01)
  - run() calls validate_params first (gate 5 / D-08)
  - No kNN or distance logic reimplemented (gate 6)
"""

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_trajectory

import torch

from eval.config import EvalConfig
from eval.stages import LabelTransferStage, PipelineStage
from eval.stages.label_transfer import LabelTransferStage as LabelTransferStageDirect
from eval.types import LabelResult


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_config(tmp_path) -> EvalConfig:
    """Minimal EvalConfig — only data_path is required."""
    return EvalConfig(data_path=str(tmp_path / "unused.mat"))


@pytest.fixture
def stage(eval_config) -> LabelTransferStage:
    """Instantiated LabelTransferStage with minimal config."""
    return LabelTransferStage(eval_config)


@pytest.fixture
def good_params() -> dict:
    """Valid params that pass LabelTransferStage.validate_params."""
    return {
        "k_neighbours": 5,
        "dist_metric": "euclidean",
        "smoothing": 0.0,
        "threshold": 0.0,
    }


@pytest.fixture
def synthetic_dataset() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with color field for label transfer tests."""
    return generate_trajectory(n_points=20, n_frames=3, seed=0)


# ---------------------------------------------------------------------------
# TestLabelTransferStageImport — FRAME-06 gate 1
# ---------------------------------------------------------------------------


class TestLabelTransferStageImport:
    """LabelTransferStage importable without ImportError (FRAME-06 gate 1)."""

    def test_import_from_stages_package(self):
        """from eval.stages import LabelTransferStage succeeds."""
        from eval.stages import LabelTransferStage  # noqa: F401
        assert LabelTransferStage is not None

    def test_import_from_module_directly(self):
        """from eval.stages.label_transfer import LabelTransferStage succeeds."""
        assert LabelTransferStageDirect is not None

    def test_is_pipeline_stage_subclass(self):
        """LabelTransferStage is a subclass of PipelineStage."""
        assert issubclass(LabelTransferStage, PipelineStage)

    def test_in_stages_all(self):
        """LabelTransferStage appears in eval.stages.__all__."""
        import eval.stages
        assert "LabelTransferStage" in eval.stages.__all__

    def test_all_three_in_stages_all(self):
        """eval.stages.__all__ contains exactly the three expected names."""
        import eval.stages
        assert eval.stages.__all__ == ["PipelineStage", "AlignmentStage", "LabelTransferStage"]


# ---------------------------------------------------------------------------
# TestRequiredParams — FRAME-06 gate 3
# ---------------------------------------------------------------------------


class TestRequiredParams:
    """REQUIRED_PARAMS class attribute has the right content and order."""

    def test_required_params_exists(self):
        """LabelTransferStage.REQUIRED_PARAMS attribute exists."""
        assert hasattr(LabelTransferStage, "REQUIRED_PARAMS")

    def test_required_params_exact_tuple(self):
        """REQUIRED_PARAMS is the exact 4-element tuple in specified order."""
        assert LabelTransferStage.REQUIRED_PARAMS == (
            "k_neighbours",
            "dist_metric",
            "smoothing",
            "threshold",
        )

    def test_required_params_accessible_without_instantiation(self):
        """REQUIRED_PARAMS is accessible as a class attribute (no instance needed)."""
        # No EvalConfig needed — class-level access
        assert isinstance(LabelTransferStage.REQUIRED_PARAMS, tuple)
        assert len(LabelTransferStage.REQUIRED_PARAMS) == 4


# ---------------------------------------------------------------------------
# TestValidateParamsMissingKeys — FRAME-06 gate 4 (missing-key branch)
# ---------------------------------------------------------------------------


class TestValidateParamsMissingKeys:
    """validate_params raises ValueError for each missing REQUIRED_PARAMS key."""

    @pytest.mark.parametrize("missing_key", [
        "k_neighbours",
        "dist_metric",
        "smoothing",
        "threshold",
    ])
    def test_missing_key_raises_value_error(self, stage, good_params, missing_key):
        """validate_params raises ValueError('Missing required param: {key}')."""
        params = {k: v for k, v in good_params.items() if k != missing_key}
        with pytest.raises(ValueError, match=f"Missing required param: {missing_key}"):
            stage.validate_params(params)

    def test_empty_dict_raises_for_first_required_param(self, stage):
        """validate_params({}) raises ValueError for k_neighbours (first REQUIRED_PARAMS key)."""
        with pytest.raises(ValueError, match="Missing required param: k_neighbours"):
            stage.validate_params({})


# ---------------------------------------------------------------------------
# TestValidateParamsKNeighbours — FRAME-06 gate 4 (k_neighbours guards)
# ---------------------------------------------------------------------------


class TestValidateParamsKNeighbours:
    """validate_params guards for k_neighbours (int >= 1, bool exclusion)."""

    def test_k_neighbours_zero_raises(self, stage, good_params):
        """k_neighbours=0 raises ValueError matching 'k_neighbours must be int >= 1'."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": 0})

    def test_k_neighbours_negative_raises(self, stage, good_params):
        """k_neighbours=-1 raises ValueError matching 'k_neighbours must be int >= 1'."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": -1})

    def test_k_neighbours_bool_true_raises(self, stage, good_params):
        """k_neighbours=True raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": True})

    def test_k_neighbours_bool_false_raises(self, stage, good_params):
        """k_neighbours=False raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": False})

    def test_k_neighbours_float_raises(self, stage, good_params):
        """k_neighbours=1.0 raises ValueError (must be int)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.validate_params({**good_params, "k_neighbours": 1.0})

    def test_k_neighbours_one_valid(self, stage, good_params):
        """k_neighbours=1 is valid (no exception raised)."""
        stage.validate_params({**good_params, "k_neighbours": 1})

    def test_k_neighbours_large_valid(self, stage, good_params):
        """k_neighbours=100 is valid (no exception raised)."""
        stage.validate_params({**good_params, "k_neighbours": 100})


# ---------------------------------------------------------------------------
# TestValidateParamsDistMetric — FRAME-06 gate 4 (dist_metric guards)
# ---------------------------------------------------------------------------


class TestValidateParamsDistMetric:
    """validate_params guards for dist_metric (non-empty str)."""

    def test_empty_string_raises(self, stage, good_params):
        """dist_metric='' raises ValueError matching 'dist_metric must be non-empty'."""
        with pytest.raises(ValueError, match="dist_metric must be non-empty"):
            stage.validate_params({**good_params, "dist_metric": ""})

    def test_non_string_raises(self, stage, good_params):
        """dist_metric=123 raises ValueError."""
        with pytest.raises(ValueError, match="dist_metric must be non-empty"):
            stage.validate_params({**good_params, "dist_metric": 123})

    def test_valid_string_passes(self, stage, good_params):
        """dist_metric='euclidean' is valid (no exception raised)."""
        stage.validate_params({**good_params, "dist_metric": "euclidean"})


# ---------------------------------------------------------------------------
# TestValidateParamsSmoothing — FRAME-06 gate 4 (smoothing guards)
# ---------------------------------------------------------------------------


class TestValidateParamsSmoothing:
    """validate_params guards for smoothing (float >= 0.0, bool exclusion)."""

    def test_negative_raises(self, stage, good_params):
        """smoothing=-1.0 raises ValueError matching 'smoothing must be float >= 0.0'."""
        with pytest.raises(ValueError, match="smoothing must be float >= 0.0"):
            stage.validate_params({**good_params, "smoothing": -1.0})

    def test_bool_true_raises(self, stage, good_params):
        """smoothing=True raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="smoothing must be float >= 0.0"):
            stage.validate_params({**good_params, "smoothing": True})

    def test_bool_false_raises(self, stage, good_params):
        """smoothing=False raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="smoothing must be float >= 0.0"):
            stage.validate_params({**good_params, "smoothing": False})

    def test_zero_valid(self, stage, good_params):
        """smoothing=0.0 is valid (no exception raised)."""
        stage.validate_params({**good_params, "smoothing": 0.0})

    def test_positive_float_valid(self, stage, good_params):
        """smoothing=0.5 is valid (no exception raised)."""
        stage.validate_params({**good_params, "smoothing": 0.5})

    def test_int_zero_valid(self, stage, good_params):
        """smoothing=0 (int) is valid — int is accepted as numeric."""
        stage.validate_params({**good_params, "smoothing": 0})


# ---------------------------------------------------------------------------
# TestValidateParamsThreshold — FRAME-06 gate 4 (threshold guards)
# ---------------------------------------------------------------------------


class TestValidateParamsThreshold:
    """validate_params guards for threshold (float >= 0.0, bool exclusion)."""

    def test_negative_raises(self, stage, good_params):
        """threshold=-1.0 raises ValueError matching 'threshold must be float >= 0.0'."""
        with pytest.raises(ValueError, match="threshold must be float >= 0.0"):
            stage.validate_params({**good_params, "threshold": -1.0})

    def test_bool_true_raises(self, stage, good_params):
        """threshold=True raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="threshold must be float >= 0.0"):
            stage.validate_params({**good_params, "threshold": True})

    def test_bool_false_raises(self, stage, good_params):
        """threshold=False raises ValueError (bool exclusion — WR-01)."""
        with pytest.raises(ValueError, match="threshold must be float >= 0.0"):
            stage.validate_params({**good_params, "threshold": False})

    def test_zero_valid(self, stage, good_params):
        """threshold=0.0 is valid (no exception raised)."""
        stage.validate_params({**good_params, "threshold": 0.0})

    def test_positive_float_valid(self, stage, good_params):
        """threshold=0.5 is valid (no exception raised)."""
        stage.validate_params({**good_params, "threshold": 0.5})


# ---------------------------------------------------------------------------
# TestRunValidatesFirst — FRAME-06 gate 5 / D-08
# ---------------------------------------------------------------------------


class TestRunValidatesFirst:
    """run() calls validate_params as its first line (D-08)."""

    def test_run_empty_params_raises_value_error(self, stage):
        """run({dataset}, {}) raises ValueError (not KeyError/TypeError) — D-08."""
        with pytest.raises(ValueError, match="Missing required param"):
            stage.run({}, {})

    def test_run_empty_dataset_empty_params_raises_value_error(self, stage):
        """run({}, {}) raises ValueError before touching the dataset — D-08 ordering."""
        with pytest.raises(ValueError, match="Missing required param"):
            stage.run({}, {})

    def test_run_invalid_k_neighbours_raises_before_dataset_access(self, stage, good_params):
        """run with k_neighbours=True raises ValueError (not attribute error from dataset)."""
        with pytest.raises(ValueError, match="k_neighbours must be int >= 1"):
            stage.run({}, {**good_params, "k_neighbours": True})


# ---------------------------------------------------------------------------
# TestRunOutput — FRAME-06 run() correctness
# ---------------------------------------------------------------------------


class TestRunOutput:
    """run() returns correct LabelResult structure."""

    def test_run_returns_label_result(self, stage, synthetic_dataset, good_params):
        """run() returns a LabelResult instance."""
        result = stage.run(synthetic_dataset, good_params)
        assert isinstance(result, LabelResult)

    def test_run_transferred_labels_keys_match_dataset(self, stage, synthetic_dataset, good_params):
        """transferred_labels has one entry per frame in the dataset."""
        result = stage.run(synthetic_dataset, good_params)
        assert set(result.transferred_labels.keys()) == set(synthetic_dataset.keys())

    def test_run_params_used_is_shallow_copy(self, stage, synthetic_dataset, good_params):
        """params_used is dict(params) — mutating original does not affect result."""
        params = dict(good_params)
        result = stage.run(synthetic_dataset, params)
        params["k_neighbours"] = 999  # mutate original
        assert result.params_used["k_neighbours"] == 5  # result unchanged

    def test_run_frame0_passthrough(self, stage, synthetic_dataset, good_params):
        """transferred_labels[min_key] is the source frame's color tensor (D-02)."""
        result = stage.run(synthetic_dataset, good_params)
        first_key = min(synthetic_dataset.keys())
        assert result.transferred_labels[first_key] is synthetic_dataset[first_key]["color"]

    def test_run_non_frame0_tensors_are_1d(self, stage, synthetic_dataset, good_params):
        """All transferred_labels values after frame 0 are 1-D tensors (squeezed)."""
        result = stage.run(synthetic_dataset, good_params)
        sorted_keys = sorted(synthetic_dataset.keys())
        for key in sorted_keys[1:]:
            assert result.transferred_labels[key].ndim == 1

    def test_run_label_result_is_frozen(self, stage, synthetic_dataset, good_params):
        """LabelResult is pydantic-frozen (attribute reassignment raises)."""
        from pydantic import ValidationError
        result = stage.run(synthetic_dataset, good_params)
        with pytest.raises((ValidationError, TypeError)):
            result.transferred_labels = {}

    def test_run_uses_knn_voting_not_reimplemented(self, stage, synthetic_dataset, good_params):
        """k_neighbours=1 is accepted and run() completes (delegates to transfer_colors)."""
        params = {**good_params, "k_neighbours": 1}
        result = stage.run(synthetic_dataset, params)
        assert isinstance(result, LabelResult)
