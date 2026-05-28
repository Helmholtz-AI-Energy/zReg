"""Tests for eval.stages.base.PipelineStage ABC and eval.stages.alignment.AlignmentStage.

Six test classes cover Phase 19 deliverables per FRAME-05 gate criteria.
"""

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_trajectory

import torch

from eval.config import EvalConfig
from eval.stages import AlignmentStage, PipelineStage
from eval.types import AlignResult, StageResult


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset_a() -> dict[int, zRegPointCloud]:
    """4-frame Gaussian-blob trajectory, seed=0."""
    return generate_trajectory(n_points=20, n_frames=4, seed=0)


@pytest.fixture
def synthetic_dataset_b() -> dict[int, zRegPointCloud]:
    """4-frame Gaussian-blob trajectory, seed=42 (second dataset for gate 2)."""
    return generate_trajectory(n_points=20, n_frames=4, seed=42)


@pytest.fixture
def default_params() -> dict:
    """Reasonable default hyperparams that pass AlignmentStage.validate_params.

    window_size=10 >= dataset length (4) avoids Pitfall 4 (window too tight).
    cpd_penalty="rigid" per Pitfall 3 (only "rigid" dispatches correctly upstream).
    """
    return {
        "window_size": 10,
        "step": 1,
        "cpd_penalty": "rigid",
        "dtw_dist_fn": "euclidean",
        "n_breakpoints": 5,
    }


@pytest.fixture
def eval_config(tmp_path) -> EvalConfig:
    """Minimal EvalConfig — only data_path is required."""
    return EvalConfig(data_path=str(tmp_path / "unused.mat"))


# ---------------------------------------------------------------------------
# TestStageResultExported — Plan 19-01 (POPULATED)
# ---------------------------------------------------------------------------


class TestStageResultExported:
    """Plan 19-01 — StageResult TypeAlias importable from eval.types (D-01)."""

    def test_stage_result_importable(self):
        """StageResult is importable and present in eval.types.__all__."""
        assert StageResult is not None
        assert "StageResult" in __import__("eval.types", fromlist=["__all__"]).__all__


# ---------------------------------------------------------------------------
# TestPipelineStageABC — Plan 19-01 (POPULATED)
# ---------------------------------------------------------------------------


class TestPipelineStageABC:
    """Plan 19-01 — PipelineStage ABC contract."""

    def test_cannot_instantiate_abstract_base(self):
        """ABC: instantiating PipelineStage directly raises TypeError matching 'abstract'."""
        with pytest.raises(TypeError, match="abstract"):
            PipelineStage(config=None)


# ---------------------------------------------------------------------------
# TestAlignmentStageRunStandalone — Plan 19-02 (POPULATED)
# ---------------------------------------------------------------------------


class TestAlignmentStageRunStandalone:
    """Gate 1: AlignmentStage.run() completes without LabelTransferStage present (FRAME-05)."""

    def test_run_returns_align_result(self, synthetic_dataset_a, default_params, eval_config):
        """Standalone run returns a fully-populated AlignResult without LabelTransferStage."""
        stage = AlignmentStage(eval_config)
        result = stage.run(synthetic_dataset_a, default_params)

        assert isinstance(result, AlignResult)
        # aligned_cloud values equal the input dataset (pydantic v2 validates dict[int, ...] into a new dict,
        # so identity check is not feasible; equality confirms pass-through semantics)
        assert result.aligned_cloud == synthetic_dataset_a
        # warp_path is a non-empty list of (int, int) tuples
        assert isinstance(result.warp_path, list) and len(result.warp_path) >= 1
        # dtw_distance is a non-negative float
        assert result.dtw_distance >= 0.0
        # n_changepoints is a non-negative int
        assert result.n_changepoints >= 0
        # params_used equals the original dict contents
        assert result.params_used == default_params
        # params_used must be a shallow copy — NOT the same object (Pitfall 7)
        assert result.params_used is not default_params


# ---------------------------------------------------------------------------
# TestAlignmentStageValidateParams — Plan 19-02 (POPULATED)
# ---------------------------------------------------------------------------


class TestAlignmentStageValidateParams:
    """Gate 3: validate_params raises on missing/invalid params (D-08, D-09, D-10)."""

    @pytest.mark.parametrize("missing_key", AlignmentStage.REQUIRED_PARAMS)
    def test_missing_param_raises(self, missing_key, default_params, eval_config):
        """Each of the 5 required keys raises ValueError when absent (tested via run — D-09)."""
        partial = {k: v for k, v in default_params.items() if k != missing_key}
        stage = AlignmentStage(eval_config)
        dataset = {0: zRegPointCloud(pos=torch.randn(5, 3))}
        with pytest.raises(ValueError, match=f"Missing required param: {missing_key}"):
            stage.run(dataset, partial)

    def test_run_calls_validate_first(self, eval_config):
        """D-09: run calls validate_params before DTW so empty params raises ValueError not KeyError/TypeError."""
        stage = AlignmentStage(eval_config)
        with pytest.raises(ValueError, match="Missing required param"):
            stage.run({}, {})

    @pytest.mark.parametrize(
        "key,bad_value,match_str",
        [
            ("window_size", 0, "window_size must be int > 0"),
            ("window_size", -1, "window_size must be int > 0"),
            ("window_size", True, "window_size must be int > 0"),
            ("step", 0, "step must be int >= 1"),
            ("cpd_penalty", "bogus", "cpd_penalty must be one of"),
            ("dtw_dist_fn", "", "dtw_dist_fn must be non-empty"),
            ("n_breakpoints", -1, "n_breakpoints must be int >= 0"),
        ],
    )
    def test_invalid_value_raises(self, key, bad_value, match_str, default_params, eval_config):
        """D-10: invalid values for each param raise descriptive ValueError."""
        params = {**default_params, key: bad_value}
        stage = AlignmentStage(eval_config)
        with pytest.raises(ValueError, match=match_str):
            stage.validate_params(params)


# ---------------------------------------------------------------------------
# TestAlignmentDistanceImproves — Plan 19-02 (POPULATED)
# ---------------------------------------------------------------------------


class TestAlignmentDistanceImproves:
    """Gate 2: DTW distance after alignment does not increase by more than 1e-3 on >=2 synthetic datasets (FRAME-05 success criterion 2).

    Uses cpd_penalty='rigid' per Pitfall 3 (all non-None cpd_penalty values behave
    identically due to upstream bug in pairwise_distance_matrix.py:182-194).
    """

    def test_cpd_does_not_worsen_dtw_distance_dataset_a(self, synthetic_dataset_a, default_params, eval_config):
        """CPD should not increase DTW distance on dataset seed=0 (with 1e-3 tolerance for noise)."""
        stage = AlignmentStage(eval_config)
        params_no_cpd = {**default_params, "cpd_penalty": None}
        params_cpd = {**default_params, "cpd_penalty": "rigid"}

        r_no = stage.run(synthetic_dataset_a, params_no_cpd)
        r_cpd = stage.run(synthetic_dataset_a, params_cpd)

        assert r_cpd.dtw_distance <= r_no.dtw_distance + 1e-3, (
            f"CPD should not increase DTW distance; got "
            f"r_cpd.dtw_distance={r_cpd.dtw_distance} > r_no.dtw_distance={r_no.dtw_distance}"
        )

    def test_cpd_does_not_worsen_dtw_distance_dataset_b(self, synthetic_dataset_b, default_params, eval_config):
        """CPD should not increase DTW distance on dataset seed=42 (second dataset for gate 2)."""
        stage = AlignmentStage(eval_config)
        params_no_cpd = {**default_params, "cpd_penalty": None}
        params_cpd = {**default_params, "cpd_penalty": "rigid"}

        r_no = stage.run(synthetic_dataset_b, params_no_cpd)
        r_cpd = stage.run(synthetic_dataset_b, params_cpd)

        assert r_cpd.dtw_distance <= r_no.dtw_distance + 1e-3, (
            f"CPD should not increase DTW distance; got "
            f"r_cpd.dtw_distance={r_cpd.dtw_distance} > r_no.dtw_distance={r_no.dtw_distance}"
        )


# ---------------------------------------------------------------------------
# TestCountJumps — Plan 19-02 (POPULATED)
# ---------------------------------------------------------------------------


class TestCountJumps:
    """D-04/D-05: _count_jumps diagonal/non-diagonal transition heuristic."""

    def test_empty_path_returns_zero(self):
        """Empty path has no steps and thus no transitions."""
        assert AlignmentStage._count_jumps([]) == 0

    def test_single_point_returns_zero(self):
        """Single-point path has no step pairs and thus no transitions."""
        assert AlignmentStage._count_jumps([(0, 0)]) == 0

    def test_pure_diagonal_no_jumps(self):
        """All diagonal steps (+1,+1) produce the same kind — zero transitions."""
        assert AlignmentStage._count_jumps([(0, 0), (1, 1), (2, 2), (3, 3)]) == 0

    def test_pure_horizontal_no_jumps(self):
        """All horizontal steps (0,+1) produce the same kind — zero transitions."""
        assert AlignmentStage._count_jumps([(0, 0), (0, 1), (0, 2), (0, 3)]) == 0

    def test_one_transition(self):
        """diag -> step is 1 transition."""
        assert AlignmentStage._count_jumps([(0, 0), (1, 1), (1, 2)]) == 1

    def test_two_transitions(self):
        """diag -> step -> diag is 2 transitions."""
        assert AlignmentStage._count_jumps([(0, 0), (1, 1), (1, 2), (2, 3)]) == 2

    def test_count_jumps_raw_count_not_capped(self):
        """_count_jumps returns RAW count; n_breakpoints cap is applied in run(), not here (D-06)."""
        # raw=2, but this function always returns the uncapped count
        assert AlignmentStage._count_jumps([(0, 0), (1, 1), (1, 2), (2, 3)]) == 2
