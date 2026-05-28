"""Tests for eval.stages.base.PipelineStage ABC and eval.stages.alignment.AlignmentStage.

Six test classes cover Phase 19 deliverables per FRAME-05 gate criteria.
"""

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_trajectory

import torch

from eval.config import EvalConfig
from eval.stages import PipelineStage
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
# TestAlignmentStageRunStandalone — Plan 19-02 (STUBBED)
# ---------------------------------------------------------------------------


class TestAlignmentStageRunStandalone:
    """Gate 1: AlignmentStage.run() completes without LabelTransferStage. Populated in Plan 19-02."""

    def test_run_returns_align_result(self, synthetic_dataset_a, default_params, eval_config):
        pytest.skip("populated in Plan 19-02")


# ---------------------------------------------------------------------------
# TestAlignmentStageValidateParams — Plan 19-02 (STUBBED)
# ---------------------------------------------------------------------------


class TestAlignmentStageValidateParams:
    """Gate 3: validate_params raises on missing/invalid params. Populated in Plan 19-02."""

    def test_missing_param_raises(self, default_params, eval_config):
        pytest.skip("populated in Plan 19-02")

    def test_run_calls_validate_first(self, eval_config):
        pytest.skip("populated in Plan 19-02")

    def test_invalid_value_raises(self, default_params, eval_config):
        pytest.skip("populated in Plan 19-02")


# ---------------------------------------------------------------------------
# TestAlignmentDistanceImproves — Plan 19-02 (STUBBED)
# ---------------------------------------------------------------------------


class TestAlignmentDistanceImproves:
    """Gate 2: DTW distance after alignment measurably smaller on >=2 synthetic datasets. Populated in Plan 19-02."""

    def test_cpd_reduces_dtw_distance_dataset_a(self, synthetic_dataset_a, default_params, eval_config):
        pytest.skip("populated in Plan 19-02")

    def test_cpd_reduces_dtw_distance_dataset_b(self, synthetic_dataset_b, default_params, eval_config):
        pytest.skip("populated in Plan 19-02")


# ---------------------------------------------------------------------------
# TestCountJumps — Plan 19-02 (STUBBED)
# ---------------------------------------------------------------------------


class TestCountJumps:
    """D-04/D-05: _count_jumps diagonal/non-diagonal transition heuristic. Populated in Plan 19-02."""

    def test_empty_path(self):
        pytest.skip("populated in Plan 19-02")

    def test_pure_diagonal(self):
        pytest.skip("populated in Plan 19-02")

    def test_one_transition(self):
        pytest.skip("populated in Plan 19-02")
