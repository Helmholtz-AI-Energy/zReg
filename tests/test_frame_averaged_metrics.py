"""Degeneracy tests for frame-averaged chamfer/hausdorff (NUM-05, U1-2/3/5/6).

Covers ``MetricsEngine._frame_averaged_chamfer_hausdorff`` directly (the
``FrameAverage`` contract: ``chamfer``, ``hausdorff``, ``n_scored``,
``flags``) and end-to-end through ``compute_stage_metrics`` /
``sanity_check``.  Nothing is mocked: reference values are computed
independently with ``zreg.evaluation.chamfer`` / ``hausdorff``.

Key contract: a degenerate outcome (no shared frame keys, non-dict inputs,
every frame degenerate) yields ``+inf`` (normalises to ``0.0`` = worst),
never ``0.0`` (which would normalise to a fake-perfect ``1.0``).
"""

import math
import statistics

# zreg MUST be imported before torch (libomp SIGABRT lesson, see conftest.py)
from zreg.core.dataset import zRegPointCloud
from zreg.evaluation import chamfer, hausdorff

import pytest
import torch

from eval.config import EvalConfig
from eval.metrics import MetricsEngine
from eval.types import StageMetrics

N_POINTS = 12


def _cloud(seed: int, n: int = N_POINTS) -> zRegPointCloud:
    """Deterministic random cloud with ``n`` points."""
    gen = torch.Generator().manual_seed(seed)
    return zRegPointCloud(pos=torch.rand(n, 3, generator=gen) * (1.0 + seed))


def _empty_cloud() -> zRegPointCloud:
    return zRegPointCloud(pos=torch.zeros(0, 3))


def _nan_cloud(seed: int) -> zRegPointCloud:
    pc = _cloud(seed)
    pos = pc["pos"].clone()
    pos[0, 0] = float("nan")
    return zRegPointCloud(pos=pos)


def _ref(a: zRegPointCloud, b: zRegPointCloud) -> tuple[float, float]:
    return chamfer(a["pos"], b["pos"]).item(), hausdorff(a["pos"], b["pos"]).item()


@pytest.fixture
def engine() -> MetricsEngine:
    return MetricsEngine(EvalConfig(data_path="x"))


@pytest.fixture
def labels() -> torch.Tensor:
    return torch.tensor([0, 1, 2] * (N_POINTS // 3), dtype=torch.long)


def _stage(engine, aligned, target, labels, points):
    return engine.compute_stage_metrics(
        aligned, target, [], [], labels, labels, points, labels, k_neighbours=3
    )


# ---------------------------------------------------------------------------
# Direct FrameAverage contract (U1-6, review MEDIUM 59-04)
# ---------------------------------------------------------------------------


class TestFrameAverageDirect:
    """Direct calls to ``_frame_averaged_chamfer_hausdorff``."""

    def test_direct_full_overlap_healthy_means(self, engine):
        from eval.metrics import FrameAverage

        aligned = {k: _cloud(k) for k in (0, 1, 2)}
        target = {k: _cloud(10 + k) for k in (0, 1, 2)}
        fa = engine._frame_averaged_chamfer_hausdorff(aligned, target)
        refs = [_ref(aligned[k], target[k]) for k in (0, 1, 2)]

        assert isinstance(fa, FrameAverage)
        assert fa.n_scored == 3
        assert fa.flags == []
        assert fa.chamfer == pytest.approx(statistics.mean(r[0] for r in refs))
        assert fa.hausdorff == pytest.approx(statistics.mean(r[1] for r in refs))
        assert math.isfinite(fa.chamfer) and fa.chamfer > 0
        assert math.isfinite(fa.hausdorff) and fa.hausdorff > 0

    def test_direct_partial_overlap(self, engine):
        aligned = {k: _cloud(k) for k in (0, 1, 2)}
        target = {k: _cloud(10 + k) for k in (1, 2, 3)}
        fa = engine._frame_averaged_chamfer_hausdorff(aligned, target)
        refs = [_ref(aligned[k], target[k]) for k in (1, 2)]

        assert fa.n_scored == 2
        assert fa.chamfer == pytest.approx(statistics.mean(r[0] for r in refs))
        assert fa.hausdorff == pytest.approx(statistics.mean(r[1] for r in refs))
        assert len(fa.flags) == 1
        flag = fa.flags[0]
        assert flag.startswith("frame coverage: partial overlap")
        assert "[0]" in flag
        assert "[3]" in flag

    def test_direct_degenerate_frame_skipped(self, engine):
        a, b = _cloud(0), _cloud(1)
        aligned = {0: a, 1: _empty_cloud()}
        target = {0: b, 1: b}
        fa = engine._frame_averaged_chamfer_hausdorff(aligned, target)

        assert fa.n_scored == 1
        assert fa.chamfer == pytest.approx(_ref(a, b)[0])
        assert fa.hausdorff == pytest.approx(_ref(a, b)[1])
        assert len(fa.flags) == 1
        assert fa.flags[0].startswith(
            "frame coverage: skipped degenerate frame 1"
        )

    def test_direct_non_finite_frame_skipped(self, engine):
        a, b = _cloud(0), _cloud(1)
        aligned = {0: a, 1: _nan_cloud(2)}
        target = {0: b, 1: b}
        fa = engine._frame_averaged_chamfer_hausdorff(aligned, target)

        assert fa.n_scored == 1
        assert fa.chamfer == pytest.approx(_ref(a, b)[0])
        assert len(fa.flags) == 1
        assert fa.flags[0].startswith(
            "frame coverage: skipped degenerate frame 1"
        )

    def test_direct_non_finite_result_frame_skipped(self, engine):
        # Finite float32 inputs whose pairwise distance overflows: chamfer
        # / hausdorff accept them but return inf / nan.
        a, b = _cloud(0), _cloud(1)
        huge = zRegPointCloud(pos=torch.full((4, 3), 3e38))
        neg_huge = zRegPointCloud(pos=torch.full((4, 3), -3e38))
        aligned = {0: a, 1: huge}
        target = {0: b, 1: neg_huge}
        fa = engine._frame_averaged_chamfer_hausdorff(aligned, target)

        assert fa.n_scored == 1
        assert fa.chamfer == pytest.approx(_ref(a, b)[0])
        assert len(fa.flags) == 1
        assert fa.flags[0].startswith(
            "frame coverage: skipped degenerate frame 1: non-finite result"
        )

    def test_direct_all_frames_degenerate(self, engine):
        aligned = {0: _empty_cloud(), 1: _empty_cloud()}
        target = {0: _cloud(0), 1: _cloud(1)}
        fa = engine._frame_averaged_chamfer_hausdorff(aligned, target)

        assert fa.chamfer == math.inf
        assert fa.hausdorff == math.inf
        assert fa.n_scored == 0
        assert len(fa.flags) == 2
        assert fa.flags[0].startswith(
            "frame coverage: skipped degenerate frame 0"
        )
        assert fa.flags[1].startswith(
            "frame coverage: skipped degenerate frame 1"
        )

    def test_direct_empty_intersection(self, engine):
        fa = engine._frame_averaged_chamfer_hausdorff(
            {0: _cloud(0)}, {5: _cloud(1)}
        )

        assert fa.chamfer == math.inf
        assert fa.hausdorff == math.inf
        assert fa.n_scored == 0
        assert len(fa.flags) == 1
        assert fa.flags[0].startswith("frame coverage: no shared frame keys")

    def test_direct_non_dict_inputs(self, engine):
        a, b = _cloud(0), _cloud(1)
        fa = engine._frame_averaged_chamfer_hausdorff(a["pos"], b["pos"])

        assert fa.chamfer == math.inf
        assert fa.hausdorff == math.inf
        assert fa.n_scored == 0
        assert len(fa.flags) == 1
        assert fa.flags[0].startswith(
            "frame coverage: inputs are not per-frame dicts"
        )
        assert "Tensor" in fa.flags[0]


# ---------------------------------------------------------------------------
# Integration through compute_stage_metrics / sanity_check (unmocked)
# ---------------------------------------------------------------------------


class TestStageMetricsCoverage:
    """Coverage outcomes reach ``StageMetrics`` and ``sanity_check``."""

    def test_empty_intersection_is_inf_and_flagged(self, engine, labels):
        a, b = _cloud(0), _cloud(1)
        m = _stage(engine, {0: a}, {5: b}, labels, a["pos"])

        assert m.chamfer_distance == math.inf
        assert m.hausdorff_distance == math.inf
        assert m.normalized["chamfer"] == 0.0
        flags = engine.sanity_check(metrics=m)
        assert "non-finite metric: chamfer_distance=inf" in flags
        assert any(f.startswith("frame coverage:") for f in flags)

    def test_tensor_inputs_never_score_perfect(self, engine, labels):
        a = _cloud(0)
        m = _stage(engine, a["pos"], a["pos"], labels, a["pos"])

        assert m.normalized["chamfer"] < 1.0
        flags = engine.sanity_check(metrics=m)
        assert any(
            f.startswith("frame coverage:") and "not per-frame dicts" in f
            for f in flags
        )

    def test_partial_overlap_flag_reaches_stage_metrics(self, engine, labels):
        aligned = {k: _cloud(k) for k in (0, 1, 2)}
        target = {k: _cloud(10 + k) for k in (1, 2, 3)}
        fa = engine._frame_averaged_chamfer_hausdorff(aligned, target)
        m = _stage(engine, aligned, target, labels, aligned[0]["pos"])

        frame_flags = [f for f in m.coverage_flags if f.startswith("frame coverage:")]
        assert frame_flags == fa.flags
        assert len(frame_flags) == 1
        flags = engine.sanity_check(metrics=m)
        for f in fa.flags:
            assert f in flags

    def test_no_transforms_temporal_stability_unavailable(self, engine, labels):
        """WR-06: empty transforms never read as a perfect temporal_stability."""
        aligned = {k: _cloud(k) for k in (0, 1)}
        m = _stage(engine, aligned, aligned, labels, aligned[0]["pos"])

        assert m.temporal_stability == math.inf
        assert m.normalized["temporal_stability"] == 0.0
        assert "metric unavailable: temporal_stability (no per-frame transforms)" in m.coverage_flags
        flags = engine.sanity_check(metrics=m)
        assert "non-finite metric: temporal_stability=inf" in flags

    def test_stage_metrics_coverage_flags_default_empty(self):
        sm = StageMetrics(
            chamfer_distance=0.1,
            hausdorff_distance=0.2,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=1.0,
            knn_consistency=1.0,
        )
        assert sm.coverage_flags == []
