"""Single-cdist regression tests for frame-averaged chamfer/hausdorff (DIST-05, U1-4).

``MetricsEngine._frame_averaged_chamfer_hausdorff`` used to call
``chamfer`` and ``hausdorff`` separately per frame, i.e. two ``torch.cdist``
calls (and two device syncs) per frame.  Both metrics are now derived from
one distance matrix via ``zreg.evaluation.chamfer_hausdorff``.  These tests
pin:

* exactly one ``torch.cdist`` call per shared, valid frame;
* bit-identical (``==`` / ``torch.equal``) values versus the separate
  ``chamfer`` / ``hausdorff`` calls;
* unchanged coverage-flag strings and their original key order, including
  mixed failure kinds (non-finite result before a validation error).

Only the collaborator ``torch.cdist`` is wrapped (counting spy that delegates
to the real function); the unit under test is never mocked.
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


def _pos(seed: int, n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic random (n, 3) positions."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, generator=gen, dtype=torch.float64) * (1.0 + seed)).to(
        dtype
    )


def _cloud(seed: int, n: int = 12, dtype: torch.dtype = torch.float32) -> zRegPointCloud:
    return zRegPointCloud(pos=_pos(seed, n, dtype))


def _overflow_pair() -> tuple[zRegPointCloud, zRegPointCloud]:
    """Finite float32 inputs whose pairwise distances overflow to inf."""
    return (
        zRegPointCloud(pos=torch.full((4, 3), 3e38)),
        zRegPointCloud(pos=torch.full((4, 3), -3e38)),
    )


def _two_call(a: zRegPointCloud, t: zRegPointCloud) -> tuple[float, float]:
    """Reference: the separate chamfer/hausdorff calls (pre-DIST-05 path)."""
    return chamfer(a["pos"], t["pos"]).item(), hausdorff(a["pos"], t["pos"]).item()


def _validation_error(a: zRegPointCloud, t: zRegPointCloud) -> str:
    with pytest.raises(ValueError) as info:
        chamfer(a["pos"], t["pos"])
    return str(info.value)


class _CdistSpy:
    """Counting wrapper around the real ``torch.cdist``."""

    def __init__(self, real):
        self.real = real
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.real(*args, **kwargs)


@pytest.fixture
def engine() -> MetricsEngine:
    return MetricsEngine(EvalConfig(data_path="x"))


# ---------------------------------------------------------------------------
# MetricsEngine._frame_averaged_chamfer_hausdorff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_frames", [1, 4])
def test_one_cdist_per_frame(engine, n_frames, monkeypatch):
    aligned = {k: _cloud(k, n=10 + k) for k in range(n_frames)}
    target = {k: _cloud(100 + k, n=8 + 2 * k) for k in range(n_frames)}
    spy = _CdistSpy(torch.cdist)
    monkeypatch.setattr(torch, "cdist", spy)

    result = engine._frame_averaged_chamfer_hausdorff(aligned, target)

    assert spy.calls == n_frames
    assert math.isfinite(result[0]) and math.isfinite(result[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_values_exactly_equal_two_call_reference(engine, dtype):
    keys = range(5)
    aligned = {k: _cloud(k, n=7 + 3 * k, dtype=dtype) for k in keys}
    target = {k: _cloud(50 + k, n=20 - 2 * k, dtype=dtype) for k in keys}
    refs = [_two_call(aligned[k], target[k]) for k in keys]

    result = engine._frame_averaged_chamfer_hausdorff(aligned, target)

    assert result[0] == statistics.mean(r[0] for r in refs)
    assert result[1] == statistics.mean(r[1] for r in refs)


def test_flags_unchanged_with_degenerate_frames(engine):
    # Coverage flags exist since Phase 59 (NUM-05); this guard pins them at
    # HEAD (the 6c1c37f baseline raised on degenerate frames instead).
    huge, neg_huge = _overflow_pair()
    nan_pos = _pos(4, 12).clone()
    nan_pos[0, 0] = float("nan")
    aligned = {
        0: _cloud(0),
        1: zRegPointCloud(pos=torch.zeros(0, 3)),
        2: _cloud(2),
        3: huge,
        4: zRegPointCloud(pos=nan_pos),
    }
    target = {
        0: _cloud(10),
        1: _cloud(11),
        2: _cloud(12),
        3: neg_huge,
        4: _cloud(14),
    }
    msg1 = _validation_error(aligned[1], target[1])
    msg4 = _validation_error(aligned[4], target[4])
    assert "finite" in msg4
    c3, h3 = _two_call(aligned[3], target[3])
    assert not (math.isfinite(c3) and math.isfinite(h3))
    refs = [_two_call(aligned[k], target[k]) for k in (0, 2)]

    result = engine._frame_averaged_chamfer_hausdorff(aligned, target)

    assert result[2] == 2
    assert list(result[3]) == [
        f"frame coverage: skipped degenerate frame 1: {msg1}",
        "frame coverage: skipped degenerate frame 3: "
        f"non-finite result (chamfer={c3}, hausdorff={h3})",
        f"frame coverage: skipped degenerate frame 4: {msg4}",
    ]
    assert result[0] == statistics.mean(r[0] for r in refs)
    assert result[1] == statistics.mean(r[1] for r in refs)


def test_flags_mixed_order_nonfinite_then_validation_error(engine):
    huge, neg_huge = _overflow_pair()
    aligned = {0: huge, 1: zRegPointCloud(pos=torch.zeros(0, 3)), 2: _cloud(2)}
    target = {0: neg_huge, 1: _cloud(11), 2: _cloud(12)}

    result = engine._frame_averaged_chamfer_hausdorff(aligned, target)
    flags = list(result[3])

    assert result[2] == 1
    assert len(flags) == 2
    assert flags[0].startswith(
        "frame coverage: skipped degenerate frame 0: non-finite result"
    )
    assert flags[1].startswith("frame coverage: skipped degenerate frame 1: ")
    assert "non-finite result" not in flags[1]
    assert result[0] == _two_call(aligned[2], target[2])[0]


# ---------------------------------------------------------------------------
# zreg.evaluation.chamfer_hausdorff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("percentile", [95.0, 50.0])
def test_chamfer_hausdorff_matches_separate_calls(squared, percentile, monkeypatch):
    from zreg.evaluation import chamfer_hausdorff

    s = _pos(3, 17)
    t = _pos(4, 11)
    # (1) references with the REAL torch.cdist, before the spy exists
    ref_c = chamfer(s, t, squared=squared)
    ref_h = hausdorff(s, t, percentile=percentile)
    # (2) only now install the counting spy
    spy = _CdistSpy(torch.cdist)
    monkeypatch.setattr(torch, "cdist", spy)
    # (3) combined call
    c, h = chamfer_hausdorff(s, t, squared=squared, percentile=percentile)
    # (4) exactly one distance matrix
    assert spy.calls == 1
    # (5) bit-identical values
    assert torch.equal(c, ref_c)
    assert torch.equal(h, ref_h)


@pytest.mark.parametrize(
    ("source", "target", "kwargs"),
    [
        (torch.zeros(4, 2), torch.zeros(4, 3), {}),
        (torch.zeros(4, 3), torch.zeros(4, 2), {}),
        (torch.zeros(0, 3), torch.ones(4, 3), {}),
        (torch.ones(4, 3), torch.zeros(0, 3), {}),
        (torch.full((4, 3), float("nan")), torch.ones(4, 3), {}),
    ],
)
def test_chamfer_hausdorff_validation_messages(source, target, kwargs):
    from zreg.evaluation import chamfer_hausdorff

    with pytest.raises(ValueError) as ref_c:
        chamfer(source, target)
    with pytest.raises(ValueError) as ref_h:
        hausdorff(source, target)
    assert str(ref_c.value) == str(ref_h.value)
    with pytest.raises(ValueError) as got:
        chamfer_hausdorff(source, target, **kwargs)
    assert str(got.value) == str(ref_c.value)


@pytest.mark.parametrize("percentile", [-1.0, 100.5])
def test_chamfer_hausdorff_percentile_message(percentile):
    from zreg.evaluation import chamfer_hausdorff

    s, t = _pos(1, 5), _pos(2, 6)
    with pytest.raises(ValueError) as ref:
        hausdorff(s, t, percentile=percentile)
    with pytest.raises(ValueError) as got:
        chamfer_hausdorff(s, t, percentile=percentile)
    assert str(got.value) == str(ref.value)
    assert str(got.value) == f"percentile must be in [0, 100], got {percentile}"
