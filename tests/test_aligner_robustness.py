"""Robustness tests for the rigid aligners (DIST-04).

ICP and SWD validate their inputs through one shared helper,
``zreg.utils.registration_bounds``: empty, coincident (zero overall extent),
non-finite and mis-shaped clouds raise an informative ``ValueError`` instead of
a cryptic ``RuntimeError``, a silent "success" or an all-NaN matrix. Planar and
collinear clouds keep registering (degenerate means zero OVERALL extent, per
61-CONTEXT.md D-04 'Clarification (plan-review convergence, cycle 1 —
2026-10-03)').
"""

from zreg.algorithms import ICPRegistration, SlicedWassersteinAligner
from zreg.core.dataset import zRegPointCloud
from zreg import utils

import math

import numpy as np
import pytest
import torch


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _rot_z(deg, dtype=torch.float32):
    a = math.radians(deg)
    return torch.tensor(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def _apply(m, p):
    """Apply a 4x4 homogeneous matrix to (N, 3) points in CPU float64."""
    m = torch.as_tensor(np.asarray(m.detach().cpu() if isinstance(m, torch.Tensor) else m), dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64).cpu()
    homo = torch.cat([p, torch.ones(p.shape[0], 1, dtype=torch.float64)], dim=1)
    return (homo @ m.T)[:, :3]


def _matrix_finite(m):
    if isinstance(m, torch.Tensor):
        return bool(torch.isfinite(m).all())
    return bool(np.isfinite(np.asarray(m)).all())


def _aligners():
    return [
        pytest.param(lambda: ICPRegistration(), id="icp"),
        pytest.param(lambda: SlicedWassersteinAligner(variant="swd", num_iterations=5), id="swd"),
    ]


def _cloud(seed=0, n=50):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=gen)


def _degenerate_cases():
    valid = _cloud()
    nan_src = _cloud().clone()
    nan_src[3, 1] = float("nan")
    inf_tgt = _cloud(1).clone()
    inf_tgt[7, 2] = float("inf")
    coincident = torch.tensor([[1.0, 2.0, 3.0]]).repeat(50, 1)
    return [
        pytest.param(torch.zeros(0, 3), valid, ("source", "empty"), id="empty-source"),
        pytest.param(valid, torch.zeros(0, 3), ("target", "empty"), id="empty-target"),
        pytest.param(torch.zeros(0, 3), torch.zeros(0, 3), ("source", "empty"), id="both-empty"),
        pytest.param(coincident, valid, ("extent",), id="coincident"),
        pytest.param(torch.tensor([[1.0, 2.0, 3.0]]), valid, ("extent",), id="single-point"),
        pytest.param(nan_src, valid, ("finite",), id="nan-source"),
        pytest.param(valid, inf_tgt, ("finite",), id="inf-target"),
        pytest.param(torch.randn(50, 2), valid, ("source", "shape"), id="source-shape"),
        pytest.param(valid, torch.randn(50, 2), ("target", "shape"), id="target-shape"),
    ]


# ---------------------------------------------------------------------------
# Task 1: degenerate-input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_aligner", _aligners())
@pytest.mark.parametrize("src, tgt, words", _degenerate_cases())
def test_degenerate_rejected(make_aligner, src, tgt, words):
    aligner = make_aligner()
    with pytest.raises(ValueError) as excinfo:
        aligner.register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt))
    msg = str(excinfo.value)
    for word in words:
        assert word in msg, f"{word!r} not in error message: {msg}"


@pytest.mark.parametrize("make_aligner", _aligners())
@pytest.mark.parametrize("kind", ["planar", "collinear"])
def test_planar_and_collinear_accepted(make_aligner, kind):
    """Zero range on one or two axes is NOT degenerate.

    Guard for 61-CONTEXT.md D-04 'Clarification (plan-review convergence,
    cycle 1 — 2026-10-03)': only zero OVERALL extent is rejected; planar and
    collinear clouds keep registering with a finite matrix.
    """
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(3)
    if kind == "planar":
        src = torch.randn(200, 3, generator=gen)
        src[:, 2] = 0.0
        tgt = src @ _rot_z(10).T
    else:
        t = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)
        src = t * torch.tensor([[1.0, 2.0, 3.0]])
        tgt = src + torch.tensor([0.5, -0.3, 0.2])
    result = make_aligner().register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt))
    assert _matrix_finite(result.transform.matrix)


@pytest.mark.parametrize("make_aligner", _aligners())
@pytest.mark.parametrize("offset", [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], ids=["origin", "offset"])
def test_tiny_nonzero_cloud_accepted(make_aligner, offset):
    """Small-but-nonzero clouds are accepted (no eps threshold on the extent)."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(5)
    off = torch.tensor(offset, dtype=torch.float32)
    local = torch.randn(100, 3, generator=gen, dtype=torch.float32) * 1e-6
    src = off + local
    tgt = off + local @ _rot_z(10).T
    assert (src.max(dim=0).values - src.min(dim=0).values).max() > 0
    lo, hi = utils.registration_bounds(src, tgt)
    assert torch.isfinite(lo) and torch.isfinite(hi)
    result = make_aligner().register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt))
    assert _matrix_finite(result.transform.matrix)


def test_registration_bounds_matches_shared_bounds():
    a = _cloud(0, 40) * 3.0 + 1.0
    b = _cloud(1, 25) - 2.0
    lo, hi = utils.registration_bounds(a, b)
    lo_ref, hi_ref = utils.shared_bounds(a, b)
    assert torch.equal(lo, lo_ref)
    assert torch.equal(hi, hi_ref)
