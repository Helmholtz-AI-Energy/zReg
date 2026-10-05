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
        pytest.param(lambda: ICPRegistration(), id="icp", marks=pytest.mark.open3d),
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


# ---------------------------------------------------------------------------
# Task 2: exact SO(3) rotation and unequal point counts (SWD)
# ---------------------------------------------------------------------------

SO3_TOL = 1e-5


def _nearest_rotation(m):
    # Imported lazily so the Task-1 tests above still collect without it.
    from zreg.algorithms.swd_aligner import _nearest_rotation as impl

    return impl(m)


def _chamfer(a, b):
    from zreg.evaluation.alignment import chamfer

    return chamfer(a, b).item()


def _so3_errors(r):
    r = torch.as_tensor(np.asarray(r.detach().cpu()) if isinstance(r, torch.Tensor) else r).double()
    eye = torch.eye(3, dtype=torch.float64)
    return (r.T @ r - eye).abs().max().item(), abs(torch.det(r).item() - 1.0)


def _check_nearest_rotation(device, dtype):
    refl = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=dtype, device=device))
    r = _nearest_rotation(refl)
    assert r.dtype == dtype and r.device.type == torch.device(device).type
    orth, det = _so3_errors(r)
    assert orth < SO3_TOL
    assert det < 1e-6

    gen = torch.Generator().manual_seed(11)
    m = torch.randn(3, 3, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    r = _nearest_rotation(m)
    assert r.dtype == dtype and r.device.type == torch.device(device).type
    orth, det = _so3_errors(r)
    assert orth < SO3_TOL
    assert det < SO3_TOL


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nearest_rotation_corrects_reflection(dtype):
    _check_nearest_rotation("cpu", dtype)
    rot = _rot_z(37, dtype=dtype)
    assert torch.allclose(_nearest_rotation(rot), rot, atol=1e-6)


@requires_cuda
def test_nearest_rotation_cuda():
    _check_nearest_rotation("cuda", torch.float32)


@pytest.mark.parametrize("learning_rate", [None, 0.05], ids=["default-lr", "lr0.05"])
def test_swd_rotation_is_exact_so3(learning_rate):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(21)
    src = torch.randn(200, 3, generator=gen)
    tgt = src @ _rot_z(30).T + torch.tensor([0.3, -0.2, 0.1])
    kwargs = dict(
        variant="aswd",
        num_iterations=50,
        init_projs=20,
        step_projs=10,
        k=2.0,
        loop_rate_thresh=0.05,
        max_slices=500,
    )
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    result = SlicedWassersteinAligner(**kwargs).register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt))
    orth, det = _so3_errors(result.transform.matrix[:3, :3])
    assert orth < SO3_TOL
    assert det < SO3_TOL


def test_swd_per_step_projection_still_recovers_rotation():
    """Per-step SO(3) projection must not stall the rotation.

    Projecting the raw Adam update every step discards its (dominant)
    symmetric part and the rotation barely moves (measured 3.95 deg of 30 deg
    after 200 steps). The aligner therefore keeps only the tangent-space part
    of the rotation gradient; with it the 30 deg rotation is recovered
    (measured 30.00 deg).
    """
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(21)
    src = torch.randn(200, 3, generator=gen)
    tgt = src @ _rot_z(30).T + torch.tensor([0.3, -0.2, 0.1])
    result = SlicedWassersteinAligner(variant="aswd", num_iterations=200, learning_rate=1e-2).register(
        zRegPointCloud(pos=src), zRegPointCloud(pos=tgt)
    )
    r = result.transform.matrix[:3, :3].double()
    angle = math.degrees(math.atan2(r[1, 0].item(), r[0, 0].item()))
    assert abs(angle - 30.0) < 1.0, f"recovered {angle:.2f} deg, expected 30"


@pytest.mark.parametrize(
    "variant, variant_kwargs",
    [
        ("swd", {}),
        ("aswd", {}),
        ("oswd", {"num_projs": 3}),
        ("gswd", {}),
        ("pswd", {}),
        ("maxswd", {}),
    ],
)
def test_swd_unequal_n_all_variants(variant, variant_kwargs):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(31)
    src = torch.randn(300, 3, generator=gen)
    idx = torch.randperm(300, generator=gen)[:220]
    tgt = src[idx] @ _rot_z(20).T
    aligner = SlicedWassersteinAligner(
        variant=variant, num_iterations=200, learning_rate=1e-2, **variant_kwargs
    )
    result = aligner.register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt))
    m = result.transform.matrix
    assert _matrix_finite(m)
    orth, det = _so3_errors(m[:3, :3])
    assert orth < SO3_TOL
    assert det < SO3_TOL
    before = _chamfer(src.double(), tgt.double())
    after = _chamfer(_apply(m, src), tgt.double())
    assert after < before, f"{variant}: chamfer {after:.4f} not below identity {before:.4f}"
