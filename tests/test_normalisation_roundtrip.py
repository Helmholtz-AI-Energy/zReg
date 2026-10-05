"""Regression tests for the normalise -> register -> denormalise round trip (NUM-01).

ICP and SWD register point clouds in a normalised [-1, 1] frame and return
``D_inv @ T @ D`` in original coordinates. Before NUM-01 both aligners
hand-built ``D_inv`` with translation ``lo`` instead of ``lo + range / 2`` and
denormalised with source bounds, so an identity registration moved
``[3, 5, 7]`` to ``[-1, 1, 3]``. These tests assert the exact inverse of the
shared helper pair and point-level round trips (never only the rotation
block).
"""

from zreg.algorithms import ICPRegistration, SlicedWassersteinAligner
from zreg.core.dataset import zRegPointCloud
from zreg.utils import (
    denormalization_matrix,
    normalization_matrix,
    normalize_point_cloud,
    shared_bounds,
)

import math

import numpy as np
import pytest
import torch

ANCHOR = [3.0, 5.0, 7.0]

# Fixed SWD aligner parameters (ASWD is stochastic; the tests seed torch and pin
# every parameter so they are deterministic and order-independent).
SWD_PARAMS = dict(
    variant="aswd",
    num_iterations=50,
    learning_rate=1e-3,
    init_projs=20,
    step_projs=10,
    k=2.0,
    loop_rate_thresh=0.05,
    max_slices=500,
)

# Seeded identity SWD round trip: measured max residual 4.77e-7 over 3 runs
# (float32 rounding; the loss gradient vanishes at identity). atol = 2x, rounded up.
SWD_IDENTITY_ATOL = 1e-6

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _apply(m, p):
    """Apply a 4x4 homogeneous matrix to (N, 3) points in CPU float64."""
    m = torch.as_tensor(np.asarray(m.detach().cpu() if isinstance(m, torch.Tensor) else m), dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64).cpu()
    homo = torch.cat([p, torch.ones(p.shape[0], 1, dtype=torch.float64)], dim=1)
    return (homo @ m.T)[:, :3]


def _anchor_cloud(seed=0, n=30):
    gen = torch.Generator().manual_seed(seed)
    rand = torch.rand(n, 3, generator=gen) * 8.0 + 2.0  # in [2, 10]
    return torch.cat([torch.tensor([ANCHOR]), rand], dim=0)


def _rot_z(deg):
    a = math.radians(deg)
    return torch.tensor(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


# ---------------------------------------------------------------------------
# Helper pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lo,hi", [(2.0, 10.0), (-5.0, 50.0)])
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
def test_denorm_times_norm_is_identity(lo, hi, dtype, atol):
    d = normalization_matrix(lo, hi, dtype=dtype)
    d_inv = denormalization_matrix(lo, hi, dtype=dtype)
    assert torch.allclose(d_inv @ d, torch.eye(4, dtype=dtype), atol=atol)


def test_matrix_matches_normalize_point_cloud():
    torch.manual_seed(3)
    p = torch.randn(40, 3) * 4 + 7
    q = torch.randn(35, 3) * 9 - 2
    lo, hi = shared_bounds(p, q)
    expected = normalize_point_cloud(p, min_vals=lo, max_vals=hi)[0]
    got = _apply(normalization_matrix(lo, hi, dtype=torch.float64), p)
    assert torch.allclose(got, expected.double(), atol=1e-5)


def test_shared_bounds():
    p = torch.tensor([[0.0, 1.0, 2.0], [3.0, -4.0, 5.0]])
    q = torch.tensor([[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    lo, hi = shared_bounds(p, q)
    assert lo.numel() == 1 and hi.numel() == 1
    assert lo.item() == -4.0
    assert hi.item() == 10.0


def test_helpers_follow_requested_device_cpu():
    lo = torch.tensor(2.0, dtype=torch.float32)
    hi = torch.tensor(10.0, dtype=torch.float32)
    for fn in (normalization_matrix, denormalization_matrix):
        m = fn(lo, hi, dtype=torch.float64, device="cpu")
        assert m.dtype == torch.float64
        assert m.device.type == "cpu"
        assert m.shape == (4, 4)


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------


@pytest.mark.open3d
def test_icp_identity_roundtrip_points():
    pts = _anchor_cloud()
    result = ICPRegistration().register(zRegPointCloud(pos=pts), zRegPointCloud(pos=pts.clone()))
    out = _apply(result.transform.matrix, pts)
    assert torch.allclose(out[0], torch.tensor(ANCHOR, dtype=torch.float64), atol=1e-4), out[0]


@pytest.mark.open3d
def test_icp_recovers_rigid_transform_with_different_bounds():
    torch.manual_seed(1)
    src = torch.randn(30, 3)
    tgt = src @ _rot_z(10.0).T + torch.tensor([3.0, -2.0, 1.0])
    result = ICPRegistration(max_iterations=100).register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt))
    m = result.transform.matrix
    residual = (_apply(m, src) - tgt.double()).abs().max().item()
    assert residual < 1e-3, residual
    det = float(np.linalg.det(np.asarray(m, dtype=np.float64)[:3, :3]))
    assert abs(det - 1.0) < 1e-3, det


@pytest.mark.open3d
def test_icp_uniform_translation():
    torch.manual_seed(2)
    src = torch.randn(30, 3)
    tgt = src + 5.0
    result = ICPRegistration().register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt))
    residual = (_apply(result.transform.matrix, src) - tgt.double()).abs().max().item()
    assert residual < 1e-3, residual


# ---------------------------------------------------------------------------
# SWD
# ---------------------------------------------------------------------------


def _swd_identity_matrix():
    torch.manual_seed(0)
    pts = _anchor_cloud()
    aligner = SlicedWassersteinAligner(**SWD_PARAMS)
    result = aligner.register(zRegPointCloud(pos=pts), zRegPointCloud(pos=pts.clone()))
    return pts, result.transform.matrix


def test_swd_identity_roundtrip_points():
    pts, m = _swd_identity_matrix()
    out = _apply(m, pts)
    assert torch.allclose(out[0], torch.tensor(ANCHOR, dtype=torch.float64), atol=SWD_IDENTITY_ATOL), out[0]


def test_swd_identity_roundtrip_is_deterministic():
    _, m1 = _swd_identity_matrix()
    _, m2 = _swd_identity_matrix()
    assert torch.allclose(m1.double(), m2.double(), atol=1e-6)


def test_swd_result_is_rigid():
    torch.manual_seed(42)
    src = torch.randn(100, 3) * 10 + 50
    tgt = src @ _rot_z(30.0).T + torch.tensor([1.0, 2.0, 0.5])
    torch.manual_seed(0)
    aligner = SlicedWassersteinAligner(**SWD_PARAMS)
    m = aligner.register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt)).transform.matrix
    r = m[:3, :3].double()
    assert abs(torch.det(r).item() - 1.0) < 0.05
    assert (r.T @ r - torch.eye(3, dtype=torch.float64)).abs().max().item() < 0.05


# ---------------------------------------------------------------------------
# CUDA (skipped where CUDA is unavailable; run on GPU nodes)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.open3d
def test_icp_cuda_input_returns_cpu_float32():
    pts = _anchor_cloud()
    result = ICPRegistration().register(
        zRegPointCloud(pos=pts.cuda()), zRegPointCloud(pos=pts.clone().cuda())
    )
    m = result.transform.matrix
    assert isinstance(m, np.ndarray)
    assert m.dtype == np.float32
    assert m.shape == (4, 4)
    out = _apply(m, pts.cpu())
    assert torch.allclose(out[0], torch.tensor(ANCHOR, dtype=torch.float64), atol=1e-4)

    torch.manual_seed(1)
    src = torch.randn(30, 3)
    tgt = src @ _rot_z(10.0).T + torch.tensor([3.0, -2.0, 1.0])
    result = ICPRegistration(max_iterations=100).register(
        zRegPointCloud(pos=src.cuda()), zRegPointCloud(pos=tgt.cuda())
    )
    m = result.transform.matrix
    assert isinstance(m, np.ndarray) and m.dtype == np.float32 and m.shape == (4, 4)
    assert (_apply(m, src) - tgt.double()).abs().max().item() < 1e-3


@requires_cuda
def test_helpers_on_cuda():
    lo = torch.tensor(2.0, device="cuda")
    hi = torch.tensor(10.0, device="cuda")
    d = normalization_matrix(lo, hi, dtype=torch.float64, device="cuda")
    d_inv = denormalization_matrix(lo, hi, dtype=torch.float64, device="cuda")
    assert d.device.type == "cuda" and d_inv.device.type == "cuda"
    assert torch.allclose(d_inv @ d, torch.eye(4, dtype=torch.float64, device="cuda"), atol=1e-12)

    d_cpu = normalization_matrix(lo, hi, dtype=torch.float64, device="cpu")
    d_inv_cpu = denormalization_matrix(lo, hi, dtype=torch.float64, device="cpu")
    assert d_cpu.device.type == "cpu" and d_inv_cpu.device.type == "cpu"
    assert torch.allclose(d_inv_cpu @ d_cpu, torch.eye(4, dtype=torch.float64), atol=1e-12)


def test_swd_recovers_translation_with_different_bounds():
    """CR-01: SWD must recover a pure translation in original coordinates.

    Shared bounds no longer pre-centre each cloud, so SWD has to start from the
    centroid offset; from a zero-translation start the 50-step Adam budget at
    lr=1e-3 recovered only a fraction of the offset.
    """
    torch.manual_seed(1)
    src = torch.rand(200, 3) * 10
    tgt = src + torch.tensor([5.0, -3.0, 2.0])
    torch.manual_seed(0)
    aligner = SlicedWassersteinAligner(**SWD_PARAMS)
    m = aligner.register(zRegPointCloud(pos=src), zRegPointCloud(pos=tgt)).transform.matrix
    residual = (_apply(m, src) - tgt.double()).abs().max().item()
    assert residual < 0.5, residual
    t = m[:3, 3].double()
    assert torch.allclose(t, torch.tensor([5.0, -3.0, 2.0], dtype=torch.float64), atol=0.5), t
