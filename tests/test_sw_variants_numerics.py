"""Numerical regression tests for the sliced-Wasserstein variants (DIST-02, DIST-04).

Covers:

* every variant follows the input dtype/device (float64 on CPU, CUDA when present);
* projections are still generated on CPU and then moved, so seeded CPU float32
  results are bit-identical to the values recorded before the fix (guard tests);
* MaxSWD honours ``max_sw_num_iters`` / ``max_sw_lr`` given to the constructor;
* MaxSWD's inner projection maximisation no longer leaks gradients into the
  caller's graph.
"""

from zreg.core.dataset import zRegPointCloud
from zreg.distance_metrics import sw_varients
from zreg.algorithms.pairwise_distance_matrix import _sanitize_pairwise_distance_matrix

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _make(name, device="cpu"):
    """Build one of the eight tested SW configurations on ``device``."""
    if name == "maxswd":
        return sw_varients.MaxSlicedWassersteinDistance(device=device)
    if name == "gswd_linear":
        return sw_varients.GeneralisedSlicedWassersteinDistance(num_projs=20, g_type="linear", device=device)
    if name == "gswd_circular":
        return sw_varients.GeneralisedSlicedWassersteinDistance(num_projs=20, g_type="circular", device=device)
    if name == "pswd":
        return sw_varients.ProjectedWassersteinDistance(num_projs=20, device=device)
    if name == "pswd_orth":
        return sw_varients.ProjectedWassersteinDistance(num_projs=3, orthogonal=True, device=device)
    if name == "swd":
        return sw_varients.SlicedWassersteinDistance(num_projs=20, device=device)
    if name == "aswd":
        return sw_varients.AdaptiveSlicedWassersteinDistance(device=device)
    if name == "oswd":
        return sw_varients.OrthogonalSlicedWassersteinDistance(num_projs=3, device=device)
    raise ValueError(name)


ALL_VARIANTS = ["maxswd", "gswd_linear", "gswd_circular", "pswd", "pswd_orth", "swd", "aswd", "oswd"]

# Seeded CPU float32 values recorded at HEAD 895b46b BEFORE the DIST-02 edit
# (sw_varients.py unchanged since 6c1c37f). float.hex() for exact comparison.
BIT_IDENTICAL = {
    "maxswd": "0x1.afa5600000000p+1",
    "gswd_linear": "0x1.fc2dba0000000p+1",
    "gswd_circular": "0x1.2fa8180000000p+1",
    "pswd": "0x1.5f676c0000000p+0",
    "pswd_orth": "0x1.51ebdc0000000p+0",
}


def _clouds(n=40, m=None, dtype=torch.float32, seed_x=11, seed_y=12):
    """Deterministic (n, 3) / (m, 3) clouds from private generators."""
    m = n if m is None else m
    g = torch.Generator().manual_seed(seed_x)
    x = torch.randn(n, 3, generator=g, dtype=dtype)
    g = torch.Generator().manual_seed(seed_y)
    y = torch.randn(m, 3, generator=g, dtype=dtype)
    return x, y


def _max_sw_inputs():
    torch.manual_seed(42)
    x = torch.randn(50, 3)
    torch.manual_seed(123)
    y = torch.randn(50, 3)
    return x, y


# ---------------------------------------------------------------------------
# dtype / device (DIST-02)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_float64_all_variants(name):
    """Every variant accepts CPU float64 and returns a finite float64 result."""
    x, y = _clouds(dtype=torch.float64)
    torch.manual_seed(0)
    result = _make(name)(x, y)
    assert result.dtype == torch.float64
    assert torch.isfinite(result).all()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_cuda_all_variants(name, dtype):
    """Every variant runs on CUDA inputs and returns a CUDA result of the input dtype."""
    x, y = _clouds(dtype=dtype)
    x, y = x.cuda(), y.cuda()
    torch.manual_seed(0)
    result = _make(name, device="cuda")(x, y)
    assert result.device.type == "cuda"
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("name", sorted(BIT_IDENTICAL))
def test_bit_identical_cpu_float32(name):
    """Guard: projections are generated on CPU exactly as before, then moved.

    Seeded CPU float32 results must equal the values recorded before the fix.
    """
    x, y = _clouds()
    torch.manual_seed(7)
    metric = _make(name)
    value = metric(x, y).item()
    assert value == float.fromhex(BIT_IDENTICAL[name])


# ---------------------------------------------------------------------------
# MaxSWD constructor kwargs (DIST-02)
# ---------------------------------------------------------------------------


def _max_sw_value(**ctor):
    x, y = _max_sw_inputs()
    torch.manual_seed(0)
    return sw_varients.MaxSlicedWassersteinDistance(device="cpu", **ctor)(x, y).item()


def test_max_sw_num_iters_ctor_changes_value():
    v0 = _max_sw_value(max_sw_num_iters=0)
    v200 = _max_sw_value(max_sw_num_iters=200, max_sw_lr=0.1)
    assert v0 != v200
    assert v200 > v0  # the inner loop maximises over the projection


def test_max_sw_lr_ctor_changes_value():
    v_lr0 = _max_sw_value(max_sw_num_iters=200, max_sw_lr=0.0)
    v_lr = _max_sw_value(max_sw_num_iters=200, max_sw_lr=0.1)
    assert v_lr0 != v_lr


def test_max_sw_ctor_attributes_stored():
    m = sw_varients.MaxSlicedWassersteinDistance(device="cpu", max_sw_num_iters=7, max_sw_lr=0.3)
    assert m.max_sw_num_iters == 7
    assert m.max_sw_lr == 0.3
    d = sw_varients.MaxSlicedWassersteinDistance(device="cpu")
    assert d.max_sw_num_iters == 50
    assert d.max_sw_lr == 1e-4


def test_max_sw_forward_kwarg_overrides_ctor():
    x, y = _max_sw_inputs()
    torch.manual_seed(0)
    overridden = sw_varients.MaxSlicedWassersteinDistance(device="cpu", max_sw_num_iters=0)(
        x, y, max_sw_num_iters=200, max_sw_lr=0.1
    )
    torch.manual_seed(0)
    plain = sw_varients.MaxSlicedWassersteinDistance(device="cpu", max_sw_num_iters=200, max_sw_lr=0.1)(x, y)
    assert overridden.item() == plain.item()


def test_sanitizer_maxswd_default_effective():
    """The sanitizer's maxswd default (100 inner iterations) now reaches the metric (assumption A2)."""
    torch.manual_seed(0)
    x = {0: zRegPointCloud(pos=torch.randn(20, 3))}
    y = {0: zRegPointCloud(pos=torch.randn(20, 3))}
    fns, _, _ = _sanitize_pairwise_distance_matrix(None, "maxswd", "random", x, y)
    assert fns[0].max_sw_num_iters == 100


# ---------------------------------------------------------------------------
# MaxSWD gradient leak (U3-new-1)
# ---------------------------------------------------------------------------


def test_maxswd_grad_leak():
    """The inner maximisation must not back-propagate into the caller's leaves."""
    x, y = _max_sw_inputs()
    R = torch.eye(3, requires_grad=True)
    xs = x @ R.T
    torch.manual_seed(0)
    loss = sw_varients.MaxSlicedWassersteinDistance(device="cpu", max_sw_num_iters=20, max_sw_lr=0.1)(xs, y)
    assert R.grad is None
    loss.backward()
    assert R.grad is not None
    assert torch.isfinite(R.grad).all()
