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


# ---------------------------------------------------------------------------
# Unequal point counts: 1-D quantile coupling (DIST-04, metric side)
# ---------------------------------------------------------------------------


def _is_non_decreasing(t):
    return bool((t[1:] >= t[:-1]).all())


def test_coupling_hand_reference():
    """The unscaled quantile coupling of x=[0,1,5], y=[0.5,2] costs 79/24."""
    ix, iy, w = sw_varients._coupling_indices(3, 2, device="cpu")
    xs = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
    ys = torch.tensor([0.5, 2.0], dtype=torch.float64)
    cost = (w * (xs[ix] - ys[iy]) ** 2).sum().item()
    assert cost == pytest.approx(79 / 24, rel=1e-12)
    assert w.sum().item() == pytest.approx(1.0, rel=1e-12)
    assert _is_non_decreasing(ix) and _is_non_decreasing(iy)
    assert int(ix.min()) >= 0 and int(ix.max()) <= 2
    assert int(iy.min()) >= 0 and int(iy.max()) <= 1


def test_coupling_equal_n_is_identity_pairing():
    ix, iy, w = sw_varients._coupling_indices(5, 5, device="cpu")
    assert torch.equal(ix, torch.arange(5))
    assert torch.equal(iy, torch.arange(5))
    assert torch.allclose(w, torch.full((5,), 0.2, dtype=w.dtype))


def test_coupling_symmetric():
    ix, iy, w = sw_varients._coupling_indices(7, 4, device="cpu")
    jx, jy, v = sw_varients._coupling_indices(4, 7, device="cpu")
    assert torch.equal(ix, jy)
    assert torch.equal(iy, jx)
    assert torch.equal(w, v)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_unequal_n_all_variants(name, swap):
    """Every variant returns a finite scalar for N != M (previously a size-mismatch crash)."""
    x, y = _clouds(n=80, m=60)
    if swap:
        x, y = y, x
    torch.manual_seed(0)
    result = _make(name)(x, y)
    assert result.ndim == 0
    assert torch.isfinite(result)


def test_unequal_n_duplicate_scaling():
    """Sort-sum variants use the legacy max(N, M) cardinality scaling (assumption A3).

    The underlying quantile coupling is exact, but the reported value is
    max(N, M) times the weighted coupling cost, so it depends on sample
    multiplicity by design: duplicating every target point (M = 2N) doubles the
    value. It is the legacy "sum over points" convention, NOT the normalised
    empirical Wasserstein distance.
    """
    x, y = _clouds(n=30)
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    y_dup = y.repeat(1, 2, 1)
    g =torch.Generator().manual_seed(5)
    P = torch.randn(1, 16, 3, generator=g)
    P = P / P.norm(dim=2, keepdim=True)
    base, _ = sw_varients.compute_practical_moments_sw_with_predefined_projections(x, y, P)
    dup, _ = sw_varients.compute_practical_moments_sw_with_predefined_projections(x, y_dup, P)
    assert dup.item() == pytest.approx(2 * base.item(), rel=1e-5)


@pytest.mark.parametrize("swap", [False, True])
def test_pswd_unequal_n_hand_reference(swap):
    """PSWD's unequal-N branch is a weighted mean with NO max(N, M) factor.

    All points lie on the x-axis, so for any projection with a non-zero
    x-component both clouds sort by x in the same direction; the quantile
    coupling costs 79/24 in x and 0 in y/z, and PSWD averages over the 3
    coordinates: 79/72.
    """
    x = torch.tensor([[0.0, 0, 0], [1, 0, 0], [5, 0, 0]], dtype=torch.float64)
    y = torch.tensor([[0.5, 0, 0], [2, 0, 0]], dtype=torch.float64)
    if swap:
        x, y = y, x
    torch.manual_seed(0)
    value = sw_varients.ProjectedWassersteinDistance(num_projs=20, device="cpu")(x, y)
    assert value.item() == pytest.approx(79 / 72, rel=1e-12)


def test_pswd_unequal_n_duplication_invariant():
    """PSWD (weighted mean) is invariant to duplicating every target point.

    Contrast with the sort-sum variants (test_unequal_n_duplicate_scaling),
    whose max(N, M)-scaled value doubles under the same duplication.
    """
    x, y = _clouds(n=80, m=60)
    y_dup = y.repeat(2, 1)
    torch.manual_seed(0)
    base = sw_varients.ProjectedWassersteinDistance(num_projs=20, device="cpu")(x, y)
    torch.manual_seed(0)
    dup = sw_varients.ProjectedWassersteinDistance(num_projs=20, device="cpu")(x, y_dup)
    assert dup.item() == pytest.approx(base.item(), rel=1e-5)


@pytest.mark.parametrize("case", ["x_empty", "y_empty", "both_empty"])
@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_empty_point_set_rejected(name, case):
    full = torch.randn(20, 3)
    empty = torch.zeros(0, 3)
    x = empty if case in ("x_empty", "both_empty") else full
    y = empty if case in ("y_empty", "both_empty") else full
    with pytest.raises(ValueError, match="empty"):
        _make(name)(x, y)


def test_coupling_indices_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        sw_varients._coupling_indices(0, 3, device="cpu")
    with pytest.raises(ValueError, match="empty"):
        sw_varients._coupling_indices(3, 0, device="cpu")


@pytest.mark.parametrize("name", ["swd", "pswd"])
def test_unequal_n_gradients_flow(name):
    torch.manual_seed(1)
    x = torch.randn(50, 3, requires_grad=True)
    y = torch.randn(30, 3)
    torch.manual_seed(0)
    _make(name)(x, y).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
