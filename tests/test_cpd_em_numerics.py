"""Regression tests for the CPD EM core numerics (CPD-01, CPD-02, CPD-03, CPD-05, CPD-08).

These tests pin the CPD EM loop to Myronenko & Song (2010). The symptoms at
6c1c37f were:

- CPD-03: the rigid fixed-scale sigma2 was ``(tr_xp1x + tr_yp1y - tr_atr) / (N_P*D)``,
  which gave sigma2 = 1.333 on a 200-point reproducer whose true residual
  variance is about 1e-4.
- CPD-02: the rigid q put the ``D*N_P/2*log(sigma2)`` term in the denominator,
  giving q = -0.0007 on a converged fit instead of ``N_P*D/2*(1 + log sigma2)``.
- CPD-01: the convergence window was seeded with ``torch.arange(4)``, so the
  check could pass at iteration 0 or 1; a 0.1% change in input scale
  (s = 1.828 -> 1.830) flipped NonRigidCPD from n_iters = 12 to n_iters = 1
  (sigma2 = 3.0).
- CPD-05: ``use_color=True`` without ``target_colors`` failed with a TypeError
  deep inside ``cdist``.
- CPD-08: RigidCPD started from a hard-coded non-orthogonal "Shah->Kobitski"
  pose, which stalled a 17-degree rotation at rms 0.44.

All tests use the real CPD code (no mocks). Symbols that do not exist at
6c1c37f are imported inside the test bodies so that a baseline run fails per
test instead of with a collection error.
"""

from zreg.algorithms import cpd
from zreg.core import transforms

import collections
import math

import pytest
import torch


def _rot_x(deg: float, dtype=torch.float64) -> torch.Tensor:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=dtype)


def _rot_zyx(az: float, ay: float, ax: float, dtype=torch.float64) -> torch.Tensor:
    def rz(d):
        a = math.radians(d)
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)

    def ry(d):
        a = math.radians(d)
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=dtype)

    return rz(az) @ ry(ay) @ _rot_x(ax, dtype)


def _identity_estep(target: torch.Tensor) -> "cpd.EstepResult":
    """EstepResult for P = identity (one-to-one correspondences)."""
    n = target.shape[0]
    ones = torch.ones(n, dtype=target.dtype)
    return cpd.EstepResult(
        ones, ones, target.clone(), torch.tensor(float(n), dtype=target.dtype), None
    )


def _hand_pair(seed: int = 0, n: int = 50):
    torch.manual_seed(seed)
    y = torch.randn(n, 3, dtype=torch.float64)
    r0 = _rot_zyx(20.0, -10.0, 5.0)
    t0 = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    x = y @ r0.T + t0 + 0.01 * torch.randn(n, 3, dtype=torch.float64)
    return y, x


class TestFixedScaleSigma2:
    """CPD-03: rigid fixed-scale sigma2 follows M&S Eq. 23 with s = 1."""

    def test_fixed_scale_sigma2_hand_reference(self):
        """With P = I, sigma2 equals the mean squared residual per coordinate.

        Baseline (6c1c37f) used ``tr_xp1x + tr_yp1y - tr_atr``, which is off by
        ``tr_yp1y - tr_atr`` and fails here.
        """
        y, x = _hand_pair()
        n = y.shape[0]
        res = cpd.RigidCPD._maximization_step(
            y, x, _identity_estep(x), None, update_scale=False
        )
        rot, t = res.transformation.rot, res.transformation.t
        resid = torch.sum((x - y @ rot.T - t) ** 2)
        expected = float(resid) / (n * 3)
        assert math.isclose(float(res.sigma2), expected, rel_tol=1e-10)

    def test_fixed_scale_sigma2_reviewer_reproducer(self):
        """The reviewer's 200-point case converges to a small sigma2 (baseline 1.333)."""
        torch.manual_seed(0)
        src = torch.randn(200, 3, dtype=torch.float64)
        r_small = _rot_zyx(3.0, 2.0, -2.0)
        tgt = src @ r_small.T + 0.01 * torch.randn(200, 3, dtype=torch.float64)
        reg = cpd.RigidCPD(src, update_scale=False, log_freq=-1)
        # Start from the identity explicitly (a pre-set transformation is never
        # replaced by _initialize), so this test isolates the sigma2 formula
        # from the CPD-08 default-pose change.
        reg.transformation = transforms.RigidTransformation(dtype=torch.float64)
        r = reg.registration(tgt, maxiter=1000, tol=1e-8)
        sigma2 = float(r.sigma2)
        assert sigma2 < 1e-3
        resid = float(torch.sum((r.transformation.transform(src) - tgt) ** 2)) / (
            200 * 3
        )
        assert abs(sigma2 - resid) / resid < 1e-6
        assert float((r.transformation.rot - r_small).abs().max()) < 1e-2


class TestRigidQ:
    """CPD-02: rigid q is the M&S negative log-likelihood (log term added).

    q = residual/(2*sigma2) + D*N_P/2*log(sigma2). After an M-step the
    residual equals N_P*D*sigma2, so q == N_P*D/2*(1 + log sigma2), which may
    be negative. Baseline (6c1c37f) put the log term in the denominator and
    gave q of order -1e-5 on a converged fit.
    """

    @pytest.mark.parametrize("update_scale", [True, False])
    def test_rigid_q_m_step_identity(self, update_scale):
        torch.manual_seed(0)
        n = 200
        src = torch.randn(n, 3, dtype=torch.float64)
        tgt = src + 0.01 * torch.randn(n, 3, dtype=torch.float64)
        reg = cpd.RigidCPD(src, update_scale=update_scale, log_freq=-1)
        r = reg.registration(tgt, maxiter=200, tol=1e-8)
        sigma2 = float(r.sigma2)
        expected = n * 3 / 2 * (1 + math.log(sigma2))
        assert math.isclose(float(r.q), expected, rel_tol=1e-9)

    @pytest.mark.parametrize("update_scale", [True, False])
    def test_rigid_q_direct_m_step(self, update_scale):
        y, x = _hand_pair()
        n = y.shape[0]
        res = cpd.RigidCPD._maximization_step(
            y, x, _identity_estep(x), None, update_scale=update_scale
        )
        resid = float(torch.sum((x - res.transformation.transform(y)) ** 2))
        sigma2 = float(res.sigma2)
        expected = resid / (2 * sigma2) + n * 3 / 2 * math.log(sigma2)
        assert math.isclose(float(res.q), expected, rel_tol=1e-9)
