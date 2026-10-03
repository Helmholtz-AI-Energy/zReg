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
  pose, which stalled a 17-degree rotation of normalised clouds at rms 0.49.

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


class TestConvergence:
    """CPD-01: convergence is tested only on four chronological real q values.

    Baseline (6c1c37f) seeded the window with ``torch.arange(4)`` and compared
    values in ring-buffer storage order, so ``abs().diff().mean()`` could pass
    at iteration 0 or 1 and a 0.1% change of input scale flipped n_iters from
    12 to 1.
    """

    def test_convergence_predicate_rejects_oscillation(self):
        """A signed endpoint statistic would cancel to 0 on [0, 10, -10, 0]."""
        from zreg.algorithms.cpd.base import _q_window_converged

        window = collections.deque([0.0, 10.0, -10.0, 0.0], maxlen=4)
        assert _q_window_converged(window, 1.0) is False

    def test_convergence_predicate_basic(self):
        from zreg.algorithms.cpd.base import _q_window_converged

        def dq(values):
            return collections.deque(values, maxlen=4)

        assert _q_window_converged(dq([5.0, 5.0, 5.0, 5.0]), 1e-9) is True
        assert _q_window_converged(dq([5.0, 5.0, 5.0]), 1e-9) is False
        assert _q_window_converged(dq([5.0, math.nan, 5.0, 5.0]), 1e-9) is False
        assert _q_window_converged(dq([math.inf] * 4), 1e-9) is False
        assert _q_window_converged(dq([0.0, 1.0, 2.0, 3.0]), 1.0) is False
        assert _q_window_converged(dq([0.0, 1.0, 2.0, 3.0]), 1.01) is True

    @pytest.mark.parametrize(
        "cls", [cpd.RigidCPD, cpd.AffineCPD, cpd.NonRigidCPD], ids=lambda c: c.__name__
    )
    def test_convergence_waits_for_four_q_values(self, cls):
        """With an infinite tolerance the loop still runs four iterations (baseline 1)."""
        torch.manual_seed(0)
        src = torch.randn(60, 3)
        r = cls(src, log_freq=-1).registration(src.clone(), maxiter=50, tol=1e9)
        assert r.n_iters == 4

    def test_convergence_scale_invariance_reviewer_case(self):
        """A 0.1% input-scale change must not change n_iters (baseline 12 vs 1)."""
        results = {}
        for s in (1.828, 1.830):
            torch.manual_seed(0)
            src = torch.randn(60, 3) * s
            tgt = src + 0.02 * torch.randn(60, 3)
            r = cpd.NonRigidCPD(src, log_freq=-1).registration(tgt)
            results[s] = r
        n1, n2 = results[1.828].n_iters, results[1.830].n_iters
        assert n1 == n2
        assert n1 >= 4 and n2 >= 4
        for r in results.values():
            assert float(r.sigma2) < 1e-2

    def test_convergence_maxiter_below_window(self):
        """maxiter below the window size runs exactly maxiter iterations."""
        torch.manual_seed(0)
        src = torch.randn(60, 3)
        r = cpd.AffineCPD(src, log_freq=-1).registration(
            src.clone(), maxiter=2, tol=1e9
        )
        assert r.n_iters == 2
        obj = cpd.AffineCPD(src, log_freq=-1)
        init = obj._initialize(src.clone()).transformation
        r0 = cpd.AffineCPD(src, log_freq=-1).registration(
            src.clone(), maxiter=0, tol=1e9
        )
        assert r0.n_iters == 0
        assert torch.equal(r0.transformation.b, init.b)
        assert torch.equal(r0.transformation.t, init.t)


class TestUseColorTargetColors:
    """CPD-05 (base half): use_color=True requires valid target_colors.

    Baseline (6c1c37f) raised a TypeError deep inside ``cdist`` when
    target_colors was missing.
    """

    @staticmethod
    def _colour_cpd():
        torch.manual_seed(0)
        src = torch.randn(40, 3)
        tgt = src + 0.01 * torch.randn(40, 3)
        obj = cpd.RigidCPD(
            src, use_color=True, source_colors=torch.rand(40, 3), log_freq=-1
        )
        return obj, tgt

    def test_use_color_requires_target_colors(self):
        obj, tgt = self._colour_cpd()
        with pytest.raises(ValueError, match="target_colors"):
            obj.registration(tgt)

    def test_use_color_target_colors_row_mismatch(self):
        obj, tgt = self._colour_cpd()
        with pytest.raises(ValueError, match="target_colors"):
            obj.registration(tgt, target_colors=torch.rand(tgt.shape[0] + 5, 3))


class TestRigidDefaultInit:
    """CPD-08: a fresh RigidCPD starts from the identity rotation.

    Baseline (6c1c37f) overwrote the rotation with a hard-coded, non-orthogonal
    "Shah->Kobitski" pose (row norms 1, 1.118, 1.118), which stalled a
    17-degree rotation at rms 0.44. The pose is now the opt-in constant
    ``cpd.SHAH_KOBITSKI_EMPIRICAL_INIT``.
    """

    @staticmethod
    def _pair(dtype=torch.float32, n=50):
        torch.manual_seed(0)
        src = torch.randn(n, 3, dtype=dtype)
        tgt = src + 0.01 * torch.randn(n, 3, dtype=dtype)
        return src, tgt

    def test_rigid_default_init_is_identity(self):
        src, tgt = self._pair()
        init = cpd.RigidCPD(src, log_freq=-1)._initialize(tgt).transformation
        assert torch.equal(init.rot, torch.eye(3))
        assert torch.equal(init.t, torch.zeros(3))
        r = cpd.RigidCPD(src, log_freq=-1).registration(tgt, maxiter=0)
        assert torch.equal(r.transformation.rot, torch.eye(3))

    def test_rigid_default_init_pose_opt_in(self):
        pose = torch.tensor(cpd.SHAH_KOBITSKI_EMPIRICAL_INIT)
        src, tgt = self._pair()
        r = cpd.RigidCPD(src, tf_init_params={"rot": pose}, log_freq=-1).registration(
            tgt, maxiter=0
        )
        assert torch.equal(r.transformation.rot, pose.to(torch.float32))
        src64, tgt64 = self._pair(torch.float64)
        r64 = cpd.RigidCPD(
            src64, tf_init_params={"rot": pose}, log_freq=-1
        ).registration(tgt64, maxiter=0)
        assert r64.transformation.rot.dtype == torch.float64
        assert torch.equal(r64.transformation.rot, pose.to(torch.float64))

    def test_rigid_default_init_constant_is_not_a_rotation(self):
        """The constant is a general linear init (det = 1), not a rotation."""
        mat = torch.tensor(cpd.SHAH_KOBITSKI_EMPIRICAL_INIT, dtype=torch.float64)
        assert abs(float(torch.linalg.det(mat)) - 1.0) < 1e-12
        assert not torch.allclose(mat @ mat.T, torch.eye(3, dtype=torch.float64))

    def test_rigid_default_init_recovers_17_degree_rotation(self):
        """A 17-degree rotation of normalised clouds is recovered from identity.

        Both clouds are normalised to [-1, 1] as in the pipeline. Baseline
        (pose start): per-point rms 0.49. With the pose and the CPD-01/02/03
        fixes it still stalls (rms 0.62 at maxiter); from the identity it is 0.015.
        """

        def normalise(p):
            lo, hi = p.min(), p.max()
            return (p - lo) / (hi - lo) * 2 - 1

        torch.manual_seed(0)
        n = 1000
        x = torch.randn(n, 3, dtype=torch.float64)
        y = x @ _rot_x(17.0).T + 0.03 * torch.randn(n, 3, dtype=torch.float64)
        x, y = normalise(x), normalise(y)
        r = cpd.RigidCPD(x, log_freq=-1).registration(y, maxiter=1000, tol=1e-5)
        diff = r.transformation.transform(x) - y
        rms = float(torch.sqrt(torch.mean(torch.sum(diff**2, dim=1))))
        assert rms < 0.05

    def test_rigid_default_init_does_not_mutate_caller_dict(self):
        src, _ = self._pair()
        params = {"rot": torch.eye(3)}
        keys_before = set(params)
        cpd.RigidCPD(src, tf_init_params=params, log_freq=-1)
        assert set(params) == keys_before

    def test_rigid_default_init_deferred_source_float64(self):
        """Deferred source: the default identity is built in the source dtype.

        Intended as a regression guard for the identity change (without the
        set_source refresh, RigidCPD(None).set_source(src64) builds a float32
        identity and registration fails with a dtype mismatch). On 6c1c37f it
        fails anyway, because the baseline start is the historical pose, not
        eye(3).
        """
        src64, tgt64 = self._pair(torch.float64)
        obj = cpd.RigidCPD(None, log_freq=-1)
        obj.set_source(src64)
        rot = obj._initialize(tgt64).transformation.rot
        assert rot.dtype == torch.float64
        assert torch.equal(rot, torch.eye(3, dtype=torch.float64))
        obj2 = cpd.RigidCPD(None, log_freq=-1)
        obj2.set_source(src64)
        r = obj2.registration(tgt64, maxiter=5)
        assert r.transformation.rot.dtype == torch.float64
        assert r.transformation.t.dtype == torch.float64

    def test_rigid_default_init_preset_transform_survives_set_source(self):
        """REGRESSION GUARD: set_source never replaces a pre-set transformation.

        Passes on 6c1c37f; protects warm starts against the set_source override.
        """
        src64, tgt64 = self._pair(torch.float64)
        rot = _rot_zyx(10.0, 5.0, -3.0)
        obj = cpd.RigidCPD(None, log_freq=-1)
        obj.transformation = transforms.RigidTransformation(
            rot=rot, dtype=torch.float64
        )
        obj.set_source(src64)
        r = obj.registration(tgt64, maxiter=0)
        assert torch.equal(r.transformation.rot, rot)
