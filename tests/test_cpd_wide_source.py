"""CPD-09 / U2-10: CPD on point clouds wider than three columns.

Regression tests for the wide-source design completed in Phase 60 plan 04
(cross-AI review HIGH 1). On the 6c1c37f baseline:

- ``CoherentPointDrift.expectation_step`` ran ``cdist`` over every column
  (``posdims = t_source.shape[1]``), so a (N, 4) input produced a (N, 4) ``px``
  and extra columns (labels, metadata) changed the correspondence
  probabilities;
- ``NonRigidCPD`` / ``ConstrainedNonRigidCPD`` combined that (N, 4) ``px`` with
  the sliced (N, 3) source in the M-step and failed with a RuntimeError
  (``w``, the kernel normalisation and ``px_tilde`` also used all columns);
- ``RigidCPD`` / ``AffineCPD`` failed with a RuntimeError deep inside the
  M-step on (N, 4) input instead of rejecting it;
- (N, 2) and 1-D inputs raised RuntimeError / IndexError, or (non-rigid with
  (N, 2) source and target) ran and returned a wrong sigma2 because the
  M-step normalises by N_P * 3.

After the fix the E-step uses xyz only, the non-rigid variants carry extra
columns through unchanged, and one shared shape check rejects non-2-D input,
fewer than three columns (every variant) and more than three columns
(rigid/affine) with a clear ValueError.
"""

from zreg.algorithms import cpd

import pytest
import torch

DT = torch.float64


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _randn(*shape, gen):
    return torch.randn(*shape, generator=gen, dtype=DT)


def _wide_rigid_cls():
    """Test-local RigidCPD subclass that admits extra columns.

    ``expectation_step`` is inherited unchanged from CoherentPointDrift, so
    this exercises exactly the base E-step. Plain RigidCPD rejects (N, 4)
    input (see ``test_wide_source_rigid_estep_rejects_four_columns``); on the
    baseline the attribute is simply ignored.
    """

    class _WideRigid(cpd.RigidCPD):
        _ACCEPTS_EXTRA_COLUMNS = True

    return _WideRigid


def _assert_estep_equal(res4, res3):
    assert torch.allclose(res4.pmat, res3.pmat)
    assert torch.allclose(res4.pt1, res3.pt1)
    assert torch.allclose(res4.p1, res3.p1)
    assert torch.allclose(res4.px, res3.px)


# ---------------------------------------------------------------------------
# Task 1: xyz-only E-step
# ---------------------------------------------------------------------------


class TestWideSourceEstep:
    """The E-step computes correspondences from the xyz columns only."""

    def _run(self, w):
        gen = _gen(0)
        t_source = _randn(20, 4, gen=gen)
        target = _randn(25, 4, gen=gen)
        obj = _wide_rigid_cls()(_randn(20, 3, gen=gen), log_freq=-1)
        sigma2 = torch.tensor(0.5, dtype=DT)
        res4 = obj.expectation_step(t_source, target, sigma2=sigma2, sigma2_c=0.0, w=w)
        res3 = obj.expectation_step(
            t_source[:, :3], target[:, :3], sigma2=sigma2, sigma2_c=0.0, w=w
        )
        return res4, res3

    def test_wide_source_estep_px_is_xyz(self):
        res4, res3 = self._run(w=0.0)
        assert res4.px.shape == (20, 3)
        _assert_estep_equal(res4, res3)

    def test_wide_source_estep_w_outlier_term_uses_xyz_dim(self):
        """With w > 0 the outlier normaliser uses D = 3, not the column count."""
        res4, res3 = self._run(w=0.2)
        assert res4.px.shape == (20, 3)
        _assert_estep_equal(res4, res3)


# ---------------------------------------------------------------------------
# Task 1: rigid / affine reject extra columns
# ---------------------------------------------------------------------------


class TestRejectExtraColumns:
    """Variants whose M-step needs exactly xyz reject wider input clearly."""

    @staticmethod
    def _assert_rejects_wide(cls):
        gen = _gen(0)
        with pytest.raises(ValueError, match="exactly 3 columns"):
            cls(_randn(20, 4, gen=gen), log_freq=-1).registration(
                _randn(20, 4, gen=gen), maxiter=5
            )
        with pytest.raises(ValueError, match="exactly 3 columns"):
            cls(_randn(20, 3, gen=gen), log_freq=-1).registration(
                _randn(20, 4, gen=gen), maxiter=5
            )

    def test_wide_source_rigid_rejected(self):
        self._assert_rejects_wide(cpd.RigidCPD)

    def test_wide_source_affine_rejected(self):
        self._assert_rejects_wide(cpd.AffineCPD)

    def test_wide_source_rigid_estep_rejects_four_columns(self):
        """Plain RigidCPD: the class flag, not the E-step slicing, admits extra columns."""
        gen = _gen(0)
        obj = cpd.RigidCPD(_randn(20, 3, gen=gen), log_freq=-1)
        with pytest.raises(ValueError, match="exactly 3 columns"):
            obj.expectation_step(
                _randn(20, 4, gen=gen),
                _randn(25, 4, gen=gen),
                sigma2=torch.tensor(0.5, dtype=DT),
                sigma2_c=0.0,
                w=0.0,
            )


# ---------------------------------------------------------------------------
# Bad shapes: non-2-D and fewer than three columns
# ---------------------------------------------------------------------------

_BAD_CASES = ["short_source", "short_target", "short_both", "flat_source", "flat_target"]


def _bad_case(case):
    """Return (source, target, match) for a bad-shape case (float64, seed 0)."""
    gen = _gen(0)
    short = _randn(20, 2, gen=gen)
    flat = _randn(20, gen=gen)
    ok = _randn(20, 3, gen=gen)
    if case == "short_source":
        return short, ok, "at least 3 columns"
    if case == "short_target":
        return ok, short, "at least 3 columns"
    if case == "short_both":
        return short, short.clone(), "at least 3 columns"
    if case == "flat_source":
        return flat, ok, "2-D"
    if case == "flat_target":
        return ok, flat, "2-D"
    raise AssertionError(case)


class TestRejectBadShape:
    """Every variant rejects non-2-D input and fewer than three columns."""

    @pytest.mark.parametrize("case", _BAD_CASES)
    @pytest.mark.parametrize("cls_name", ["RigidCPD", "AffineCPD"])
    def test_wide_source_bad_shape_rejected_rigid_affine(self, cls_name, case):
        cls = getattr(cpd, cls_name)
        src, tgt, match = _bad_case(case)
        with pytest.raises(ValueError, match=match):
            cls(src, log_freq=-1).registration(tgt, maxiter=3)

    def test_wide_source_estep_rejects_short_input(self):
        gen = _gen(0)
        obj = cpd.RigidCPD(_randn(20, 3, gen=gen), log_freq=-1)
        sigma2 = torch.tensor(0.5, dtype=DT)
        with pytest.raises(ValueError, match="at least 3 columns"):
            obj.expectation_step(
                _randn(20, 2, gen=gen), _randn(25, 2, gen=gen),
                sigma2=sigma2, sigma2_c=0.0, w=0.0,
            )
        with pytest.raises(ValueError, match="2-D"):
            obj.expectation_step(
                _randn(20, gen=gen), _randn(25, 3, gen=gen),
                sigma2=sigma2, sigma2_c=0.0, w=0.0,
            )


# ---------------------------------------------------------------------------
# Task 2: non-rigid variants register wide sources end to end
# ---------------------------------------------------------------------------

_NONRIGID = {
    "NonRigidCPD": {},
    "ConstrainedNonRigidCPD": {"idx_source": [0, 1], "idx_target": [0, 1]},
}


def _make_nonrigid(cls_name, src):
    return getattr(cpd, cls_name)(src, log_freq=-1, **_NONRIGID[cls_name])


class TestWideSourceNonRigid:
    """NonRigidCPD / ConstrainedNonRigidCPD accept (N, >3) input."""

    @staticmethod
    def _pair(seed=0):
        gen = _gen(seed)
        src = _randn(20, 4, gen=gen)
        tgt = src + 0.05 * _randn(20, 4, gen=gen)
        return src, tgt

    def test_wide_source_nonrigid(self):
        src, tgt = self._pair()
        obj = _make_nonrigid("NonRigidCPD", src)
        obj.registration(tgt, maxiter=10)
        out = obj.transformation.transform(src)
        assert out.shape == (20, 4)
        assert torch.equal(out[:, 3], src[:, 3])
        assert torch.isfinite(out).all()

    def test_wide_source_constrained(self):
        src, tgt = self._pair()
        obj = _make_nonrigid("ConstrainedNonRigidCPD", src)
        obj.registration(tgt, maxiter=10)
        out = obj.transformation.transform(src)
        assert out.shape == (20, 4)
        assert torch.equal(out[:, 3], src[:, 3])
        assert torch.isfinite(out).all()
        assert obj.px_tilde.shape == (20, 3)

    @pytest.mark.parametrize("cls_name", list(_NONRIGID))
    def test_wide_source_extra_column_does_not_change_registration(self, cls_name):
        gen = _gen(1)
        src3 = _randn(20, 3, gen=gen)
        tgt3 = src3 + 0.05 * _randn(20, 3, gen=gen)
        col_a_src, col_a_tgt = 10 * _randn(20, 1, gen=gen), 10 * _randn(20, 1, gen=gen)
        col_b_src, col_b_tgt = 10 * _randn(20, 1, gen=gen), 10 * _randn(20, 1, gen=gen)

        def run(src, tgt):
            obj = _make_nonrigid(cls_name, src)
            obj.registration(tgt, maxiter=20)
            return obj.transformation.transform(src)[:, :3]

        out_a = run(torch.cat([src3, col_a_src], 1), torch.cat([tgt3, col_a_tgt], 1))
        out_b = run(torch.cat([src3, col_b_src], 1), torch.cat([tgt3, col_b_tgt], 1))
        out_3 = run(src3, tgt3)
        assert torch.allclose(out_a, out_b, atol=1e-6, rtol=0)
        assert torch.allclose(out_a, out_3, atol=1e-6, rtol=0)


class TestRejectBadShapeNonRigid:
    """Non-rigid variants still reject non-2-D and fewer-than-3-column input."""

    @pytest.mark.parametrize("case", _BAD_CASES)
    @pytest.mark.parametrize("cls_name", list(_NONRIGID))
    def test_wide_source_bad_shape_rejected_nonrigid(self, cls_name, case):
        src, tgt, match = _bad_case(case)
        with pytest.raises(ValueError, match=match):
            _make_nonrigid(cls_name, src).registration(tgt, maxiter=3)
        if case in ("short_source", "flat_source"):
            obj = _make_nonrigid(cls_name, None)
            with pytest.raises(ValueError, match=match):
                obj.set_source(src)
