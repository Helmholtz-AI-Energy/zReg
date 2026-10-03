"""Regression tests for the CPD variants (Phase 60, plan 60-02).

Each test documents a defect observed on the 6c1c37f baseline:

- CPD-06: ``AffineTransformation()`` defaulted to ``t = ones(3)`` (not the
  identity); ``AffineCPD._initialize`` overwrote a warm start set by
  ``init_cpd_from_existing``; ``AffineCPD`` crashed on float64 sources because
  dtype/device were never propagated to the default transformation.
- CPD-04: ``ConstrainedNonRigidCPD`` left ``self.transformation`` as ``None``
  after registration.
- CPD-05: ``use_color=True`` at ``cpd_registration`` /
  ``init_cpd_from_existing`` and in the affine / non-rigid constructors failed
  with unrelated TypeError / ValueError instead of a clear rejection.
- CPD-09: ``rbf_kernel_matrix`` used ``exp(-d^2 / (2 beta^2))`` while CPD uses
  ``exp(-d^2 / (2 beta))``.

Tests marked CHARACTERIZATION pass on the baseline too; they guard against a
regression introduced by the fix itself.
"""

import zreg  # noqa: F401  (import zreg before torch)
from zreg.algorithms import cpd
from zreg.core import transforms

import pytest
import torch


@pytest.fixture
def source_target_pair():
    """Create a source and target point cloud pair."""
    source = torch.randn(50, 3)
    rot = torch.tensor([
        [0.866, -0.5, 0.0],
        [0.5, 0.866, 0.0],
        [0.0, 0.0, 1.0]
    ])  # 30 degree rotation around z
    t = torch.tensor([1.0, 2.0, 0.0])
    target = source @ rot.T + t
    return source, target


# ---------------------------------------------------------------------------
# CPD-06: AffineTransformation identity default, AffineCPD warm start / dtype
# ---------------------------------------------------------------------------


class TestAffineDefaults:
    """The default affine transformation and AffineCPD start are the identity."""

    def test_affine_transformation_default_is_identity(self):
        """AffineTransformation() maps the origin to the origin (baseline: ones)."""
        tf = transforms.AffineTransformation()
        out = tf.transform(torch.zeros(2, 3))
        assert torch.allclose(out, torch.zeros(2, 3))
        assert torch.equal(tf.t, torch.zeros(3))
        assert torch.equal(tf.b, torch.eye(3))

    def test_affine_cpd_initialize_identity(self, source_target_pair):
        """AffineCPD._initialize starts from b = I, t = 0 (baseline t = ones)."""
        src, tgt = source_target_pair
        obj = cpd.AffineCPD(src, log_freq=-1)
        res = obj._initialize(tgt)
        assert torch.equal(res.transformation.b, torch.eye(3))
        assert torch.equal(res.transformation.t, torch.zeros(3))


class TestAffineWarmStart:
    """init_cpd_from_existing warm-starts AffineCPD."""

    def test_affine_warm_start_from_existing(self, source_target_pair):
        """A pre-set affine transform survives _initialize (baseline: I and ones)."""
        src, tgt = source_target_pair
        b = 3.0 * torch.eye(3)
        t = torch.tensor([5.0, 5.0, 5.0])
        tf = transforms.AffineTransformation(b=b, t=t)
        obj = cpd.init_cpd_from_existing(tf, {"pos": src}, {"pos": tgt}, log_freq=-1)
        assert isinstance(obj, cpd.AffineCPD)
        res = obj.registration(tgt, maxiter=0)
        assert res.n_iters == 0
        assert torch.equal(res.transformation.b, b)
        assert torch.equal(res.transformation.t, t)


class TestAffineDtype:
    """AffineCPD builds its default transformation in the source dtype/device."""

    def test_affine_cpd_float64_runs(self, source_target_pair):
        """float64 sources work without tf_init_params (baseline: RuntimeError)."""
        src, tgt = source_target_pair
        obj = cpd.AffineCPD(src.double(), log_freq=-1)
        res = obj.registration(tgt.double(), maxiter=5)
        assert res.transformation.b.dtype == torch.float64
        assert res.transformation.t.dtype == torch.float64

    def test_affine_cpd_does_not_mutate_caller_dict(self, source_target_pair):
        """CHARACTERIZATION test: the caller's tf_init_params dict is not mutated.

        The 6c1c37f AffineCPD stores the dict without mutating it, so this test
        passes there as well. It pins that the new dtype/device merge copies the
        dict instead of writing into the caller's object.
        """
        src, _ = source_target_pair
        params = {}
        cpd.AffineCPD(src.double(), tf_init_params=params, log_freq=-1)
        assert params == {}

    def test_affine_cpd_deferred_source_float64(self, source_target_pair):
        """AffineCPD(None).set_source(src64) builds a float64 identity (baseline: RuntimeError)."""
        src, tgt = source_target_pair
        src64, tgt64 = src.double(), tgt.double()

        obj = cpd.AffineCPD(None, log_freq=-1)
        obj.set_source(src64)
        res = obj.registration(tgt64, maxiter=5)
        assert res.transformation.b.dtype == torch.float64
        assert res.transformation.t.dtype == torch.float64

        obj2 = cpd.AffineCPD(None, log_freq=-1)
        obj2.set_source(src64)
        init = obj2._initialize(tgt64)
        assert torch.equal(init.transformation.b, torch.eye(3, dtype=torch.float64))
        assert torch.equal(init.transformation.t, torch.zeros(3, dtype=torch.float64))

    def test_affine_preset_transform_survives_set_source(self, source_target_pair):
        """set_source never replaces or casts a pre-set transformation (CPD-06).

        Baseline-failing: on 6c1c37f _initialize overwrites the pre-set
        transform. After the fix it also serves as a regression guard that the
        new AffineCPD.set_source dtype/device refresh does not rebuild it.
        """
        src, tgt = source_target_pair
        src64, tgt64 = src.double(), tgt.double()
        b = 3.0 * torch.eye(3, dtype=torch.float64)
        t = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)

        obj = cpd.AffineCPD(None, log_freq=-1)
        obj.transformation = transforms.AffineTransformation(b=b, t=t, dtype=torch.float64)
        obj.set_source(src64)
        res = obj.registration(tgt64, maxiter=0)
        assert torch.equal(res.transformation.b, b)
        assert torch.equal(res.transformation.t, t)


# ---------------------------------------------------------------------------
# CPD-04: ConstrainedNonRigidCPD exposes its transformation
# ---------------------------------------------------------------------------


class TestConstrainedNonRigid:
    """ConstrainedNonRigidCPD sets self.transformation like NonRigidCPD."""

    def test_constrained_sets_instance_transformation(self, source_target_pair):
        """After registration obj.transformation is the result (baseline: None)."""
        src, tgt = source_target_pair
        obj = cpd.ConstrainedNonRigidCPD(
            src,
            idx_source=torch.tensor([0, 1]),
            idx_target=torch.tensor([0, 1]),
            log_freq=-1,
        )
        res = obj.registration(tgt, maxiter=10)
        assert obj.transformation is res.transformation
        assert torch.allclose(
            obj.transformation.transform(src), res.transformation.transform(src)
        )


# ---------------------------------------------------------------------------
# CPD-05: use_color=True is rejected where it cannot work
# ---------------------------------------------------------------------------


def _labelled_pair(n: int = 20):
    """Source/target dicts with xyz positions and 1-D class labels."""
    src = {"pos": torch.randn(n, 3), "label": torch.randint(0, 3, (n,))}
    tgt = {"pos": torch.randn(n, 3), "label": torch.randint(0, 3, (n,))}
    return src, tgt


class TestUseColorEntryPoints:
    """use_color=True raises NotImplementedError outside direct RigidCPD use."""

    @pytest.mark.parametrize(
        "tf_type_name", ["rigid", "affine", "nonrigid", "nonrigid_constrained"]
    )
    def test_use_color_cpd_registration_rejected(self, tf_type_name):
        """cpd_registration(use_color=True) is rejected (baseline: ValueError/RuntimeError)."""
        pc_s, pc_t = _labelled_pair()
        with pytest.raises(NotImplementedError, match="use_color"):
            cpd.cpd_registration(
                pc_s, pc_t, tf_type_name=tf_type_name, use_color=True, log_freq=-1
            )

    def test_use_color_init_cpd_from_existing_rejected(self):
        """init_cpd_from_existing(use_color=True) is rejected."""
        pc_s, pc_t = _labelled_pair()
        with pytest.raises(NotImplementedError, match="use_color"):
            cpd.init_cpd_from_existing(
                transforms.RigidTransformation(), pc_s, pc_t, use_color=True, log_freq=-1
            )

    @pytest.mark.parametrize(
        "cls_name", ["AffineCPD", "NonRigidCPD", "ConstrainedNonRigidCPD"]
    )
    def test_use_color_constructors_rejected(self, cls_name):
        """Affine / non-rigid constructors reject use_color=True (baseline: ValueError)."""
        cls = getattr(cpd, cls_name)
        with pytest.raises(NotImplementedError, match="use_color"):
            cls(torch.randn(20, 3), use_color=True, log_freq=-1)
