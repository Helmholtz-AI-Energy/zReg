"""DTW cost for ``distance_metric="cpd"`` and ``cpd_type`` validation (D-02, CPD-02).

Before Phase 60 the cpd branch of ``create_pairwise_distance_matrix`` fed the CPD
objective ``reg.q`` into DTW. That objective is negative for rigid and affine CPD
(baseline 6c1c37f: rigid minimum -0.037, affine minimum -1881 on the Phase 60
reproduction), and after the 60-01 fix the rigid objective
``q = N_P*D/2*(1 + log sigma2)`` is large, negative and extensive in N. Negative or
offset costs bias the DTW dynamic programme towards long staircase paths.

Decision D-02: the DTW cost of one CPD registration is the converged ``sigma2``
(``_cpd_dtw_cost(reg)``), which is >= 0, ~0 at a perfect fit and independent of the
point count. ``reg.q`` stays the mathematically correct objective.

The module also checks that unsupported ``cpd_type`` values (typos, the constrained
non-rigid spellings) are rejected with ``ValueError`` instead of silently running rigid
CPD.

Test cost: the cost tests share one module-scoped cache that computes each cost matrix
once per ``cpd_type`` on 2x2-frame trajectories of 40 points.
"""

from copy import deepcopy
from pathlib import Path
import importlib
import re

import pytest

import zreg  # noqa: F401  (import zreg before torch)
import torch

from zreg.algorithms import cpd
from zreg.algorithms import pairwise_distance_matrix as pdm
from zreg.algorithms.dtw import DynamicTimeWarping
from zreg.core.dataset import zRegPointCloud
from zreg.preprocessing import downsampling
from zreg import utils


CPD_TYPES = ("rigid", "affine", "nonrigid")
N_POINTS = 40


def _make_trajectory(base, gen, n_frames, scale):
    traj = {}
    for i in range(n_frames):
        noise = torch.randn(base.shape, generator=gen)
        traj[i] = zRegPointCloud(
            pos=base + scale * (i + 1) * noise,
            label=torch.zeros(base.shape[0], 3),
            id=torch.arange(base.shape[0]),
        )
    return traj


def _make_pair(n_points=N_POINTS, n_frames=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(n_points, 3, generator=gen)
    x = _make_trajectory(base, gen, n_frames, 0.05)
    y = _make_trajectory(base, gen, n_frames, 0.05)
    return x, y


def _build_cpd(cpd_type, source):
    tf_params = {"device": source.device, "dtype": source.dtype}
    if cpd_type == "nonrigid":
        return cpd.NonRigidCPD(source=source, use_color=False, log_freq=-1)
    if cpd_type == "affine":
        return cpd.AffineCPD(source=source, use_color=False, tf_init_params=tf_params, log_freq=-1)
    return cpd.RigidCPD(source=source, use_color=False, tf_init_params=tf_params, log_freq=-1)


@pytest.fixture(scope="module")
def cpd_cost_case():
    """One 2x2-frame trajectory pair and a lazily filled cost-matrix cache per cpd_type."""
    x, y = _make_pair()
    cache = {}

    def get(cpd_type):
        if cpd_type not in cache:
            cache[cpd_type] = pdm.create_pairwise_distance_matrix(
                x,
                y,
                distance_metric="cpd",
                cpd_type=cpd_type,
                downsample_method=None,
                window=None,
            ).cost_matrix
        return cache[cpd_type]

    return x, y, get


class TestCpdDtwCost:
    """D-02: the cpd DTW cost is the converged sigma2 (finite, non-negative)."""

    @pytest.mark.parametrize("cpd_type", CPD_TYPES)
    def test_cpd_cost_matrix_finite_and_non_negative(self, cpd_cost_case, cpd_type):
        """Every cost entry is finite and >= 0.

        Fails on 6c1c37f for rigid and affine (reg.q negative); the nonrigid case is a
        consistency guard (NonRigidCPD already returned q = sigma2).
        """
        _, _, get = cpd_cost_case
        cm = get(cpd_type)
        assert torch.isfinite(cm).all(), cm
        assert (cm >= 0).all(), cm

    @pytest.mark.parametrize("cpd_type", CPD_TYPES)
    def test_cpd_cost_equals_converged_sigma2(self, cpd_cost_case, cpd_type):
        """cost_matrix[0, i, j] equals sigma2 of an identical, independently run CPD.

        The reference replicates the cpd branch exactly (normalize=True default:
        remove_outliers_knn + normalize_point_cloud on both clouds; downsample_method=None;
        source = x[i], target = y[j]). Fails on 6c1c37f for rigid and affine (the cost was
        reg.q); the nonrigid case is a consistency guard (q already equalled sigma2).
        """
        x, y, get = cpd_cost_case
        cm = get(cpd_type)
        i, j = 1, 0
        xi = deepcopy(x[i])
        yj = deepcopy(y[j])
        downsampling.remove_outliers_knn(xi, inplace=True)
        downsampling.remove_outliers_knn(yj, inplace=True)
        xi["pos"], _ = utils.normalize_point_cloud(xi["pos"])
        yj["pos"], _ = utils.normalize_point_cloud(yj["pos"])
        cpd_obj = _build_cpd(cpd_type, xi["pos"])
        reg = cpd_obj.registration(yj["pos"], w=0.0, maxiter=1000, tol=1e-5)
        expected = float(reg.sigma2)
        assert float(cm[0, i, j]) == pytest.approx(expected, rel=1e-5, abs=1e-12)

    def test_cpd_dtw_cost_helper_non_negative(self):
        """_cpd_dtw_cost returns reg.sigma2 (>= 0) for a real MstepResult. Fails on 6c1c37f (no helper)."""
        from zreg.algorithms.pairwise_distance_matrix import _cpd_dtw_cost

        x, y = _make_pair(n_frames=1, seed=3)
        cpd_obj = _build_cpd("rigid", x[0]["pos"])
        reg = cpd_obj.registration(y[0]["pos"], w=0.0, maxiter=20, tol=1e-5)
        cost = _cpd_dtw_cost(reg)
        assert isinstance(cost, torch.Tensor)
        assert float(cost) >= 0.0
        assert float(cost) == pytest.approx(float(reg.sigma2), rel=1e-12, abs=0.0)


class TestCpdTypeValidation:
    """Unsupported cpd_type values raise ValueError instead of silently running rigid CPD."""

    @pytest.mark.parametrize("cpd_type", ["nonrigid_constrained", "constrained_nonrigid", "Rigid", "bogus"])
    def test_unknown_cpd_type_rejected(self, cpd_type):
        """Fails on 6c1c37f: the unknown value fell through to rigid CPD without an error."""
        x, y = _make_pair(n_points=20, n_frames=1, seed=5)
        with pytest.raises(ValueError, match="cpd_type") as excinfo:
            pdm.create_pairwise_distance_matrix(
                x, y, distance_metric="cpd", cpd_type=cpd_type, downsample_method=None
            )
        if "constrained" in cpd_type:
            assert "constrained" in str(excinfo.value)

    def test_unknown_cpd_type_rejected_via_dtw(self):
        """DynamicTimeWarping.compute() surfaces the same ValueError. Fails on 6c1c37f."""
        x, y = _make_pair(n_points=20, n_frames=1, seed=6)
        dtw_obj = DynamicTimeWarping(
            x, y, distance_metric="cpd", cpd_type="nonrigid_constrained", downsample_method=None
        )
        with pytest.raises(ValueError, match="cpd_type"):
            dtw_obj.compute()

    @pytest.mark.parametrize(
        "cpd_type, metric",
        [(None, "euclidean"), ("rigid", "cpd"), ("affine", "cpd"), ("nonrigid", "cpd")],
    )
    def test_valid_cpd_types_accepted(self, cpd_type, metric):
        """Consistency guard (passes on 6c1c37f too): the four valid values still run."""
        x, y = _make_pair(n_points=30, n_frames=1, seed=7)
        res = pdm.create_pairwise_distance_matrix(
            x, y, distance_metric=metric, cpd_type=cpd_type, downsample_method=None
        )
        assert torch.isfinite(res.cost_matrix).all()


REPO_ROOT = Path(__file__).resolve().parent.parent
_STALE_TRANSFORMS = re.compile(r"(?<!core\.)zreg\.transforms")


class TestStalePaths:
    """CPD-09 (U2-13): docstrings/comments name real module paths."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "src/zreg/algorithms/dtw/core.py",
            "src/zreg/algorithms/pairwise_distance_matrix.py",
            "src/zreg/distance_metrics/_protocol.py",
        ],
    )
    def test_stale_paths_absent_in_zreg_sources(self, rel_path):
        text = (REPO_ROOT / rel_path).read_text()
        assert "zreg.distances" not in text
        assert not _STALE_TRANSFORMS.search(text)

    def test_stale_paths_alignment_comments(self):
        text = (REPO_ROOT / "eval" / "stages" / "alignment.py").read_text()
        # The baseline CR-01 comment wraps after "does not set", so match the line fragment.
        assert "NonRigidCPD does not set" not in text
        assert "``zreg.dtw." not in text

    def test_stale_paths_types_names_real_classes(self):
        text = (REPO_ROOT / "src/zreg/core/types.py").read_text()
        assert "RigidCPDTransformation" not in text

    def test_stale_paths_distance_metric_protocol_resolves(self):
        """Consistency guard: the path the docstrings now name exists."""
        mod = importlib.import_module("zreg.distance_metrics")
        assert hasattr(mod, "DistanceMetric")
