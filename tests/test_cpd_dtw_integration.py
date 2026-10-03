"""Phase 60 integration gate for the CPD/DTW pipeline (CPD-01..CPD-09).

Each Phase 60 plan tests its own unit. This module checks that the fixes compose in
the real pipeline, with real CPD, DTW and AlignmentStage objects only (no test doubles):

- D-01: non-cancelling convergence window (four real q values, mean |delta q| < tol).
- D-02: the DTW cost for ``distance_metric="cpd"`` is the converged, non-negative CPD
  sigma2 (``_cpd_dtw_cost``), not the CPD objective q, which may be negative.
- D-03: rigid fixed-scale sigma2 per Myronenko & Song Eq. 23.
- D-04: RigidCPD and AffineCPD start from the identity; the historical Shah->Kobitski
  pose (``SHAH_KOBITSKI_EMPIRICAL_INIT``) is opt-in only and no pipeline path passes it
  (Assumption A1).
- D-05: every CPD variant is complete (exposes ``.transformation``), ``use_color=True``
  is rejected where it cannot work, ``cpd_type`` is validated, DTW save/load keeps the
  cpd config, and non-rigid CPD accepts sources wider than three columns.

Sizes are small and seeds fixed so the module stays cheap and deterministic. Tests
assert relations and bounds rather than exact iteration counts or cost values.
"""

import math
from pathlib import Path

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.algorithms import cpd
from zreg.algorithms.dtw import DynamicTimeWarping
from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_trajectory

import torch

from eval.stages import AlignmentStage

REPO_ROOT = Path(__file__).resolve().parent.parent

_ROTATION_DEG = 17.0
_SEED = 0


# ---------------------------------------------------------------------------
# Deterministic trajectory construction
# ---------------------------------------------------------------------------


def _rot_z(deg: float) -> torch.Tensor:
    a = math.radians(deg)
    return torch.tensor(
        [
            [math.cos(a), -math.sin(a), 0.0],
            [math.sin(a), math.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _distinct_frames(gen: torch.Generator) -> list[torch.Tensor]:
    """Three asymmetric, structurally distinct shapes (40-60 points each).

    The shapes stay distinguishable after the per-axis min-max normalisation that
    create_pairwise_distance_matrix applies before CPD (a helix, an L with unequal
    arms, and a dumbbell with unequal cluster sizes).
    """
    t = torch.linspace(0.0, 3.0 * math.pi, 50)
    helix = torch.stack([torch.cos(t), torch.sin(t), 0.25 * t], dim=1)
    helix = helix + 0.01 * torch.randn(50, 3, generator=gen)

    arm_long = torch.stack([torch.linspace(0.0, 1.0, 32), torch.zeros(32), torch.zeros(32)], dim=1)
    arm_short = torch.stack([torch.zeros(14), torch.linspace(0.08, 0.45, 14), torch.zeros(14)], dim=1)
    ell = torch.cat([arm_long, arm_short]) + 0.02 * torch.randn(46, 3, generator=gen)

    big = 0.06 * torch.randn(38, 3, generator=gen)
    small = 0.06 * torch.randn(12, 3, generator=gen) + torch.tensor([1.0, 0.7, 0.4])
    dumbbell = torch.cat([big, small])

    return [helix, ell, dumbbell]


def _pc(pos: torch.Tensor) -> zRegPointCloud:
    n = pos.shape[0]
    return zRegPointCloud(pos=pos, label=torch.zeros(n, dtype=torch.long), id=torch.arange(n))


def _rotated_trajectory_pair(seed: int = _SEED):
    """x-frame i is a distinct shape; y-frame i is x-frame i rotated 17 deg plus 0.005 noise."""
    gen = torch.Generator().manual_seed(seed)
    rot = _rot_z(_ROTATION_DEG)
    x, y = {}, {}
    for i, pos in enumerate(_distinct_frames(gen)):
        rotated = pos @ rot.T + 0.005 * torch.randn(pos.shape[0], 3, generator=gen)
        x[i] = _pc(pos.clone())
        y[i] = _pc(rotated)
    return x, y


@pytest.fixture(scope="module")
def rotated_pair():
    return _rotated_trajectory_pair()


def _square_cost(cost: torch.Tensor) -> torch.Tensor:
    return cost[0] if cost.ndim == 3 else cost


# ---------------------------------------------------------------------------
# DTW with the cpd metric (D-01, D-02, D-03, D-04)
# ---------------------------------------------------------------------------


def test_rigid_cpd_dtw_finds_diagonal_on_rotated_trajectory(rotated_pair):
    """Rigid CPD DTW recovers the diagonal on a 17-degree rotated trajectory.

    Combined effect of D-01..D-04: costs are converged sigma2 values (finite, >= 0),
    and each matching pair is cheaper than every non-matching pair in its row.
    """
    x, y = rotated_pair
    result = DynamicTimeWarping(
        x, y, distance_metric="cpd", cpd_type="rigid", downsample_method=None
    ).compute()

    assert result.warping_path == [(0, 0), (1, 1), (2, 2)]
    cost = _square_cost(result.cost_matrix)
    assert cost.shape == (3, 3)
    assert torch.isfinite(cost).all()
    assert (cost >= 0).all()
    for i in range(3):
        off_diag = torch.cat([cost[i, :i], cost[i, i + 1 :]])
        assert cost[i, i] < off_diag.min(), f"row {i}: {cost[i].tolist()}"


@pytest.mark.parametrize("cpd_type", ["rigid", "affine", "nonrigid"])
def test_cpd_dtw_distance_non_negative_all_types(rotated_pair, cpd_type):
    """The DTW distance built from cpd costs is finite and non-negative for every cpd_type (D-02)."""
    x, y = rotated_pair
    result = DynamicTimeWarping(
        x, y, distance_metric="cpd", cpd_type=cpd_type, downsample_method=None
    ).compute()
    assert math.isfinite(result.distance)
    assert result.distance >= 0


def test_dtw_rejects_constrained_cpd_type(rotated_pair):
    """Constrained non-rigid CPD needs correspondences, so DTW rejects it (CPD-09)."""
    x, y = rotated_pair
    with pytest.raises(ValueError, match="cpd_type"):
        DynamicTimeWarping(
            x, y, distance_metric="cpd", cpd_type="nonrigid_constrained", downsample_method=None
        ).compute()


def test_cpd_dtw_save_load_keeps_cpd_config(rotated_pair, tmp_path):
    """A saved and loaded cpd DTW result keeps its metric and a non-negative distance (CPD-07)."""
    x, y = rotated_pair
    dtw = DynamicTimeWarping(x, y, distance_metric="cpd", cpd_type="rigid", downsample_method=None)
    dtw.compute()
    path = tmp_path / "dtw_cpd.pt"
    dtw.save(path)

    loaded = DynamicTimeWarping.load(path)
    assert loaded.config is not None
    assert loaded.config["distance_metric"] == "cpd"
    assert math.isfinite(loaded.distance)
    assert loaded.distance >= 0


# ---------------------------------------------------------------------------
# AlignmentStage D-10 fallback (D-04, CPD-08)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n_points", "seed"), [(60, 2), (50, 3), (30, 4)])
def test_alignment_fallback_rigid_aligns_identical_frames(n_points, seed):
    """The D-10 fallback with rigid CPD maps identical source/target frames onto each other.

    This is a behavioural check of the pipeline path; identity initialisation itself is
    proven directly by the 60-01 ``_initialize`` / ``maxiter=0`` tests. With the old
    Shah->Kobitski pose the fallback stalled away from the target on these cases
    (max abs error 2.7-6.3 on 6c1c37f); from the identity it lands within ~1e-6.
    """
    source = generate_trajectory(n_points=n_points, n_frames=2, seed=seed)
    target = generate_trajectory(n_points=n_points, n_frames=2, seed=seed)
    source_sub = {i: source[k] for i, k in enumerate(sorted(source.keys()))}
    target_sub = {i: target[k] for i, k in enumerate(sorted(target.keys()))}

    aligned, _ = AlignmentStage._build_aligned_cloud(
        source=source,
        target=target,
        source_sub=source_sub,
        target_sub=target_sub,
        warp_path=[(0, 0), (1, 1)],
        cpd_penalty="rigid",
        stored_transforms={},
    )

    assert set(aligned.keys()) == set(target.keys())
    for k in target:
        torch.testing.assert_close(aligned[k]["pos"], target[k]["pos"], atol=1e-3, rtol=0.0)


def test_pose_not_injected_by_pipeline():
    """Assumption A1 guard: no pipeline CPD caller passes the opt-in Shah->Kobitski pose."""
    for rel in ("src/zreg/algorithms/pairwise_distance_matrix.py", "eval/stages/alignment.py"):
        text = (REPO_ROOT / rel).read_text()
        assert "SHAH_KOBITSKI_EMPIRICAL_INIT" not in text, rel


# ---------------------------------------------------------------------------
# CPD variants through the convenience entry point (CPD-04, CPD-05, CPD-06)
# ---------------------------------------------------------------------------


def _pair(n: int = 30, cols: int = 3, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    src = torch.randn(n, cols, generator=gen)
    tgt = src.clone()
    tgt[:, :3] = src[:, :3] @ _rot_z(10.0).T + 0.01 * torch.randn(n, 3, generator=gen)
    return src, tgt


def _variant_kwargs(tf_type_name: str) -> dict:
    if tf_type_name == "nonrigid_constrained":
        return {"idx_source": torch.tensor([0, 1]), "idx_target": torch.tensor([0, 1])}
    return {}


_VARIANTS = ["rigid", "affine", "nonrigid", "nonrigid_constrained"]


@pytest.mark.parametrize("tf_type_name", _VARIANTS)
def test_all_cpd_variants_run_end_to_end(tf_type_name):
    """Every variant registers through cpd_registration and exposes a transformation;
    use_color=True is rejected with NotImplementedError (CPD-04, CPD-05)."""
    src, tgt = _pair()
    result = cpd.cpd_registration(
        _pc(src), _pc(tgt), tf_type_name=tf_type_name, maxiter=10, log_freq=-1,
        **_variant_kwargs(tf_type_name),
    )
    assert result.transformation is not None
    moved = result.transformation.transform(src)
    assert moved.shape == src.shape
    assert torch.isfinite(moved).all()

    with pytest.raises(NotImplementedError, match="use_color"):
        cpd.cpd_registration(
            _pc(src), _pc(tgt), tf_type_name=tf_type_name, maxiter=10, log_freq=-1,
            use_color=True, **_variant_kwargs(tf_type_name),
        )


def test_affine_cpd_registration_starts_from_identity():
    """With maxiter=0 the affine result is the default start, which is the identity (CPD-06)."""
    src, tgt = _pair()
    result = cpd.cpd_registration(_pc(src), _pc(tgt), tf_type_name="affine", maxiter=0, log_freq=-1)
    torch.testing.assert_close(result.transformation.b, torch.eye(3))
    torch.testing.assert_close(result.transformation.t, torch.zeros(3))


def test_nonrigid_wide_source_end_to_end():
    """NonRigidCPD registers a (30, 4) cloud on xyz and carries column 3 through (CPD-09)."""
    src, tgt = _pair(cols=4)
    reg = cpd.NonRigidCPD(src, log_freq=-1)
    result = reg.registration(tgt, maxiter=10)
    moved = result.transformation.transform(src)
    assert moved.shape == (30, 4)
    assert torch.isfinite(moved).all()
    torch.testing.assert_close(moved[:, 3], src[:, 3])


def test_rigid_pose_constant_is_opt_in():
    """The historical pose stays available as an explicit opt-in (D-04)."""
    from zreg.algorithms.cpd import SHAH_KOBITSKI_EMPIRICAL_INIT

    src, tgt = _pair()
    reg = cpd.RigidCPD(
        src, tf_init_params={"rot": torch.tensor(SHAH_KOBITSKI_EMPIRICAL_INIT)}, log_freq=-1
    )
    result = reg.registration(tgt, maxiter=0)
    torch.testing.assert_close(result.transformation.rot, torch.tensor(SHAH_KOBITSKI_EMPIRICAL_INIT))
