"""Tests for the eval-side supported DTW distance set (Phase 63 D-08).

``SUPPORTED_DTW_DIST_FNS`` in ``eval.config`` must match what the real
``DynamicTimeWarping`` dispatch accepts when called the way ``AlignmentStage``
calls it (``downsample_method=None``).  ``cosine`` is a label-transfer
``dist_metric``, not a DTW distance; HPO search spaces that listed it scored
every cosine trial ``-inf``.

Names from ``eval.config`` are imported inside the test functions so the module
still collects on trees that predate the constant (RED evidence protocol).
"""

import pytest

from zreg.algorithms.dtw import DynamicTimeWarping
from zreg.data_generation.generators import generate_trajectory

_SUPPORTED_FOR_PARAMETRIZE = ("euclidean", "manhattan", "minkowski", "cpd")


def _run_dtw(distance_metric: str):
    """Run DTW exactly as AlignmentStage does, on tiny 3-frame clouds."""
    x = generate_trajectory(n_points=20, n_frames=3, seed=0)
    y = generate_trajectory(n_points=20, n_frames=3, seed=1)
    return DynamicTimeWarping(
        x=x,
        y=y,
        distance_metric=distance_metric,
        cpd_type="rigid" if distance_metric == "cpd" else None,
        window=2,
        downsample_method=None,
    ).compute()


def test_parametrize_list_matches_constant():
    """The parametrize list below mirrors the public constant (no silent drift)."""
    from eval.config import SUPPORTED_DTW_DIST_FNS

    assert tuple(SUPPORTED_DTW_DIST_FNS) == _SUPPORTED_FOR_PARAMETRIZE


@pytest.mark.parametrize("fn", _SUPPORTED_FOR_PARAMETRIZE)
def test_supported_value_accepted_by_dispatch(fn):
    """Every supported value runs through the real DTW dispatch."""
    from eval.config import SUPPORTED_DTW_DIST_FNS

    assert fn in SUPPORTED_DTW_DIST_FNS
    result = _run_dtw(fn)
    assert len(result.warping_path) > 0


def test_cosine_rejected_by_dispatch():
    """Characterization guard: the DTW dispatch has no cosine distance."""
    with pytest.raises(ValueError, match="Invalid distance function"):
        _run_dtw("cosine")


def test_supported_dtw_dist_fns_is_public():
    """The constant is part of the public eval.config API (Review LOW-10)."""
    import eval.config

    assert "SUPPORTED_DTW_DIST_FNS" in eval.config.__all__
