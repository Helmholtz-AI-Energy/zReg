"""Key-layout, distance_kwargs and cpd-guard tests for the pairwise sweep (Phase 61 DIST-03).

* The sweep walks frames by sorted key position, so non-zero-based ({5,6,7}) and
  gapped ({0,2,4}) key layouts give the same positional matrix as the same frames
  re-keyed {0,1,2} (previously a KeyError).
* ``_sanitize_pairwise_distance_matrix`` keeps later kwargs after a leading None,
  always checks the kwargs/metrics length and never mutates the caller's dict.
* ``create_pairwise_distance_matrix_given_rigid_rot`` rejects ``"cpd"`` with a
  ValueError (it runs no CPD) instead of failing with an IndexError.
"""

from zreg.algorithms import pairwise_distance_matrix as pm
from zreg.algorithms.pairwise_distance_matrix import _sanitize_pairwise_distance_matrix
from zreg.core.dataset import zRegPointCloud

import pytest
import torch


def _frames(n: int = 3) -> list[zRegPointCloud]:
    return [
        zRegPointCloud(pos=torch.randn(10, 3), label=torch.rand(10, 3), id=torch.arange(10))
        for _ in range(n)
    ]


@pytest.fixture
def frames():
    torch.manual_seed(0)
    return _frames(), _frames()


def _keyed(frames: list[zRegPointCloud], keys: list[int]) -> dict[int, zRegPointCloud]:
    return dict(zip(keys, frames))


def _run(which: str, x, y, window):
    if which == "create":
        return pm.create_pairwise_distance_matrix(
            x, y, window=window, normalize=False, distance_metric="euclidean"
        ).cost_matrix
    return pm.create_pairwise_distance_matrix_given_rigid_rot(
        x, y, rotation=torch.eye(3), translation=torch.zeros(3), scale=1.0,
        window=window, normalize=False, distance_metric="euclidean",
    )


@pytest.mark.parametrize("which", ["create", "given_rigid_rot"])
@pytest.mark.parametrize("layout", [[5, 6, 7], [0, 2, 4]], ids=["offset", "gapped"])
@pytest.mark.parametrize("window", [None, 1], ids=["nowindow", "window1"])
def test_nonzero_keys_equal_rekeyed(frames, which, layout, window):
    """Any sorted key layout yields the positional matrix of the re-keyed frames (DIST-03)."""
    fx, fy = frames
    ref = _run(which, _keyed(fx, [0, 1, 2]), _keyed(fy, [0, 1, 2]), window)
    got = _run(which, _keyed(fx, layout), _keyed(fy, layout), window)
    assert got.shape == (1, 3, 3)
    assert torch.equal(got, ref)


def test_nonzero_keys_cpd_rigid(frames):
    """cpd/rigid on keys {5,6,7}: same cost matrix as {0,1,2}, positional transform keys, sigma2 >= 0."""
    fx, fy = frames

    def run(keys):
        return pm.create_pairwise_distance_matrix(
            _keyed(fx, keys), _keyed(fy, keys),
            distance_metric="cpd", cpd_type="rigid", downsample_method=None, window=None,
        )

    ref = run([0, 1, 2])
    got = run([5, 6, 7])
    assert torch.equal(got.cost_matrix, ref.cost_matrix)
    positional = {(i, j) for i in range(3) for j in range(3)}
    assert set(got.stored_transforms) <= positional
    assert set(got.stored_transforms) == set(ref.stored_transforms)
    assert (got.cost_matrix >= 0).all()


def test_kwargs_leading_none_keeps_later(frames):
    """[None, {'p': 7}] keeps p=7 for the second metric (previously silently p=3)."""
    fx, fy = frames
    x, y = _keyed(fx, [0, 1, 2]), _keyed(fy, [0, 1, 2])
    fns, _, _ = _sanitize_pairwise_distance_matrix([None, {"p": 7}], ["euclidean", "minkowski"], None, x, y)
    assert fns[1].keywords["p"] == 7


def test_kwargs_wrong_length_with_leading_none_raises(frames):
    """A leading None no longer disables the kwargs/metrics length check."""
    fx, fy = frames
    x, y = _keyed(fx, [0, 1, 2]), _keyed(fy, [0, 1, 2])
    with pytest.raises(RuntimeError, match="len distance kwargs != len distance metrics"):
        _sanitize_pairwise_distance_matrix([None, {}, {}], ["euclidean", "minkowski"], None, x, y)


def test_kwargs_single_dict_not_mutated(frames):
    """The caller's kwargs dict does not gain the sanitizer's defaults (e.g. 'device')."""
    fx, fy = frames
    x, y = _keyed(fx, [0, 1, 2]), _keyed(fy, [0, 1, 2])
    kw = {"num_projs": 10}
    _sanitize_pairwise_distance_matrix(kw, "swd", "random", x, y)
    assert kw == {"num_projs": 10}


@pytest.mark.parametrize("kwargs", [None, [None], [None, None]], ids=["None", "[None]", "[None,None]"])
def test_kwargs_default_forms_still_accepted(frames, kwargs):
    """None, [None] and [None, None] still mean per-metric defaults."""
    fx, fy = frames
    x, y = _keyed(fx, [0, 1, 2]), _keyed(fy, [0, 1, 2])
    fns, _, _ = _sanitize_pairwise_distance_matrix(kwargs, ["euclidean", "minkowski"], None, x, y)
    assert len(fns) == 2
    assert fns[0].keywords == {}
    assert fns[1].keywords["p"] == 3


@pytest.mark.parametrize("metric", ["cpd", ["euclidean", "cpd"]], ids=["cpd", "euclidean+cpd"])
def test_given_rigid_rot_cpd_raises_valueerror(frames, metric):
    """given_rigid_rot runs no CPD, so 'cpd' is rejected up front with a ValueError."""
    fx, fy = frames
    with pytest.raises(ValueError, match="cpd"):
        pm.create_pairwise_distance_matrix_given_rigid_rot(
            _keyed(fx, [0, 1, 2]), _keyed(fy, [0, 1, 2]),
            rotation=torch.eye(3), translation=torch.zeros(3),
            normalize=False, distance_metric=metric,
        )
