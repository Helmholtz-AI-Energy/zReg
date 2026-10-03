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


# --- EvalConfig validator (Phase 63 D-08 part 3) -----------------------------


def test_search_space_cosine_rejected(tmp_path):
    """A search space listing cosine is a construction error naming key, value and allowed set."""
    from pydantic import ValidationError

    from eval.config import EvalConfig

    with pytest.raises(ValidationError, match=r"search_space\.dtw_dist_fn") as exc:
        EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            search_space={"dtw_dist_fn": ["euclidean", "cosine"]},
        )
    msg = str(exc.value)
    assert "'cosine'" in msg
    assert "euclidean" in msg


def test_default_params_swd_rejected(tmp_path):
    """default_params.dtw_dist_fn='swd' needs downsampling, which AlignmentStage disables."""
    from pydantic import ValidationError

    from eval.config import EvalConfig

    with pytest.raises(ValidationError, match=r"default_params\.dtw_dist_fn"):
        EvalConfig(data_path=str(tmp_path / "data.mat"), default_params={"dtw_dist_fn": "swd"})


def test_scalar_search_space_value_rejected(tmp_path):
    """A scalar search_space value is checked as a one-element list."""
    from pydantic import ValidationError

    from eval.config import EvalConfig

    with pytest.raises(ValidationError, match=r"search_space\.dtw_dist_fn"):
        EvalConfig(data_path=str(tmp_path / "data.mat"), search_space={"dtw_dist_fn": "cosine"})


def test_from_yaml_cosine_raises_eval_config_error(tmp_path):
    """from_yaml surfaces the validator as EvalConfigError carrying the key path."""
    from eval.config import EvalConfig, EvalConfigError

    p = tmp_path / "cfg.yaml"
    p.write_text(
        f"data_path: {tmp_path / 'data.mat'}\n"
        "search_space:\n"
        "  dtw_dist_fn: [euclidean, cosine]\n"
    )
    with pytest.raises(EvalConfigError, match="dtw_dist_fn"):
        EvalConfig.from_yaml(p)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_params": {"dtw_dist_fn": "cpd"}},
        {"search_space": {"dtw_dist_fn": ["euclidean", "cpd"]}},
        {"search_space": {"dtw_dist_fn": "manhattan"}},
        {},
        {"default_params": {"window_size": 10}, "search_space": {"step": [1, 2]}},
    ],
    ids=["default-cpd", "search-euclidean-cpd", "search-scalar", "no-key", "other-keys"],
)
def test_valid_values_construct(tmp_path, kwargs):
    """Supported values and configs without dtw_dist_fn construct fine."""
    from eval.config import EvalConfig

    EvalConfig(data_path=str(tmp_path / "data.mat"), **kwargs)


def test_model_validate_revalidates_merged_params(tmp_path):
    """model_validate re-runs the validator; model_copy(update=...) does not.

    This is why merged warm-start params must be re-validated through
    ``EvalConfig.model_validate`` (Phase 63-07, baseline_experiments/scripts/run_all.py)
    instead of being applied with ``model_copy``.
    """
    from pydantic import ValidationError

    from eval.config import EvalConfig

    cfg = EvalConfig(data_path=str(tmp_path / "data.mat"), default_params={"dtw_dist_fn": "euclidean"})
    merged = {**cfg.default_params, "dtw_dist_fn": "cosine"}
    with pytest.raises(ValidationError, match=r"default_params\.dtw_dist_fn"):
        EvalConfig.model_validate({**cfg.model_dump(), "default_params": merged})
    copied = cfg.model_copy(update={"default_params": merged})
    assert copied.default_params["dtw_dist_fn"] == "cosine"
