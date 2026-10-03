"""Warm-start and default_params contract of the HPO path (Phase 63 HPC-01, D-01, D-09).

``baseline_with_combined`` injects the merged selfcal/ground_truth calibration
as warm start.  These tests check that the injected params are actually the
first HPO trial, that merged params and upstream ``best_params.json`` artifacts
are validated, that ``config.default_params`` is used for every non-searched
key, and that every search strategy honours the warm start.

The tests use the real ``HyperparamOptimizer`` and ``run_all`` on a cheap
synthetic sanity-tier config; only ``PropulateSearch`` (not installed) and the
MPI transport are replaced.

PROVISIONAL DECISIONS (pending user confirmation, Phase 63 D-09):

(a) Propulate cannot seed its population.  The optimizer evaluates each tier's
    incoming warm-start seeds as ordinary trials on rank 0 before the
    Propulate search and logs one WARNING that the population is not seeded.
(b) ``HyperparamOptimizer`` merges ``config.default_params`` over its builtin
    defaults for non-searched keys, which changes HPO results for configs that
    fix keys in ``default_params``.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

# zreg.* before torch -> optuna -> eval.* (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud  # noqa: F401

import torch  # noqa: F401

import optuna  # noqa: F401  (import-order contract: torch -> optuna -> eval)

from eval.config import EvalConfig
from eval.runners import optimizer as optimizer_module
from eval.runners.optimizer import HyperparamOptimizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "baseline_experiments" / "scripts"
_OPT_LOGGER = "eval.runners.optimizer"

_OK_K = 3
_FAIL_K = 10000

# A complete stage-params dict whose k_neighbours (5) is the LAST grid value of
# the search spaces below, so a grid/sobol run only evaluates it first when the
# warm start is honoured.
_SEED = {
    "window_size": 5,
    "step": 1,
    "cpd_penalty": None,
    "dtw_dist_fn": "euclidean",
    "n_breakpoints": 5,
    "k_neighbours": 5,
    "dist_metric": "euclidean",
    "smoothing": 0.0,
    "threshold": 0.0,
}


def _cfg_dict(out_dir: Path, *, strategy="grid", tier="sanity", search_space=None, default_params=None) -> dict:
    """Cheap synthetic subsample_pair config (same data as tests/_failed_trials_mpi_worker.py)."""
    return {
        "data_path": str(out_dir / "unused.mat"),
        "output_dir": str(out_dir),
        "pipeline_mode": "synthetic",
        "run_alignment": True,
        "run_label_transfer": True,
        "transform_spec": {
            "type": "subsample_pair",
            "synthesize": True,
            "seed": 1,
            "n_classes": 3,
            "n_points": 80,
            "source_fraction": 0.8,
            "target_fraction": 0.8,
            "rotation_deg": 0.0,
            "rotation_axis": [0.0, 0.0, 1.0],
            "scale_factor": 1.0,
        },
        "tier": tier,
        "n_trials": 1,
        "search_strategy": strategy,
        "search_space": search_space if search_space is not None else {"k_neighbours": [3, 4, 5]},
        "default_params": dict(default_params or {}),
    }


def _cfg(out_dir: Path, **kw) -> EvalConfig:
    return EvalConfig(**_cfg_dict(out_dir, **kw))


def _write_yaml(path: Path, data: dict) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _import_run_all():
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import run_all

    return run_all


def _project(params: dict, keys) -> dict:
    return {k: params.get(k, "<missing>") for k in keys}


# --- Task 1: default_params consumption and the run_all path ----------------


def test_default_params_consumed_for_unsearched_keys(tmp_path) -> None:
    """A non-searched key comes from config.default_params, not a hardcoded default.

    k_neighbours=10000 exceeds the frame point count, so LabelTransferStage raises
    in every trial once the config's default is honoured.
    """
    cfg = _cfg(
        tmp_path / "hpo",
        search_space={"window_size": [3, 4]},
        default_params={"k_neighbours": _FAIL_K},
    )
    with pytest.raises(RuntimeError, match="trials failed"):
        HyperparamOptimizer(cfg).run()


def test_run_all_first_trial_is_injected_combined_params(tmp_path) -> None:
    """run_optimize_then_eval passes the injected params as warm start (HPC-01)."""
    run_all = _import_run_all()
    out = tmp_path / "hpo"
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", _cfg_dict(out, strategy="grid"))
    run_all.run_optimize_then_eval("t", cfg_path, force=True, dry_run=False, warmstart_params=dict(_SEED))
    history = _read_json(out / "search_history.json")
    assert _project(history[0]["params"], _SEED) == _SEED


def _upstream_tree(tmp_path: Path, *, stale_gt_kobitski: bool) -> tuple[Path, dict]:
    """configs_dir with the 5 upstream configs + the combined config and their artifacts."""
    configs_dir = tmp_path / "configs"
    rel = {
        "kobitski_sc_alignment": "selfcal/kobitski_ew06_alignment.yaml",
        "shah_sc_alignment": "selfcal/shah_alignment.yaml",
        "shah_sc_label_transfer": "selfcal/shah_label_transfer.yaml",
        "kobitski_gt_alignment": "ground_truth/kobitski_ew06.yaml",
        "shah_gt_both": "ground_truth/shah_sample1.yaml",
    }
    artifacts = {}
    for i, (key, r) in enumerate(rel.items()):
        out = tmp_path / "out" / key
        _write_yaml(configs_dir / r, _cfg_dict(out))
        params = {"k_neighbours": 3 + i % 3, "dtw_dist_fn": "euclidean"}
        if stale_gt_kobitski and key == "kobitski_gt_alignment":
            params["dtw_dist_fn"] = "cosine"
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_params.json").write_text(json.dumps(params))
        artifacts[key] = (out / "best_params.json", params)
    _write_yaml(
        configs_dir / "baseline_with_combined" / "ew06_vs_shah.yaml",
        _cfg_dict(tmp_path / "out" / "combined", default_params={"k_neighbours": 4}),
    )
    return configs_dir, artifacts


def test_combined_best_params_rejects_stale_cosine_artifact(tmp_path) -> None:
    """A stale best_params.json (dtw_dist_fn: cosine) fails loudly with a regeneration hint."""
    run_all = _import_run_all()
    configs_dir, artifacts = _upstream_tree(tmp_path, stale_gt_kobitski=True)
    stale_path = artifacts["kobitski_gt_alignment"][0]
    with pytest.raises(ValueError) as excinfo:
        run_all._combined_best_params(configs_dir)
    msg = str(excinfo.value)
    assert str(stale_path) in msg
    assert "dtw_dist_fn" in msg
    assert "--force" in msg
    assert "does not delete best_params.json" in msg


def test_combined_best_params_valid_artifacts_unchanged(tmp_path) -> None:
    """Valid artifacts pass validation and are returned unchanged."""
    run_all = _import_run_all()
    configs_dir, artifacts = _upstream_tree(tmp_path, stale_gt_kobitski=False)
    result = run_all._combined_best_params(configs_dir)
    assert len(result) == 6
    assert list(result[:5]) == [p for _, p in artifacts.values()]
    assert result[5] == {"k_neighbours": 4}


def test_run_optimize_then_eval_revalidates_warmstart(tmp_path) -> None:
    """Merged warm-start params go through EvalConfig validation before any trial."""
    run_all = _import_run_all()
    out = tmp_path / "hpo"
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", _cfg_dict(out, strategy="grid"))
    with pytest.raises(ValueError, match="dtw_dist_fn"):
        run_all.run_optimize_then_eval(
            "t", cfg_path, force=True, dry_run=False, warmstart_params={"dtw_dist_fn": "cosine"}
        )
    assert not (out / "search_history.json").exists()
    assert not (out / "run_config.yaml").exists()


# --- Task 2: every non-Propulate strategy evaluates the seed first ----------


@pytest.mark.parametrize(
    "strategy",
    [
        "grid",
        "random",
        # Sanity tier runs SANITY_N_TRIALS=5 < SOBOL_MIN_TRIALS=8 trials, so this
        # case exercises SobolSearch's documented RandomSearch fallback, not the
        # native Sobol path (see test_sobol_native_path_evaluates_seed_first).
        "sobol",
        "bayesian",
    ],
)
def test_first_trial_is_seed(tmp_path, strategy) -> None:
    """The warm-start seed is the first recorded trial for every in-process strategy."""
    from eval.search_strategies import SOBOL_MIN_TRIALS

    assert optimizer_module.SANITY_N_TRIALS < SOBOL_MIN_TRIALS
    cfg = _cfg(tmp_path / "hpo", strategy=strategy)
    result = HyperparamOptimizer(cfg, warm_start=[dict(_SEED)]).run()
    first = result.history[0].params
    assert _project(first, {"k_neighbours"}) == {"k_neighbours": 5}
    if strategy != "bayesian":
        assert first == _SEED
    else:
        # Optuna records only the search-space keys; the other keys come from
        # config.default_params / builtins, so the seed's extra keys are accepted.
        assert set(first) == {"k_neighbours"}


def test_sobol_native_path_evaluates_seed_first(caplog) -> None:
    """Native Sobol (n_trials >= SOBOL_MIN_TRIALS) evaluates the seed first, no fallback."""
    from eval.search_strategies import SOBOL_MIN_TRIALS, SobolSearch

    assert SOBOL_MIN_TRIALS == 8
    space = {"k_neighbours": [3, 4, 5], "window_size": [3, 4, 5]}
    seed_params = {"k_neighbours": 5, "window_size": 5}
    seen: list[dict] = []

    def objective(params: dict) -> float:
        seen.append(dict(params))
        return float(params["k_neighbours"])

    with caplog.at_level(logging.DEBUG, logger="eval.search_strategies"):
        SobolSearch().search(space, objective, n_trials=SOBOL_MIN_TRIALS, seed=0, warm_start=[seed_params])
    assert seen[0] == seed_params
    assert len(seen) >= 2
    assert not any("RandomSearch fallback" in r.getMessage() for r in caplog.records)


def test_bayesian_off_choice_seed_raises_up_front(tmp_path) -> None:
    """An Optuna-incompatible seed fails before any trial with a warm-start error."""
    out = tmp_path / "hpo"
    cfg = _cfg(out, strategy="bayesian")
    with pytest.raises(ValueError, match="warm-start"):
        HyperparamOptimizer(cfg, warm_start=[{"k_neighbours": 7}]).run()
    assert not (out / "search_history.json").exists()
