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
    # 63-REVIEW WR-04: a recovery the sbatch launchers can actually perform.
    assert "ZREG_FORCE=1" in msg
    assert str(stale_path.parent / "eval_report.json") in msg
    # 63-REVIEW iter-2 WR-02: the single-run redo must also drop the checkpoints.
    assert "ZREG_CLEAR_CHECKPOINTS=1" in msg
    assert "*.pickle" in msg


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


def test_project_to_search_space_snaps_off_grid_values() -> None:
    """CR-01 / WR-02: averaged and foreign seed values are snapped onto the choices."""
    run_all = _import_run_all()
    space = {"k_neighbours": [3, 5, 10], "smoothing": [0.0, 0.1, 0.3], "dtw_dist_fn": ["cpd"], "window_size": [10]}
    seed = {"k_neighbours": 6, "smoothing": 0.15, "dtw_dist_fn": "euclidean", "window_size": 5, "n_breakpoints": 7}
    projected, changed = run_all._project_to_search_space(seed, space)
    assert projected == {
        "k_neighbours": 5,
        "smoothing": 0.1,  # tie 0.1 / 0.3 -> first listed
        "dtw_dist_fn": "cpd",
        "window_size": 10,
        "n_breakpoints": 7,  # not searched: unchanged
    }
    assert set(changed) == {"k_neighbours", "smoothing", "dtw_dist_fn", "window_size"}
    on_grid = {"k_neighbours": 3, "smoothing": 0.3}
    assert run_all._project_to_search_space(on_grid, space) == (on_grid, {})


def test_stale_checkpoint_search_space_fails_with_clear_hint(tmp_path) -> None:
    """63-REVIEW WR-03: checkpoints from another search space stop the run with a recovery hint."""
    run_all = _import_run_all()
    out = tmp_path / "hpo"
    # A finished earlier run under k_neighbours [3, 4] that left a checkpoint.
    assert run_all._checkpoint_search_space_error(out, {"k_neighbours": [3, 4]}) is None
    assert (out / run_all.SEARCH_SPACE_FINGERPRINT).exists()
    ckpt = out / "island_0_ckpt.pickle"
    ckpt.write_bytes(b"x")
    assert run_all._checkpoint_search_space_error(out, {"k_neighbours": [3, 4]}) is None

    cfg_path = _write_yaml(tmp_path / "cfg.yaml", _cfg_dict(out, strategy="grid", default_params=_SEED))  # [3, 4, 5]
    with pytest.raises(RuntimeError, match="ZREG_CLEAR_CHECKPOINTS=1") as excinfo:
        run_all.run_optimize_then_eval("t", cfg_path, force=True, dry_run=False)
    assert "--clear-checkpoints" in str(excinfo.value)
    assert ckpt.exists(), "checkpoints must never be deleted without the opt-in"
    assert not (out / "run_config.yaml").exists()

    # The opt-in clears them and records the new search space.
    run_all.run_optimize_then_eval("t", cfg_path, force=True, dry_run=False, clear_checkpoints=True)
    assert not ckpt.exists()
    assert (out / "search_history.json").exists()
    assert (out / run_all.SEARCH_SPACE_FINGERPRINT).read_text() == run_all._search_space_fingerprint(
        {"k_neighbours": [3, 4, 5]}
    )


def _completed_run(tmp_path: Path, run_all) -> tuple[Path, Path, Path]:
    """A finished grid run whose output_dir also holds a (fake) Propulate checkpoint."""
    out = tmp_path / "hpo"
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", _cfg_dict(out, strategy="grid", default_params=_SEED))
    run_all.run_optimize_then_eval("t", cfg_path, force=False, dry_run=False)
    report = out / "eval_report.json"
    assert report.exists()
    ckpt = out / "island_0_ckpt.pickle"
    ckpt.write_bytes(b"x")
    return out, cfg_path, ckpt


def test_force_redo_discards_completed_runs_checkpoints(tmp_path) -> None:
    """63-REVIEW iter-2 WR-02: --force on a completed run is a fresh HPO, not a resume."""
    run_all = _import_run_all()
    out, cfg_path, ckpt = _completed_run(tmp_path, run_all)
    (out / "eval_report.json").write_text("{}")  # stands in for the old report

    run_all.run_optimize_then_eval("t", cfg_path, force=True, dry_run=False)
    assert not ckpt.exists(), "a forced redo must not resume the finished search"
    assert (out / "eval_report.json").read_text() != "{}", "the redo writes a new report"
    assert (out / run_all.SEARCH_SPACE_FINGERPRINT).exists()


def test_force_dry_run_and_unforced_skip_keep_checkpoints(tmp_path) -> None:
    """Neither a dry run nor a skipped run touches the completed run's files."""
    run_all = _import_run_all()
    out, cfg_path, ckpt = _completed_run(tmp_path, run_all)
    run_all.run_optimize_then_eval("t", cfg_path, force=True, dry_run=True)
    run_all.run_optimize_then_eval("t", cfg_path, force=False, dry_run=False)
    assert ckpt.exists()
    assert (out / "eval_report.json").exists()


def test_force_on_unfinished_run_resumes_checkpoints(tmp_path) -> None:
    """Without eval_report.json the run is unfinished: --force keeps resuming its checkpoints."""
    run_all = _import_run_all()
    out, cfg_path, ckpt = _completed_run(tmp_path, run_all)
    (out / "eval_report.json").unlink()
    run_all.run_optimize_then_eval("t", cfg_path, force=True, dry_run=False)
    assert ckpt.exists(), "checkpoints of an unfinished run are only cleared on opt-in"
    assert (out / "eval_report.json").exists()


def test_unrecorded_checkpoints_resume_with_warning(tmp_path, caplog) -> None:
    """Checkpoints from before the record existed resume, with a WARNING naming the opt-in."""
    run_all = _import_run_all()
    out = tmp_path / "hpo"
    out.mkdir()
    (out / "island_0_ckpt.pickle").write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="run_all"):
        assert run_all._checkpoint_search_space_error(out, {"k_neighbours": [3]}) is None
    assert "ZREG_CLEAR_CHECKPOINTS=1" in caplog.text
    assert not (out / run_all.SEARCH_SPACE_FINGERPRINT).exists()


def test_run_all_bayesian_off_grid_merged_seed_is_projected(tmp_path, caplog) -> None:
    """CR-01: an averaged merge no longer crashes BayesianSearch; the seed is snapped."""
    run_all = _import_run_all()
    out = tmp_path / "hpo"
    cfg_path = _write_yaml(
        tmp_path / "cfg.yaml",
        _cfg_dict(out, strategy="bayesian", search_space={"k_neighbours": [3, 5, 10]}),
    )
    seed = {**_SEED, "k_neighbours": 6}  # e.g. merge_two(5, 7) -> 6, not a choice
    with caplog.at_level(logging.WARNING, logger="run_all"):
        run_all.run_optimize_then_eval("t", cfg_path, force=True, dry_run=False, warmstart_params=seed)
    history = _read_json(out / "search_history.json")
    assert history[0]["params"]["k_neighbours"] == 5
    assert all(h["params"]["k_neighbours"] in (3, 5, 10) for h in history)
    assert "k_neighbours: 6 -> 5" in caplog.text


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


# --- Task 3: Propulate pre-evaluates every tier's incoming seeds ------------
#
# PropulateSearch needs the uninstalled propulate library and real MPI; the
# stand-ins below replace only that dependency (and, for two ranks, the MPI
# transport). The optimizer code under test runs unmodified.

_JOIN_TIMEOUT = 300.0


class _EvaluatingPropulateSearch:
    """Single-process stand-in: evaluates every combination, returns (params, score).

    Records the ``warm_start`` it was given, so tests can check that the
    optimizer passes ``None`` after pre-evaluating the seeds (Phase 63 cycle-2).
    """

    received: list = []

    def search(self, search_space, objective_fn, n_trials, output_dir, warm_start=None):
        import itertools

        type(self).received.append(warm_start)
        keys = list(search_space)
        out = []
        for combo in itertools.product(*(search_space[k] for k in keys)):
            params = dict(zip(keys, combo))
            score = objective_fn(params)
            if score != float("-inf"):
                out.append((dict(params), score))
        return out


class _EmptyPropulateSearch:
    """Stand-in whose search evaluates nothing (e.g. a used-up resume budget)."""

    received: list = []

    def search(self, search_space, objective_fn, n_trials, output_dir, warm_start=None):
        type(self).received.append(warm_start)
        return []


def _not_seeded_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "not seeded" in r.getMessage()
    ]


def test_propulate_seed_evaluated_first_single_process(tmp_path, monkeypatch, caplog) -> None:
    """The seed is the first recorded trial, Propulate gets no seed, one WARNING."""
    monkeypatch.setattr(_EvaluatingPropulateSearch, "received", [])
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _EvaluatingPropulateSearch)
    cfg = _cfg(tmp_path / "hpo", strategy="propulate")
    with caplog.at_level(logging.WARNING):
        result = HyperparamOptimizer(cfg, warm_start=[dict(_SEED)]).run()
    assert _project(result.history[0].params, _SEED) == _SEED
    assert result.history[0].tier == "sanity"
    assert len(_not_seeded_warnings(caplog)) == 1
    assert _EvaluatingPropulateSearch.received == [None]


def test_propulate_without_warm_start_no_preevaluation(tmp_path, monkeypatch, caplog) -> None:
    """No warm start: nothing is pre-evaluated and no WARNING is logged."""
    monkeypatch.setattr(_EvaluatingPropulateSearch, "received", [])
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _EvaluatingPropulateSearch)
    cfg = _cfg(tmp_path / "hpo", strategy="propulate")
    with caplog.at_level(logging.WARNING):
        result = HyperparamOptimizer(cfg).run()
    assert [t.params for t in result.history] == [{"k_neighbours": k} for k in (3, 4, 5)]
    assert _not_seeded_warnings(caplog) == []
    assert _EvaluatingPropulateSearch.received == [None]


def test_propulate_inter_tier_seed_evaluated_on_dev_tier(tmp_path, monkeypatch, caplog) -> None:
    """Every Propulate tier pre-evaluates its incoming seeds on its own dataset.

    Sanity evaluates the external seed; dev evaluates the top pruned sanity
    candidates before Propulate's own individuals. One WARNING per tier.
    """
    monkeypatch.setattr(_EvaluatingPropulateSearch, "received", [])
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _EvaluatingPropulateSearch)
    cfg = _cfg(tmp_path / "hpo", strategy="propulate", tier="dev")
    with caplog.at_level(logging.WARNING):
        result = HyperparamOptimizer(cfg, warm_start=[dict(_SEED)]).run()
    sanity = [t for t in result.history if t.tier == "sanity"]
    dev = [t for t in result.history if t.tier == "dev"]
    assert sanity and _project(sanity[0].params, _SEED) == _SEED
    top = HyperparamOptimizer.prune_candidates(sanity, keep_top_k=3)
    assert [t.params for t in dev[: len(top)]] == top
    assert len(_not_seeded_warnings(caplog)) == 2
    assert _EvaluatingPropulateSearch.received == [None, None]


def test_propulate_inter_tier_only_seed_single_candidate(tmp_path, monkeypatch, caplog) -> None:
    """With an empty Propulate search the pruned seed is the seed itself, on both tiers."""
    monkeypatch.setattr(_EmptyPropulateSearch, "received", [])
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _EmptyPropulateSearch)
    cfg = _cfg(tmp_path / "hpo", strategy="propulate", tier="dev")
    with caplog.at_level(logging.WARNING):
        result = HyperparamOptimizer(cfg, warm_start=[dict(_SEED)]).run()
    assert [(t.tier, t.params) for t in result.history] == [("sanity", _SEED), ("dev", _SEED)]
    assert len(_not_seeded_warnings(caplog)) == 2


class _GlobalPopulationPropulateSearch:
    """Stand-in for rank 0 of a multi-rank Propulate run.

    Rank 0 evaluates only k_neighbours=3 itself; the returned population also
    holds k_neighbours=4, evaluated "on another rank" (never seen by this
    rank's objective) with the best score.
    """

    def search(self, search_space, objective_fn, n_trials, output_dir, warm_start=None):
        own = {"k_neighbours": 3}
        return [(own, objective_fn(dict(own))), ({"k_neighbours": 4}, 1e6)]


def test_propulate_next_tier_seeds_use_global_population(tmp_path, monkeypatch) -> None:
    """63-REVIEW WR-01: rank 0 prunes from Propulate's global population, not only its own trials."""
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _GlobalPopulationPropulateSearch)
    cfg = _cfg(tmp_path / "hpo", strategy="propulate", tier="dev")
    result = HyperparamOptimizer(cfg).run()
    real_dev = [t for t in result.history if t.tier == "dev" and not t.flags]
    # The other rank's best individual is the first dev-tier seed rank 0 evaluates.
    assert [t.params for t in real_dev[:2]] == [{"k_neighbours": 4}, {"k_neighbours": 3}]


def _prune_stub(returned):
    """Stand-in ``self`` for ``_prune_propulate_global`` (only reads ``_propulate_returned``)."""
    from types import SimpleNamespace

    return SimpleNamespace(_propulate_returned=list(returned))


def _local_trial(params: dict, score: float, tier: str):
    from types import SimpleNamespace

    return SimpleNamespace(params=dict(params), score=score, tier=tier)


def test_propulate_prune_ignores_individuals_restored_from_previous_tier() -> None:
    """63-REVIEW iter-2 WR-01: restored sanity individuals keep sanity scores and must not compete in dev."""
    stub = _prune_stub([
        # sanity tier: Propulate's population after the sanity search
        ("sanity", {"k": 1}, 0.95),
        ("sanity", {"k": 2}, 0.90),
        # dev tier: the same two individuals, restored from the sanity checkpoint
        ("dev", {"k": 1}, 0.95),
        ("dev", {"k": 2}, 0.90),
        # dev tier: individuals actually evaluated on the dev dataset
        ("dev", {"k": 3}, 0.60),
        ("dev", {"k": 4}, 0.55),
        ("dev", {"k": 5}, 0.50),
    ])
    got = HyperparamOptimizer._prune_propulate_global(stub, [], "dev", keep_top_k=3)
    assert got == [{"k": 3}, {"k": 4}, {"k": 5}]


def test_propulate_prune_keeps_rescored_seeds_and_drops_earlier_local_trials() -> None:
    """63-REVIEW iter-2 WR-01: a seed rank 0 re-scored in dev competes; its sanity trial does not."""
    stub = _prune_stub([
        ("sanity", {"k": 1}, 0.95),
        ("dev", {"k": 1}, 0.95),  # restored, stale sanity score
        ("dev", {"k": 3}, 0.60),
    ])
    history = [
        _local_trial({"k": 1}, 0.95, "sanity"),
        _local_trial({"k": 9}, 0.99, "sanity"),  # never evaluated in dev
        _local_trial({"k": 1}, 0.40, "dev"),  # rank 0's dev re-score of the seed
    ]
    got = HyperparamOptimizer._prune_propulate_global(stub, history, "dev", keep_top_k=3)
    assert got == [{"k": 3}, {"k": 1}]


def test_propulate_prune_first_tier_keeps_all_returned_pairs() -> None:
    """With no earlier tier, every returned pair of the tier is a candidate."""
    stub = _prune_stub([("sanity", {"k": 1}, 0.2), ("sanity", {"k": 2}, 0.8)])
    got = HyperparamOptimizer._prune_propulate_global(stub, [], "sanity", keep_top_k=3)
    assert got == [{"k": 2}, {"k": 1}]


# Two thread-simulated ranks (copied from tests/test_optimizer_scoring_contract.py,
# owned by Phase 62-05; only the MPI transport is simulated).


class _ThreadHub:
    def __init__(self, size: int) -> None:
        import threading

        self.size = size
        self.barrier = threading.Barrier(size, timeout=_JOIN_TIMEOUT)
        self.slots: list = [None] * size
        self.bvalue = None
        self.local = threading.local()
        self.population: list = []
        self.lock = threading.Lock()


class _ThreadComm:
    """In-process stand-in for the mpi4py COMM_WORLD transport (lowercase API)."""

    def __init__(self, hub: _ThreadHub, rank: int) -> None:
        self.hub = hub
        self.rank = rank
        self.calls: list[str] = []

    def Get_rank(self) -> int:  # noqa: N802 (mpi4py API)
        return self.rank

    def Get_size(self) -> int:  # noqa: N802 (mpi4py API)
        return self.hub.size

    def gather(self, obj, root=0):
        import pickle

        self.calls.append("gather")
        self.hub.slots[self.rank] = pickle.loads(pickle.dumps(obj))
        self.hub.barrier.wait()
        out = list(self.hub.slots) if self.rank == root else None
        self.hub.barrier.wait()
        return out

    def bcast(self, obj, root=0):
        import pickle

        self.calls.append("bcast")
        if self.rank == root:
            self.hub.bvalue = pickle.dumps(obj)
        self.hub.barrier.wait()
        out = pickle.loads(self.hub.bvalue)
        self.hub.barrier.wait()
        return out


def _run_in_rank_threads(hub: _ThreadHub, fns: list) -> tuple[list, list]:
    import threading

    results: list = [None] * len(fns)
    errors: list = [None] * len(fns)

    def runner(r: int) -> None:
        hub.local.rank = r
        try:
            results[r] = fns[r]()
        except BaseException as exc:  # noqa: BLE001 — reported per rank
            errors[r] = exc

    threads = [threading.Thread(target=runner, args=(r,), daemon=True) for r in range(len(fns))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(_JOIN_TIMEOUT)
    assert not any(t.is_alive() for t in threads), "a rank hung: divergent collectives"
    return results, errors


class _ContractPropulateSearch:
    """Return contract of PropulateSearch.search (copied from the 62-05 harness).

    Every rank evaluates the objective; the final barrier makes every rank's
    individuals visible on rank 0; non-zero ranks return []; rank 0 returns
    ``(params, -loss)`` for every finite individual.
    """

    hub: _ThreadHub | None = None
    received: list = []

    def search(self, search_space, objective_fn, n_trials, output_dir, warm_start=None):
        import itertools

        hub = type(self).hub
        rank = hub.local.rank
        with hub.lock:
            type(self).received.append((rank, warm_start))
        keys = list(search_space)
        for combo in itertools.product(*(search_space[k] for k in keys)):
            params = dict(zip(keys, combo))
            loss = -objective_fn(params)
            with hub.lock:
                hub.population.append((params, loss))
        hub.barrier.wait()  # final intra-island receive + comm.Barrier()
        if rank != 0:
            return []
        return [(dict(p), -loss) for p, loss in hub.population if loss != float("inf")]


def test_propulate_seed_evaluated_once_two_ranks(tmp_path, monkeypatch, caplog) -> None:
    """Rank 0 alone evaluates the seed; it appears once in the merged history."""
    hub = _ThreadHub(2)
    monkeypatch.setattr(_ContractPropulateSearch, "hub", hub)
    monkeypatch.setattr(_ContractPropulateSearch, "received", [])
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _ContractPropulateSearch)
    opts, comms, outs = [], [], []
    for r, ks in enumerate(([3], [4])):
        out = tmp_path / f"rank{r}"
        opt = HyperparamOptimizer(
            _cfg(out, strategy="propulate", search_space={"k_neighbours": ks}),
            warm_start=[dict(_SEED)],
        )
        opt._comm = _ThreadComm(hub, r)
        opts.append(opt)
        comms.append(opt._comm)
        outs.append(out)
    with caplog.at_level(logging.WARNING):
        results, errors = _run_in_rank_threads(hub, [opts[0].run, opts[1].run])
    assert errors == [None, None]
    merged = results[0].history
    seed_trials = [t for t in merged if _project(t.params, _SEED) == _SEED]
    assert len(seed_trials) == 1
    assert seed_trials[0].tier == "sanity"
    assert merged[0].params == _SEED
    persisted = _read_json(outs[0] / "search_history.json")
    assert sum(1 for t in persisted if t["params"] == _SEED) == 1
    # No new collective: both ranks issued the same collectives in the same order.
    assert comms[0].calls == comms[1].calls
    assert sorted(_ContractPropulateSearch.received, key=lambda x: x[0]) == [(0, None), (1, None)]
    assert len(_not_seeded_warnings(caplog)) == 1


def _fake_propulate_modules(monkeypatch) -> None:
    """Install a minimal stand-in for the uninstalled propulate package."""
    import types

    class _Propulator:
        def __init__(self, **kwargs) -> None:
            self.population: list = []

        def propulate(self, logging_interval=1) -> None:
            return None

    pkg = types.ModuleType("propulate")
    pkg.Propulator = _Propulator
    utils = types.ModuleType("propulate.utils")
    utils.get_default_propagator = lambda **kwargs: object()
    utils.set_logger_config = lambda **kwargs: None
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "propulate", pkg)
    monkeypatch.setitem(sys.modules, "propulate.utils", utils)


@pytest.mark.parametrize("seeds, n_warnings", [([{"k_neighbours": 5}], 1), (None, 0)])
def test_propulate_search_warns_when_given_seeds(tmp_path, monkeypatch, caplog, seeds, n_warnings) -> None:
    """PropulateSearch never drops seeds silently: a direct caller gets a WARNING."""
    _fake_propulate_modules(monkeypatch)
    from eval.search_strategies import PropulateSearch

    with caplog.at_level(logging.WARNING, logger="eval.search_strategies"):
        out = PropulateSearch().search(
            {"k_neighbours": [3, 4]}, lambda p: 0.5, n_trials=2, output_dir=str(tmp_path), warm_start=seeds
        )
    assert out == []
    assert len(_not_seeded_warnings(caplog)) == n_warnings


def test_propulate_loss_hint_for_individual_missing_a_new_key(tmp_path, monkeypatch) -> None:
    """63-REVIEW IN-07: a checkpoint individual lacking a since-added key gets the recovery hint."""
    _fake_propulate_modules(monkeypatch)
    import propulate

    class _StaleIndividualPropulator:
        def __init__(self, *, loss_fn, **kwargs) -> None:
            self.loss_fn = loss_fn
            self.population: list = []

        def propulate(self, logging_interval=1) -> None:
            self.loss_fn({"k_neighbours": "3"})  # written before "window_size" was searched

    monkeypatch.setattr(propulate, "Propulator", _StaleIndividualPropulator)
    from eval.search_strategies import PropulateSearch

    with pytest.raises(ValueError, match="ZREG_CLEAR_CHECKPOINTS=1") as excinfo:
        PropulateSearch().search(
            {"k_neighbours": [3, 4], "window_size": [5]}, lambda p: 0.5, n_trials=2, output_dir=str(tmp_path)
        )
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_main_reports_stale_checkpoints_as_one_line_error(tmp_path, monkeypatch, capsys) -> None:
    """63-REVIEW IN-08: main() prints the stale-checkpoint error and returns 1 (no traceback)."""
    run_all = _import_run_all()

    def _stale(*args, **kwargs):
        raise run_all.StaleCheckpointError("[t] checkpoints from another search space; ZREG_CLEAR_CHECKPOINTS=1")

    monkeypatch.setattr(run_all, "_build_phase_lists", lambda configs_dir: {})
    monkeypatch.setattr(run_all, "run_phase", _stale)
    monkeypatch.chdir(tmp_path)
    assert run_all.main(["--phase", "selfcal", "--configs-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: [t] checkpoints from another search space")
    assert issubclass(run_all.StaleCheckpointError, RuntimeError)
