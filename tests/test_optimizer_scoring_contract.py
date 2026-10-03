"""Contract tests for HyperparamOptimizer scoring (Phase 59 NUM-02, NUM-03, NUM-04).

These are *contract* tests: they drive the real ``_score_subsample_pair_multiseed``,
the real ``_objective``, the real ``MetricsEngine.compute_stage_metrics`` and the
real stages/``DataFactory``.  They must never patch the units under test (D-06,
review pattern 3): a mocked scorer is exactly how the multi-seed path came to
score raw tensors against the *unaligned* source view (chamfer 0.0 for a 0 deg
and a 90 deg pair alike) without any test noticing.

The only stub allowed here is data loading (``DataFactory.load_target``) in the
paired-mode tests, because the paired path would otherwise need real files.

Allowed dependency substitutions (Phase 62, LT-02 / 59-REVIEW IN-09b/c):
  - ``_FlaggingLabelTransferStage``: a subclass of the real
    ``LabelTransferStage`` that runs ``super().run`` unchanged and appends one
    known flag to the returned ``LabelResult.flags``. Used only where no
    cpd_weighted posterior exists (multiseed knn_voting flag-prefix test).
  - ``_injected_alignment_stage``: ``_injected_alignment_stage`` is an allowed
    dependency substitution: it runs the real AlignmentStage and only fixes the
    posterior values handed to the real LabelTransferStage; it does not mock the
    unit under test. It zeroes the first ``n_zero`` receiver columns of every
    frame's CPD posterior so the fallback / fraction-bound policy of
    ``repair_pmat_rows`` is reached deterministically (a displaced real point
    cannot produce a zero-mass row: sigma2 grows with the displacement).

Covers:
  - NUM-02: multi-seed HPO scores the aligned per-frame dict against the target
    dict, so a misaligned pair scores worse, both in the last-seed metrics and
    in the averaged objective the search strategies rank on.
  - NUM-03: a list-valued seed runs the sanity and dev tiers on ``seed[0]``;
    ``_tier_dataset`` never returns ``None`` on the subsample_pair path.
  - NUM-04 (HPO half): ``_objective`` honours ``label_source`` through the
    shared ``resolve_label_transfer_pair`` helper and zero-fills F1 when it is
    unavailable.
  - NUM-05 (optimizer half): a raising trial scores ``-inf``, is logged with its
    traceback and recorded as a failure; counters reset per ``run()``; failures
    and successful histories of every MPI rank are merged on rank 0 (finite
    scores only); a run raises only when no rank produced a successful trial.

NUM-05 multi-rank tests substitute only the MPI transport (``_ThreadComm``) and,
for the Propulate branch, the uninstalled propulate library
(``_ContractPropulateSearch``); the optimizer code under test is unmodified.
"""

import itertools
import json
import logging
import math
import pickle
import sys
import threading
import types
from pathlib import Path

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.core.dataset import zRegPointCloud

import torch

import optuna  # noqa: F401  (import-order contract: torch -> optuna -> eval)

from eval.config import EvalConfig
from eval.runners import optimizer as optimizer_module
from eval.runners.optimizer import HyperparamOptimizer
from eval.types import StageMetrics, Trial

_OPT_LOGGER = "eval.runners.optimizer"


def _cfg(
    tmp_path, *, rot=0.0, seed=None, tier="full", search_space=None, search_strategy="grid",
    **extra,
) -> EvalConfig:
    """Synthetic subsample_pair config used by the multi-seed / list-seed tests."""
    if seed is None:
        seed = [1, 2]
    if search_space is None:
        search_space = {"window_size": [3], "k_neighbours": [3]}
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="synthetic",
        run_alignment=True,
        run_label_transfer=True,
        transform_spec={
            "type": "subsample_pair",
            "synthesize": True,
            "seed": seed,
            "n_classes": 3,
            "n_points": 80,
            "source_fraction": 0.8,
            "target_fraction": 0.8,
            "rotation_deg": rot,
            "rotation_axis": [0.0, 0.0, 1.0],
            "scale_factor": 1.0,
        },
        tier=tier,
        n_trials=1,
        search_strategy=search_strategy,
        search_space=search_space,
        **extra,
    )


def _trial_failed_records(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Trial failed" in r.getMessage()]


# ---------------------------------------------------------------------------
# NUM-02: multi-seed scoring measures alignment quality
# ---------------------------------------------------------------------------


def test_multiseed_scores_alignment_quality(tmp_path) -> None:
    """A 90 deg misaligned pair scores strictly worse than a 0 deg pair.

    With the default params (``cpd_penalty=None`` => temporal-only, no spatial
    registration) the 90 deg pair stays misaligned in the scored dict.
    """
    metrics = {}
    returned = {}
    for rot in (0.0, 90.0):
        opt = HyperparamOptimizer(_cfg(tmp_path / f"rot{int(rot)}", rot=rot))
        hist: list = []
        returned[rot] = opt._score_subsample_pair_multiseed(
            {}, dict(opt._default_params), "full", [1, 2], hist
        )
        # The Trial records exactly the float the search strategy receives.
        assert len(hist) == 1
        assert hist[-1].score == returned[rot]
        metrics[rot] = hist[-1].metrics

    m0, m90 = metrics[0.0], metrics[90.0]
    returned_0, returned_90 = returned[0.0], returned[90.0]

    # (a) last-seed metrics: aligned dicts were scored, not raw tensors.
    assert math.isfinite(m0.chamfer_distance)
    assert m0.chamfer_distance > 0.0  # 6c1c37f: exactly 0.0; after 59-04 only: inf
    assert m90.chamfer_distance > m0.chamfer_distance
    assert m90.hausdorff_distance > m0.hausdorff_distance
    assert m90.normalized["chamfer"] < m0.normalized["chamfer"] < 1.0
    assert [f for f in m0.coverage_flags if f.startswith("frame coverage:")] == []

    # (b) the averaged objective the search actually ranks on.
    assert math.isfinite(returned_0)
    assert math.isfinite(returned_90)
    assert returned_90 < returned_0


# ---------------------------------------------------------------------------
# NUM-03: list-valued seeds run every tier
# ---------------------------------------------------------------------------


def test_list_seed_runs_every_tier_on_seed0(tmp_path, caplog) -> None:
    """A list seed records trials in sanity, dev and full with no swallowed failure."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=[1, 2, 3], tier="full"))
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        result = opt.run()
    assert {t.tier for t in result.history} == {"sanity", "dev", "full"}
    assert _trial_failed_records(caplog) == []
    assert opt._tier_dataset("dev") is not None
    # The pre-populated dev/full dataset was generated from seed[0].
    reference = HyperparamOptimizer(_cfg(tmp_path / "ref", seed=1, tier="full"))
    reference._factory.generate_subsample_pair(
        None, {**reference.config.transform_spec, "seed": 1}
    )
    dev_view = opt._tier_dataset("dev")
    ref_view = reference._tier_dataset("dev")
    assert sorted(dev_view) == sorted(ref_view)
    for k in dev_view:
        assert torch.equal(dev_view[k]["pos"], ref_view[k]["pos"])


def test_list_seed_sanity_tier_uses_seed0(tmp_path, caplog) -> None:
    """A list seed with tier='sanity' records the sanity trials (no ValueError swallowed)."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=[5, 6], tier="sanity"))
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        result = opt.run()
    # grid over a 1x1 search space -> exactly one sanity trial
    assert len(result.history) == 1
    assert {t.tier for t in result.history} == {"sanity"}
    assert _trial_failed_records(caplog) == []


def test_list_seed_empty_list_raises_in_run(tmp_path) -> None:
    """An empty seed list is rejected loudly by run()'s pre-populate step."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=[], tier="dev"))
    with pytest.raises(ValueError, match="must be non-empty"):
        opt.run()


@pytest.mark.parametrize("tier", ["dev", "full"])
def test_tier_dataset_raises_when_not_prepopulated(tmp_path, tier) -> None:
    """_tier_dataset never returns None on the subsample_pair path."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=7))
    with pytest.raises(RuntimeError, match="pre-populate"):
        opt._tier_dataset(tier)


# ---------------------------------------------------------------------------
# NUM-04 (HPO half): _objective honours label_source
# ---------------------------------------------------------------------------


def _frames(n_points: int, labels: bool, offset: float) -> dict:
    gen = torch.Generator().manual_seed(n_points)
    out = {}
    for f in range(3):
        pos = torch.rand(n_points, 3, generator=gen) + offset
        label = (
            torch.randint(10, 13, (n_points,), generator=gen, dtype=torch.long)
            if labels
            else None
        )
        out[f] = zRegPointCloud(pos=pos, label=label, id=None)
    return out


def test_objective_paired_target_direction_uses_provider_labels(
    tmp_path, monkeypatch, caplog
) -> None:
    """label_source='target': labels flow target -> source, F1 is zero-filled."""
    cfg = EvalConfig(
        data_path=str(tmp_path / "kobitski.mat"),
        target_data_path=str(tmp_path / "shah.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        label_source="target",
        run_alignment=False,
        run_label_transfer=True,
        label_transfer_method="knn_voting",
        search_strategy="grid",
        search_space={"k_neighbours": [3]},
    )
    source = _frames(25, labels=False, offset=0.0)  # unlabeled Kobitski-like
    target = _frames(40, labels=True, offset=0.0)  # labelled Shah-like
    opt = HyperparamOptimizer(cfg)
    monkeypatch.setattr(opt._factory, "load_target", lambda: target)

    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"k_neighbours": 3}, source, "dev", hist)
        # second trial: the F1-unavailable reason is logged once per instance
        opt._objective({"k_neighbours": 3}, source, "dev", hist)

    assert _trial_failed_records(caplog) == []
    assert len(hist) == 2
    assert math.isfinite(score)
    assert hist[0].score == score
    assert hist[0].metrics.f1_score == 0.0
    assert hist[0].metrics.normalized["f1"] == 0.0
    f1_logs = [r for r in caplog.records if "f1 unavailable" in r.getMessage()]
    assert len(f1_logs) == 1


def test_objective_paired_target_without_label_transfer_zero_fills_f1(
    tmp_path, monkeypatch, caplog
) -> None:
    """F1 unavailable and label transfer off: no GT read, unequal counts do not raise."""
    cfg = EvalConfig(
        data_path=str(tmp_path / "kobitski.mat"),
        target_data_path=str(tmp_path / "shah.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        label_source="target",
        run_alignment=True,
        run_label_transfer=False,
        search_strategy="grid",
        search_space={"window_size": [3]},
    )
    source = _frames(25, labels=False, offset=0.0)
    target = _frames(40, labels=True, offset=0.0)
    opt = HyperparamOptimizer(cfg)
    monkeypatch.setattr(opt._factory, "load_target", lambda: target)

    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"window_size": 3}, source, "dev", hist)

    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(score)
    assert hist[0].metrics.f1_score == 0.0


# ---------------------------------------------------------------------------
# LT-04 / U5-3 WR-01 (Phase 62): kNN consistency is scored on the receiver
# frame's positions together with its UNTRUNCATED transferred labels; the
# WR-01 truncation applies to the F1 pair only.
# ---------------------------------------------------------------------------


def _paired_source_cfg(tmp_path, *, run_alignment, run_label_transfer, search_space) -> EvalConfig:
    """Paired config with the default label_source='source' (F1 live)."""
    return EvalConfig(
        data_path=str(tmp_path / "source.mat"),
        target_data_path=str(tmp_path / "target.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        run_alignment=run_alignment,
        run_label_transfer=run_label_transfer,
        label_transfer_method="knn_voting",
        search_strategy="grid",
        search_space=search_space,
    )


def test_objective_knn_uses_untruncated_receiver_labels(tmp_path, monkeypatch, caplog) -> None:
    """25 labelled source points -> 40 target points: kNN scores all 40 receiver points."""
    from eval.stages import LabelTransferStage
    from zreg.evaluation.label_transfer import knn_consistency

    cfg = _paired_source_cfg(
        tmp_path, run_alignment=False, run_label_transfer=True,
        search_space={"k_neighbours": [3]},
    )
    source = _frames(25, labels=True, offset=0.0)
    target = _frames(40, labels=True, offset=0.0)
    opt = HyperparamOptimizer(cfg)
    monkeypatch.setattr(opt._factory, "load_target", lambda: target)

    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"k_neighbours": 3}, source, "dev", hist)

    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(score)

    # Independent reference: the real stage on the same inputs, last receiver frame.
    reference = LabelTransferStage(cfg).run(source, target, {**opt._default_params, "k_neighbours": 3})
    last = sorted(reference.transferred_labels)[-1]
    expected_labels = reference.transferred_labels[last]
    assert expected_labels.shape[0] == 40
    expected = knn_consistency(target[last]["pos"], expected_labels, 3)
    assert hist[0].metrics.knn_consistency == pytest.approx(expected)


def test_objective_knn_placeholder_matches_receiver_without_label_transfer(
    tmp_path, monkeypatch, caplog
) -> None:
    """Temporal-only alignment, label transfer off: the kNN placeholder has the receiver's length."""
    cfg = _paired_source_cfg(
        tmp_path, run_alignment=True, run_label_transfer=False,
        search_space={"window_size": [3]},
    )
    source = _frames(25, labels=True, offset=0.0)
    target = _frames(40, labels=True, offset=0.0)
    opt = HyperparamOptimizer(cfg)  # a configuration the optimizer accepts
    monkeypatch.setattr(opt._factory, "load_target", lambda: target)

    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"window_size": 3, "cpd_penalty": None}, source, "dev", hist)

    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(score)
    assert math.isfinite(hist[0].metrics.knn_consistency)


def test_knn_inputs_placeholder_on_receiver_device(tmp_path) -> None:
    """Without a label result the placeholder labels match the receiver frame's length and device."""
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_OK_K]))
    points = torch.rand(40, 3)
    knn_points, labels = opt._knn_inputs(None, None, points)
    assert knn_points is points
    assert labels.shape == (40,)
    assert labels.dtype == torch.long
    assert labels.device == points.device
    assert torch.count_nonzero(labels) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_knn_inputs_placeholder_on_cuda(tmp_path) -> None:
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_OK_K]))
    points = torch.rand(40, 3, device="cuda")
    _, labels = opt._knn_inputs(None, None, points)
    assert labels.device == points.device


def test_multiseed_knn_finite(tmp_path, caplog) -> None:
    """Regression guard: the multiseed path still records a trial with finite kNN consistency."""
    opt = HyperparamOptimizer(_cfg(tmp_path))
    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        opt._score_subsample_pair_multiseed({}, dict(opt._default_params), "full", [1, 2], hist)
    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(hist[0].metrics.knn_consistency)


def _paired_target_cfg(tmp_path, f1_weight) -> EvalConfig:
    return EvalConfig(
        data_path=str(tmp_path / "s.mat"),
        target_data_path=str(tmp_path / "t.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        label_source="target",
        tier="sanity",
        search_strategy="grid",
        search_space={"k_neighbours": [3]},
        metric_weights={
            "chamfer": 0.30,
            "hausdorff": 0.15,
            "path_smoothness": 0.10,
            "temporal_stability": 0.10,
            "f1": f1_weight,
            "knn_consistency": 0.10,
        },
    )


@pytest.mark.parametrize("f1_weight,expect_warning", [(0.25, True), (0.0, False)])
def test_run_warns_on_dead_f1_weight(tmp_path, monkeypatch, caplog, f1_weight, expect_warning) -> None:
    """WR-07: run() warns when F1 is unavailable but still carries a positive weight."""
    opt = HyperparamOptimizer(_paired_target_cfg(tmp_path, f1_weight))
    monkeypatch.setattr(opt, "_run_tiers", lambda _out: [])
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        opt.run()
    warned = any("metric_weights['f1']" in r.getMessage() for r in caplog.records)
    assert warned is expect_warning


# ---------------------------------------------------------------------------
# NUM-05 (optimizer half): failed trials are recorded, worst-scored and merged
# across MPI ranks.
#
# A real failing trial needs no mock: k_neighbours larger than the frame point
# count makes LabelTransferStage raise ValueError
# (eval/stages/label_transfer.py, "exceeds source frame").
# ---------------------------------------------------------------------------

_OK_K = 3
_FAIL_K = 10000
_JOIN_TIMEOUT = 300.0


def _num05_cfg(tmp_path, k_values, *, strategy="grid") -> EvalConfig:
    """Cheap sanity-tier config whose trials succeed (k=3) or raise (k=10000)."""
    return _cfg(
        tmp_path,
        seed=1,
        tier="sanity",
        search_space={"k_neighbours": list(k_values)},
        search_strategy=strategy,
    )


def _zero_metrics() -> StageMetrics:
    return StageMetrics(
        chamfer_distance=0.0,
        hausdorff_distance=0.0,
        path_smoothness=0.0,
        temporal_stability=0.0,
        f1_score=0.0,
        knn_consistency=0.0,
    )


def _trial(k: int, score: float) -> Trial:
    return Trial(params={"k_neighbours": k}, score=score, metrics=_zero_metrics(), tier="sanity")


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def test_failed_trial_is_recorded_and_worst(tmp_path, caplog) -> None:
    """A raising trial returns -inf, appends no Trial and is recorded with its traceback."""
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_FAIL_K]))
    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"k_neighbours": _FAIL_K}, opt._tier_dataset("sanity"), "sanity", hist)

    assert score == float("-inf")  # 6c1c37f: 0.0
    assert hist == []
    assert len(opt._failed_trials) == 1
    record = opt._failed_trials[0]
    assert {"params", "tier", "error", "error_type"} <= set(record)
    assert record["tier"] == "sanity"
    assert record["params"] == {"k_neighbours": _FAIL_K}
    assert "k_neighbours" in record["error"]
    assert record["error_type"] == "ValueError"
    failed_logs = [r for r in caplog.records if "Trial failed" in r.getMessage()]
    assert failed_logs, "the failure must be logged"
    assert all(r.exc_info for r in failed_logs), "the traceback must be logged (exc_info)"


def _no_scorable_frame(*_args, **_kwargs):
    """Frame average of an alignment whose every shared frame is degenerate."""
    from eval.metrics import FrameAverage

    inf = float("inf")
    return FrameAverage(inf, inf, 0, ["frame coverage: skipped degenerate frame 0: non-finite result"])


def test_trial_without_scorable_frame_is_recorded_as_failed(tmp_path, monkeypatch, caplog) -> None:
    """WR-01: zero scored frames is a failed trial (-inf), not a finite rankable score.

    Only the frame-average primitive is substituted to simulate a degenerate
    alignment (e.g. NaN positions after a CPD divergence); ``_objective`` and
    ``compute_stage_metrics`` run unmodified.
    """
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_OK_K]))
    monkeypatch.setattr(opt._engine, "_frame_averaged_chamfer_hausdorff", _no_scorable_frame)
    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective({"k_neighbours": _OK_K}, opt._tier_dataset("sanity"), "sanity", hist)

    assert score == float("-inf")
    assert hist == []
    assert opt._n_succeeded == 0
    assert len(opt._failed_trials) == 1
    assert "no scorable frame" in opt._failed_trials[0]["error"]
    assert _trial_failed_records(caplog)


def test_multiseed_trial_without_scorable_frame_is_recorded_as_failed(tmp_path, monkeypatch) -> None:
    """WR-01: the multiseed path applies the same no-scorable-frame rule."""
    opt = HyperparamOptimizer(_cfg(tmp_path, seed=[1, 2], tier="full"))
    monkeypatch.setattr(opt._engine, "_frame_averaged_chamfer_hausdorff", _no_scorable_frame)
    hist: list = []
    score = opt._objective({"window_size": 3, "k_neighbours": 3}, {}, "full", hist)

    assert score == float("-inf")
    assert hist == []
    assert len(opt._failed_trials) == 1
    assert "no scorable frame" in opt._failed_trials[0]["error"]


def test_successful_trial_counted(tmp_path) -> None:
    """A successful trial increments the success counter and records no failure."""
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_OK_K]))
    hist: list = []
    score = opt._objective({"k_neighbours": _OK_K}, opt._tier_dataset("sanity"), "sanity", hist)
    assert math.isfinite(score)
    assert len(hist) == 1
    assert opt._n_succeeded == 1
    assert opt._failed_trials == []


def test_run_resets_counters_between_calls(tmp_path) -> None:
    """A second run() on the same instance reports only its own outcomes (cycle-2 LOW)."""
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_OK_K, _FAIL_K]))
    first = opt.run()
    n_ok_first = opt._n_succeeded
    second = opt.run()
    assert len(first.failed_trials) >= 1
    assert len(second.failed_trials) == len(first.failed_trials)
    assert opt._n_succeeded == n_ok_first


def test_all_failed_run_raises_and_writes_failed_trials_json(tmp_path) -> None:
    """Every trial failing raises loudly after failed_trials.json is written (A5)."""
    cfg = _num05_cfg(tmp_path, [_FAIL_K])
    opt = HyperparamOptimizer(cfg)
    with pytest.raises(RuntimeError, match=r"All .*failed"):
        opt.run()  # 6c1c37f: returned SearchResult(best_params={}) silently
    failed_path = Path(cfg.output_dir).resolve() / "failed_trials.json"
    assert failed_path.exists()
    records = _read_json(failed_path)
    assert isinstance(records, list)
    assert len(records) == 1  # grid over one value -> one attempted trial
    assert all(r["rank"] == 0 for r in records)
    assert all(r["error_type"] == "ValueError" for r in records)


def test_search_result_carries_failed_trials(tmp_path) -> None:
    """Mixed outcomes: the finite trial wins, failures are kept and persisted."""
    cfg = _num05_cfg(tmp_path, [_OK_K, _FAIL_K])
    result = HyperparamOptimizer(cfg).run()
    out = Path(cfg.output_dir).resolve()
    assert len(result.failed_trials) >= 1
    assert result.best_params["k_neighbours"] == _OK_K
    assert math.isfinite(result.best_score)
    assert len(_read_json(out / "failed_trials.json")) == len(result.failed_trials)
    history = _read_json(out / "search_history.json")
    assert isinstance(history, list) and history
    assert all(math.isfinite(t["score"]) for t in history)
    assert _read_json(out / "best_params.json") == dict(result.best_params)


@pytest.mark.parametrize("strategy", ["grid", "random", "sobol", "bayesian"])
def test_mixed_outcomes_every_strategy(tmp_path, strategy) -> None:
    """Every search strategy survives -inf trials and never selects one as best.

    warm_start lists both values: every strategy's search() evaluates warm-start
    params first (Optuna via enqueue_trial), so the failing value is evaluated
    deterministically even though random/sobol sample only SANITY_N_TRIALS points.
    """
    cfg = _num05_cfg(tmp_path, [_OK_K, _FAIL_K], strategy=strategy)
    opt = HyperparamOptimizer(cfg, warm_start=[{"k_neighbours": _OK_K}, {"k_neighbours": _FAIL_K}])
    result = opt.run()
    out = Path(cfg.output_dir).resolve()
    assert result.best_params["k_neighbours"] == _OK_K
    assert math.isfinite(result.best_score)
    assert result.history
    assert all(math.isfinite(t.score) for t in result.history)
    assert all(math.isfinite(t["score"]) for t in _read_json(out / "search_history.json"))
    assert len(_read_json(out / "failed_trials.json")) >= 1


def test_merge_drops_non_finite_scores() -> None:
    """The merge keeps finite trials only; non-finite ones become failure records."""
    from eval.runners.optimizer import _merge_trial_histories

    ok = _trial(3, 0.4).model_dump()
    neg_inf = _trial(4, float("-inf")).model_dump()
    nan = _trial(5, float("nan")).model_dump()
    merged, bad = _merge_trial_histories([[ok], [neg_inf, nan]], start_rank=0)
    assert [t.score for t in merged] == [0.4]
    assert isinstance(merged[0], Trial)
    assert len(bad) == 2
    assert all(r["error_type"] == "NonFiniteScore" for r in bad)
    assert all(r["rank"] == 1 for r in bad)
    assert [r["params"]["k_neighbours"] for r in bad] == [4, 5]


# --- thread-simulated MPI ranks --------------------------------------------


class _ThreadHub:
    """Shared rendezvous state for the ``_ThreadComm`` ranks of one test."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.barrier = threading.Barrier(size, timeout=_JOIN_TIMEOUT)
        self.slots: list = [None] * size
        self.bvalue = None
        self.local = threading.local()
        self.population: list = []
        self.lock = threading.Lock()


class _ThreadComm:
    """In-process stand-in for the mpi4py COMM_WORLD transport (lowercase API).

    Only the transport is simulated: payloads round-trip through pickle exactly
    like mpi4py's lowercase collectives, and each collective's name is logged so
    tests can assert that every rank issued the same collectives in the same
    order. The optimizer code under test runs unmodified.
    """

    def __init__(self, hub: _ThreadHub, rank: int) -> None:
        self.hub = hub
        self.rank = rank
        self.calls: list[str] = []

    def Get_rank(self) -> int:  # noqa: N802 (mpi4py API)
        return self.rank

    def Get_size(self) -> int:  # noqa: N802 (mpi4py API)
        return self.hub.size

    def gather(self, obj, root=0):
        self.calls.append("gather")
        self.hub.slots[self.rank] = pickle.loads(pickle.dumps(obj))
        self.hub.barrier.wait()
        out = list(self.hub.slots) if self.rank == root else None
        self.hub.barrier.wait()
        return out

    def bcast(self, obj, root=0):
        self.calls.append("bcast")
        if self.rank == root:
            self.hub.bvalue = pickle.dumps(obj)
        self.hub.barrier.wait()
        out = pickle.loads(self.hub.bvalue)
        self.hub.barrier.wait()
        return out


def _run_in_rank_threads(hub: _ThreadHub, fns: list) -> tuple[list, list]:
    """Run ``fns[r]()`` on thread r; return (results, exceptions) per rank.

    A thread still alive after the timeout means divergent collectives (a
    deadlock under real MPI) and fails the test.
    """
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


def _rank_optimizers(tmp_path, k_per_rank: list, *, strategy="grid"):
    hub = _ThreadHub(len(k_per_rank))
    opts, cfgs = [], []
    for r, ks in enumerate(k_per_rank):
        cfg = _num05_cfg(tmp_path / f"rank{r}", ks, strategy=strategy)
        opt = HyperparamOptimizer(cfg)
        opt._comm = _ThreadComm(hub, r)
        opts.append(opt)
        cfgs.append(cfg)
    return hub, opts, cfgs


def test_reduce_trial_outcomes_merges_all_ranks(tmp_path) -> None:
    """gather + bcast merge failures, successes and histories of every rank."""
    hub, opts, _ = _rank_optimizers(tmp_path, [[_OK_K], [_OK_K]])
    opts[0]._failed_trials = [
        {"params": {"k_neighbours": 9}, "tier": "sanity", "error": "e0", "error_type": "ValueError"},
    ]
    opts[0]._n_succeeded = 1
    opts[1]._failed_trials = [
        {"params": {"k_neighbours": 8}, "tier": "sanity", "error": "e1", "error_type": "ValueError"},
        {"params": {"k_neighbours": 7}, "tier": "sanity", "error": "e2", "error_type": "ValueError"},
    ]
    opts[1]._n_succeeded = 1
    local = [[_trial(3, 0.4)], [_trial(5, 0.6)]]

    results, errors = _run_in_rank_threads(
        hub, [lambda r=r: opts[r]._reduce_trial_outcomes(local[r]) for r in range(2)]
    )
    assert errors == [None, None]
    for _merged, _failures, n_ok, n_fail, n_hist in results:
        assert (n_ok, n_fail, n_hist) == (2, 3, 2)
    merged0, failures0 = results[0][0], results[0][1]
    assert sorted(t.score for t in merged0) == [0.4, 0.6]
    assert len(failures0) == 3
    assert {r["rank"] for r in failures0} == {0, 1}
    assert results[1][0] == []
    assert opts[0]._comm.calls == ["gather", "bcast"]
    assert opts[1]._comm.calls == ["gather", "bcast"]


def test_run_threads_rank1_all_failed_does_not_raise(tmp_path) -> None:
    """Rank 1 failing everything is fine when rank 0 succeeded (global success)."""
    hub, opts, cfgs = _rank_optimizers(tmp_path, [[_OK_K], [_FAIL_K]])
    _, errors = _run_in_rank_threads(hub, [opts[0].run, opts[1].run])
    assert errors == [None, None]
    out0 = Path(cfgs[0].output_dir).resolve()
    out1 = Path(cfgs[1].output_dir).resolve()
    failed = _read_json(out0 / "failed_trials.json")
    assert any(r["rank"] == 1 for r in failed)
    assert _read_json(out0 / "best_params.json")["k_neighbours"] == _OK_K
    for name in ("best_params.json", "search_history.json", "failed_trials.json"):
        assert not (out1 / name).exists(), f"rank 1 must not write {name}"
    assert opts[0]._comm.calls == opts[1]._comm.calls == ["gather", "bcast"]


def test_run_threads_rank0_all_failed_rank1_succeeds(tmp_path) -> None:
    """Cycle-2 HIGH inverse case: rank 0 persists rank 1's successful trial."""
    hub, opts, cfgs = _rank_optimizers(tmp_path, [[_FAIL_K], [_OK_K]])
    results, errors = _run_in_rank_threads(hub, [opts[0].run, opts[1].run])
    assert errors == [None, None]
    out0 = Path(cfgs[0].output_dir).resolve()
    assert _read_json(out0 / "best_params.json")["k_neighbours"] == _OK_K
    history = _read_json(out0 / "search_history.json")
    assert history, "rank 0 must persist rank 1's successful history"
    assert all(math.isfinite(t["score"]) for t in history)
    assert any(t["params"]["k_neighbours"] == _OK_K for t in history)
    assert results[0].best_params["k_neighbours"] == _OK_K
    failed = _read_json(out0 / "failed_trials.json")
    assert any(r["rank"] == 0 for r in failed)


def test_run_threads_all_ranks_failed_raise_consistently(tmp_path) -> None:
    """No successful trial on any rank: every rank raises; rank 0 lists all failures."""
    hub, opts, cfgs = _rank_optimizers(tmp_path, [[_FAIL_K], [_FAIL_K]])
    _, errors = _run_in_rank_threads(hub, [opts[0].run, opts[1].run])
    assert all(isinstance(e, RuntimeError) for e in errors), errors
    assert all("All" in str(e) for e in errors)
    failed = _read_json(Path(cfgs[0].output_dir).resolve() / "failed_trials.json")
    assert {r["rank"] for r in failed} == {0, 1}


def test_run_threads_rank_abort_before_reduction_raises_everywhere(tmp_path) -> None:
    """WR-03: a rank raising outside _objective still joins the reduction; all ranks raise.

    Before the fix the healthy rank blocked in ``gather`` forever (the barrier
    timeout here would surface that as a ``BrokenBarrierError``).
    """
    hub, opts, _ = _rank_optimizers(tmp_path, [[_OK_K], [_OK_K]])

    def _boom(_output_dir):
        raise OSError("load_real failed on rank 1")

    opts[1]._run_tiers = _boom
    _, errors = _run_in_rank_threads(hub, [opts[0].run, opts[1].run])
    assert isinstance(errors[1], OSError), errors
    assert isinstance(errors[0], RuntimeError), errors
    assert "rank 1" in str(errors[0]) and "load_real failed" in str(errors[0])
    assert opts[0]._comm.calls == opts[1]._comm.calls == ["gather", "bcast"]


def test_run_without_comm_reraises_directly(tmp_path) -> None:
    """WR-03: single-process runs re-raise the original exception without collectives."""
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_OK_K]))

    def _boom(_output_dir):
        raise OSError("disk gone")

    opt._run_tiers = _boom
    with pytest.raises(OSError, match="disk gone"):
        opt.run()


class _ContractPropulateSearch:
    """Stand-in for the uninstalled propulate library (dependency, not unit under test).

    Reproduces the verified return contract of ``PropulateSearch.search``
    (eval/search_strategies.py:393-526): every rank evaluates the objective
    (loss = -score), Propulate's final intra-island receive (propulate 1.2.2
    propulator.py ~519) plus ``comm.Barrier()`` make every rank's individuals
    visible on rank 0, non-zero ranks return ``[]`` (~505), and rank 0 returns
    ``(params, -loss)`` for every individual whose loss is not ``inf`` (513).
    """

    hub: _ThreadHub | None = None
    # Ranks that evaluate nothing (e.g. more ranks than individuals to evaluate).
    idle_ranks: tuple = ()

    def search(self, search_space, objective_fn, n_trials, output_dir, warm_start=None):
        hub = type(self).hub
        rank = hub.local.rank
        keys = list(search_space)
        combos = [] if rank in type(self).idle_ranks else list(
            itertools.product(*(search_space[k] for k in keys))
        )
        for combo in combos:
            params = dict(zip(keys, combo))
            loss = -objective_fn(params)
            with hub.lock:
                hub.population.append((params, loss))
        hub.barrier.wait()  # final intra-island receive + comm.Barrier()
        if rank != 0:
            return []
        return [(dict(p), -loss) for p, loss in hub.population if loss != float("inf")]


def test_run_threads_propulate_contract_rank0_all_failed(tmp_path, monkeypatch) -> None:
    """Propulate branch: rank 0's history is already global; the gather adds no duplicate."""
    hub, opts, cfgs = _rank_optimizers(tmp_path, [[_FAIL_K], [_OK_K]], strategy="propulate")
    monkeypatch.setattr(_ContractPropulateSearch, "hub", hub)
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _ContractPropulateSearch)
    results, errors = _run_in_rank_threads(hub, [opts[0].run, opts[1].run])
    assert errors == [None, None]
    out0 = Path(cfgs[0].output_dir).resolve()
    assert _read_json(out0 / "best_params.json")["k_neighbours"] == _OK_K
    history = _read_json(out0 / "search_history.json")
    assert len(history) == 1
    assert math.isfinite(history[0]["score"])
    failed = _read_json(out0 / "failed_trials.json")
    assert any(r["rank"] == 0 for r in failed)
    assert results[0].best_params["k_neighbours"] == _OK_K


def test_run_threads_idle_rank_raises_with_pointer_to_failed_trials(tmp_path, monkeypatch) -> None:
    """A rank with no local trial still raises when no rank succeeded (same decision everywhere)."""
    hub, opts, cfgs = _rank_optimizers(tmp_path, [[_FAIL_K], [_OK_K]], strategy="propulate")
    monkeypatch.setattr(_ContractPropulateSearch, "hub", hub)
    monkeypatch.setattr(_ContractPropulateSearch, "idle_ranks", (1,))
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _ContractPropulateSearch)
    _, errors = _run_in_rank_threads(hub, [opts[0].run, opts[1].run])
    assert all(isinstance(e, RuntimeError) for e in errors), errors
    assert "first local failure: ValueError" in str(errors[0])
    assert "failed_trials.json" in str(errors[1])
    failed = _read_json(Path(cfgs[0].output_dir).resolve() / "failed_trials.json")
    assert [r["rank"] for r in failed] == [0]


class _SuccessDroppingSearch:
    """Strategy stand-in whose evaluated (successful) trial never reaches the result."""

    def search(self, search_space, objective_fn, n_trials, output_dir, warm_start=None):
        objective_fn({k: v[0] for k, v in search_space.items()})
        return []


def test_success_missing_from_history_raises(tmp_path, monkeypatch) -> None:
    """A success that never reaches the merged history is not silently persisted as {}."""
    cfg = _num05_cfg(tmp_path, [_OK_K], strategy="propulate")
    monkeypatch.setattr(optimizer_module, "PropulateSearch", _SuccessDroppingSearch)
    opt = HyperparamOptimizer(cfg)
    with pytest.raises(RuntimeError, match="No successful HPO trial reached rank 0"):
        opt.run()
    assert _read_json(Path(cfg.output_dir).resolve() / "failed_trials.json") == []


class _FakeWorld:
    def __init__(self, size=None, exc=None) -> None:
        self._size, self._exc = size, exc

    def Get_size(self) -> int:  # noqa: N802 (mpi4py API)
        if self._exc is not None:
            raise self._exc
        return self._size


def _fake_mpi4py(world: _FakeWorld) -> types.ModuleType:
    mod = types.ModuleType("mpi4py")
    mod.MPI = types.SimpleNamespace(COMM_WORLD=world)
    return mod


def test_mpi_world_comm_probe(monkeypatch) -> None:
    """COMM_WORLD only for size > 1; None for size 1, missing mpi4py or MPI init errors."""
    from eval.runners.optimizer import _mpi_world_comm

    world2 = _FakeWorld(size=2)
    monkeypatch.setitem(sys.modules, "mpi4py", _fake_mpi4py(world2))
    assert _mpi_world_comm() is world2
    monkeypatch.setitem(sys.modules, "mpi4py", _fake_mpi4py(_FakeWorld(size=1)))
    assert _mpi_world_comm() is None
    monkeypatch.setitem(sys.modules, "mpi4py", _fake_mpi4py(_FakeWorld(exc=RuntimeError("init"))))
    assert _mpi_world_comm() is None
    monkeypatch.setitem(sys.modules, "mpi4py", None)  # import mpi4py -> ImportError
    assert _mpi_world_comm() is None


# ---------------------------------------------------------------------------
# LT-02 (Phase 62): cpd_weighted runs inside HPO (align_result wired) and its
# label-transfer flags reach the Trial record (59-REVIEW IN-09b); over-bound
# fallback frames fail the trial (IN-09c).
# ---------------------------------------------------------------------------

from eval.stages import AlignmentStage, LabelTransferStage  # noqa: E402

_TEST_FLAG = "test flag: dependency wrapper"


class _FlaggingLabelTransferStage(LabelTransferStage):
    """Real LabelTransferStage whose result carries one extra known flag."""

    def run(self, source, target, params, align_result=None):
        result = super().run(source, target, params, align_result=align_result)
        return result.model_copy(update={"flags": [*result.flags, _TEST_FLAG]})


def _injected_alignment_stage(n_zero: int):
    """Real AlignmentStage whose posterior has its first ``n_zero`` receiver columns zeroed.

    ``estep_results[tk].pmat`` has shape ``(n_aligned_source, n_target)``; with
    the default ``label_source="source"`` the target points are the receivers
    (the stage transposes), so zeroing columns zeroes receiver rows.
    """

    class _InjectedAlignmentStage(AlignmentStage):
        def run(self, source, target, params):
            result = super().run(source, target, params)
            injected = {}
            for tk, est in result.estep_results.items():
                pmat = est.pmat.clone()
                pmat[:, :n_zero] = 0.0
                injected[tk] = est._replace(pmat=pmat)
            return result.model_copy(update={"estep_results": injected})

    return _InjectedAlignmentStage


_CENTRES = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
_PROVIDER_CLASSES = (10, 11, 12)

_CPD_WEIGHTED_PARAMS = {
    "window_size": 3,
    "step": 1,
    "cpd_penalty": "rigid",
    "dtw_dist_fn": "euclidean",
    "n_breakpoints": 5,
    "alignment_method": "cpd",
    "k_neighbours": 3,
    "dist_metric": "euclidean",
    "smoothing": 0.0,
    "threshold": 0.5,
}


def _clustered(per_cluster: int, seed: int) -> zRegPointCloud:
    """Labelled points around the three _CENTRES (copied from test_cpd_weighted_direction)."""
    gen = torch.Generator().manual_seed(seed)
    cluster = torch.arange(3).repeat_interleave(per_cluster)
    pos = _CENTRES[cluster] + 0.5 * torch.randn(cluster.numel(), 3, generator=gen)
    return zRegPointCloud(pos=pos, label=torch.tensor(_PROVIDER_CLASSES)[cluster], id=None)


def _cpd_weighted_trajectories() -> tuple[dict, dict]:
    """3 frames, 42 labelled points each on both sides."""
    source = {k: _clustered(14, 300 + 2 * k) for k in range(3)}
    target = {k: _clustered(14, 301 + 2 * k) for k in range(3)}
    return source, target


def _cpd_weighted_paired_cfg(tmp_path) -> EvalConfig:
    return EvalConfig(
        data_path=str(tmp_path / "source.mat"),
        target_data_path=str(tmp_path / "target.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        run_alignment=True,
        alignment_method="cpd",
        run_label_transfer=True,
        label_transfer_method="cpd_weighted",
        search_strategy="grid",
        search_space={"cpd_penalty": ["rigid"]},
        save_plots=False,
    )


def _run_cpd_weighted_objective(tmp_path, monkeypatch, caplog):
    source, target = _cpd_weighted_trajectories()
    opt = HyperparamOptimizer(_cpd_weighted_paired_cfg(tmp_path))
    monkeypatch.setattr(opt._factory, "load_target", lambda: target)
    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective(dict(_CPD_WEIGHTED_PARAMS), source, "dev", hist)
    return opt, hist, score


def test_objective_cpd_weighted_runs_in_hpo(tmp_path, monkeypatch, caplog) -> None:
    """cpd_weighted gets the alignment posterior inside HPO (no 'requires align_result')."""
    opt, hist, score = _run_cpd_weighted_objective(tmp_path, monkeypatch, caplog)
    assert opt._failed_trials == []
    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(score)
    assert math.isfinite(hist[0].metrics.f1_score)


def test_multiseed_cpd_weighted_runs_in_hpo(tmp_path, caplog) -> None:
    """The multiseed path wires align_result too (synthetic subsample pair, rigid CPD)."""
    params = {"cpd_penalty": "rigid", "window_size": 3, "k_neighbours": 3}
    cfg = _cfg(
        tmp_path,
        search_space={"cpd_penalty": ["rigid"], "window_size": [3], "k_neighbours": [3]},
        alignment_method="cpd",
        label_transfer_method="cpd_weighted",
    )
    opt = HyperparamOptimizer(cfg)
    hist: list = []
    with caplog.at_level(logging.WARNING, logger=_OPT_LOGGER):
        score = opt._objective(params, {}, "full", hist)
    assert opt._failed_trials == []
    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(score)
    assert math.isfinite(hist[0].metrics.f1_score)


def test_trial_flags_default_and_round_trip() -> None:
    """Trial.flags defaults to [] and survives model_dump/model_validate."""
    assert _trial(3, 0.4).flags == []
    flagged = Trial(
        params={"k_neighbours": 3}, score=0.4, metrics=_zero_metrics(), tier="sanity",
        flags=["frame 0: a", "frame 1: b"],
    )
    restored = Trial.model_validate(flagged.model_dump())
    assert restored.flags == ["frame 0: a", "frame 1: b"]


def test_objective_cpd_weighted_fallback_flag_on_trial(tmp_path, monkeypatch, caplog) -> None:
    """One zero-mass receiver row per frame: the real stage fallback flags reach the Trial."""
    monkeypatch.setattr(optimizer_module, "AlignmentStage", _injected_alignment_stage(1))
    opt, hist, score = _run_cpd_weighted_objective(tmp_path, monkeypatch, caplog)
    assert opt._failed_trials == []
    assert _trial_failed_records(caplog) == []
    assert len(hist) == 1
    assert math.isfinite(score)
    flags = hist[0].flags
    assert len(flags) == 3
    for k in range(3):
        assert f"frame {k}" in flags[k]
        assert "1 of 42" in flags[k]
    # what save_best_params writes to search_history.json
    dumped = hist[0].model_dump()
    assert dumped["flags"] == flags
    assert json.loads(json.dumps(dumped["flags"])) == flags


def test_multiseed_flags_prefixed_by_seed(tmp_path, monkeypatch) -> None:
    """Multiseed trials carry every seed's flags, prefixed 'seed {s}: ', in seed order."""
    monkeypatch.setattr(optimizer_module, "LabelTransferStage", _FlaggingLabelTransferStage)
    opt = HyperparamOptimizer(_cfg(tmp_path))
    hist: list = []
    opt._score_subsample_pair_multiseed({}, dict(opt._default_params), "full", [1, 2], hist)
    assert len(hist) == 1
    assert hist[0].flags == [f"seed 1: {_TEST_FLAG}", f"seed 2: {_TEST_FLAG}"]


def test_multiseed_flags_empty_without_flags(tmp_path) -> None:
    """Without label-transfer flags the multiseed Trial has flags == []."""
    opt = HyperparamOptimizer(_cfg(tmp_path))
    hist: list = []
    opt._score_subsample_pair_multiseed({}, dict(opt._default_params), "full", [1, 2], hist)
    assert hist[0].flags == []


_RANK1_FLAGS = ["seed 1: frame 7: x", "frame 2: y"]


def _flagged_trial() -> Trial:
    return Trial(
        params={"k_neighbours": 5}, score=0.6, metrics=_zero_metrics(), tier="sanity",
        flags=list(_RANK1_FLAGS),
    )


def test_reduce_trial_outcomes_keeps_flags_across_ranks(tmp_path) -> None:
    """Non-empty Trial.flags of rank 1 reach rank 0's merged history unchanged (pickle transport)."""
    hub, opts, _ = _rank_optimizers(tmp_path, [[_OK_K], [_OK_K]])
    local = [[_trial(3, 0.4)], [_flagged_trial()]]
    results, errors = _run_in_rank_threads(
        hub, [lambda r=r: opts[r]._reduce_trial_outcomes(local[r]) for r in range(2)]
    )
    assert errors == [None, None]
    merged0 = {t.params["k_neighbours"]: t for t in results[0][0]}
    assert set(merged0) == {3, 5}
    assert merged0[5].flags == _RANK1_FLAGS
    assert merged0[3].flags == []


def test_reduce_trial_outcomes_keeps_flags_single_process(tmp_path) -> None:
    """The single-process reduction keeps non-empty Trial.flags."""
    opt = HyperparamOptimizer(_num05_cfg(tmp_path, [_OK_K]))
    merged = opt._reduce_trial_outcomes([_flagged_trial()])[0]
    assert merged[0].flags == _RANK1_FLAGS


def test_objective_cpd_weighted_fallback_bound_fails_trial(tmp_path, monkeypatch, caplog) -> None:
    """IN-09c: 32 of 42 receiver rows bad (> 50%) -> the real stage raises, the trial fails (-inf)."""
    monkeypatch.setattr(optimizer_module, "AlignmentStage", _injected_alignment_stage(32))
    opt, hist, score = _run_cpd_weighted_objective(tmp_path, monkeypatch, caplog)
    assert score == float("-inf")
    assert hist == []
    assert opt._n_succeeded == 0
    assert len(opt._failed_trials) == 1
    record = opt._failed_trials[0]
    assert record["error_type"] == "ValueError"
    assert "max_fallback_fraction" in record["error"]
    assert "0.5" in record["error"]
    assert "frame 0" in record["error"]


def test_objective_cpd_weighted_all_bad_fails_trial(tmp_path, monkeypatch, caplog) -> None:
    """Every receiver row bad -> the real stage raises its all-bad error, the trial fails."""
    monkeypatch.setattr(optimizer_module, "AlignmentStage", _injected_alignment_stage(42))
    opt, hist, score = _run_cpd_weighted_objective(tmp_path, monkeypatch, caplog)
    assert score == float("-inf")
    assert hist == []
    assert len(opt._failed_trials) == 1
    assert opt._failed_trials[0]["error_type"] == "ValueError"
    assert "every receiver point" in opt._failed_trials[0]["error"]
    assert "frame 0" in opt._failed_trials[0]["error"]
