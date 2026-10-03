"""HyperparamOptimizer: orchestration layer for the FRAME-09 HPO framework.

This module implements ``HyperparamOptimizer``, which composes ``DataFactory``,
``AlignmentStage``, ``LabelTransferStage``, ``MetricsEngine``, and the search
strategy classes (``GridSearch``, ``RandomSearch``, ``BayesianSearch``) into a
tiered hyperparameter search pipeline.

Key design decisions implemented here:

- **D-01** ``run()`` stops at ``config.tier`` — if ``tier="sanity"``, only the
  sanity tier runs; if ``tier="dev"``, runs sanity then dev; if ``tier="full"``,
  runs all three in sequence.
- **D-02** Each tier uses a different dataset and trial count.  ``_tier_dataset``
  provides the per-tier dataset slice.
- **D-03** ``prune_candidates(history, keep_top_k)`` prunes candidate list (top-k
  ``Trial`` param dicts by score); the search space is unchanged between tiers.
- **D-04** Top-k pruned params from tier N are passed as ``warm_start`` to tier N+1.
- **D-05** Unified search space — list of values per param key.
- **D-06** Search space keys are exact param names passed to stages (no mapping).
- **D-07** ``_objective(params) → float`` calls ``AlignmentStage``,
  ``LabelTransferStage``, and ``MetricsEngine`` **directly** — does NOT call
  ``EvaluationRunner``.  Keeps each trial lightweight.
- **D-08** Dataset passed to ``_objective`` is the tier-specific subset from
  ``_tier_dataset``, not the full dataset.
- **D-09** If a trial raises an exception, ``_objective`` logs it with its
  traceback, records it as a failure and returns ``float("-inf")`` (worst for the
  maximised objective; Phase 59 NUM-05).  The search continues.  Failures of all
  MPI ranks are gathered to rank 0 into ``SearchResult.failed_trials`` /
  ``failed_trials.json``; successful trials of all ranks are merged on rank 0
  (finite scores only); a run with no successful trial in the merged history
  raises ``RuntimeError`` on every rank.
- **D-10** Study name fixed as ``"zreg-hpo"``; re-running with the same
  ``output_dir`` resumes existing study (``load_if_exists=True``).
- **D-11** SQLite at ``{output_dir}/optuna.db``.
- **D-12** ``GridSearch`` and ``RandomSearch`` are stateless; ``save_best_params``
  writes JSON at the end of ``run()``.
- **D-04** ``_detect_backend()`` resolves ``search_strategy="auto"`` → ``"propulate"``
  (MPI world_size > 1 or SLURM_JOB_ID set) or ``"bayesian"`` (fallback). EXT-03.
- **D-11 / Phase 62 RD-5..RD-7** The Propulate dispatch branch passes the same
  history-bound objective as every other strategy, so the rank that evaluates
  an individual keeps the real ``Trial`` (metrics, tier, flags) and
  ``_reduce_trial_outcomes`` gathers it to rank 0.  The ``(params, score)``
  pairs ``PropulateSearch.search()`` returns on rank 0 only back-fill
  individuals without an evaluation record of this run (e.g. restored from a
  Propulate checkpoint) as zero-metric placeholders carrying
  ``PROPULATE_PLACEHOLDER_FLAG``; a pair matching a real trial is never
  duplicated.

Security mitigations:

- **T-22-01** ``__init__`` validates ``config.search_space`` — each value must be a
  non-empty list.  Prevents non-list values from reaching ``itertools.product`` or
  ``suggest_categorical``.
- **T-22-02** ``run()`` resolves ``output_dir`` via ``Path(...).resolve()`` as its
  first executable line — prevents path traversal (``../../../etc/passwd``).

Notes
-----
**Import order — macOS-ARM SIGABRT (Pitfall 7):**
``zreg.*`` MUST precede ``torch``; ``torch`` MUST precede ``optuna``; ``optuna``
MUST precede ``eval.*``.

**Anti-pattern:** Do NOT call ``EvaluationRunner.run()`` inside ``_objective``
(D-07) — each call writes JSON, optionally renders plots, and calls DataFactory,
making each trial 10–100x slower than needed.

**Default params merge (Pitfall 4):** ``_objective`` merges ``self._default_params``
with the trial's ``params`` before calling stages, so stage ``validate_params``
never fails when the search space covers only a subset of required keys.

**Sanity tier labels (Pitfall 5):** ``_tier_dataset("sanity")`` calls
``generate_labels(generate_trajectory(...))`` so ``pc["id"]`` is populated and
``DataFactory.get_ground_truth()`` returns valid tensors. Per D-13, this call
consults ``self.config.label_generation`` when set (forwarding its
``n_labels``/``label_specs``/``mode``/``seed`` fields), falling back to the
hardcoded ``n_labels=4, seed=42`` literal otherwise.

**JSON serialisation:** Always ``model_dump()`` + ``json.dump()``.  The JSON
shortcut raises for ``torch.Tensor`` fields (Pitfall 2 from eval/types.py).

**Phase 30 source/target dispatch (Pitfall 7):** sanity tier reuses
``tier_dataset`` as both source and target (D-03 smoke-test parity); dev/full
tiers call ``self._factory.load_target()`` to obtain the target trajectory from
``config.target_data_path``.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/runners/eval_runner.py:46-51.
from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_labels, generate_trajectory

import torch

import optuna

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.metrics import MetricsEngine
from eval.runners._label_direction import f1_unavailable_reason, resolve_label_transfer_pair
from eval.search_strategies import BayesianSearch, GridSearch, PropulateSearch, RandomSearch, SobolSearch
from eval.stages import AlignmentStage, LabelTransferStage
from eval.types import SearchResult, StageMetrics, Trial

__all__ = ["HyperparamOptimizer", "PROPULATE_PLACEHOLDER_FLAG"]

_log = logging.getLogger(__name__)

# Module-level tier trial count constants (RESEARCH Q3 resolved)
SANITY_N_TRIALS: int = 5
DEV_N_TRIALS: int = 20

# Phase 57 GT-04/D-07: multi-seed averaging for subsample_pair mode is gated
# to this tier only — sanity/dev tiers gracefully fall back to a single seed
# (see _resolve_subsample_pair_seeds) rather than paying the full N-seed cost.
SUBSAMPLE_PAIR_MULTISEED_TIER: str = "full"

# Phase 62 RD-6/RD-8: flag of a search_history.json entry that Propulate returned
# without an evaluation record of this run; its metrics are zero placeholders.
PROPULATE_PLACEHOLDER_FLAG: str = (
    "propulate: individual returned without an evaluation record from this run "
    "(e.g. restored from a Propulate checkpoint); metrics are placeholders"
)


def _apply_transform_to_dataset(
    dataset: dict,
    transform_spec: dict,
    config: "EvalConfig",
) -> dict:
    """Apply transform_spec to dataset using a scratch DataFactory (D-11).

    This helper is used by ``_objective`` for the sanity tier in synthetic mode.
    It MUST NOT touch the caller's ``_factory`` instance — using a scratch
    ``DataFactory`` avoids overwriting ``_synthetic_target`` on the main factory
    (Pitfall 3 from Phase 31 RESEARCH.md, D-11 from CONTEXT.md).

    Parameters
    ----------
    dataset : dict
        Source dataset to transform.
    transform_spec : dict
        Transform specification (e.g. ``{"type": "noise", "sigma": 0.1}``).
        The ``"type"`` key is stripped before passing to ``augment()`` dispatch.
    config : EvalConfig
        Current evaluation config — used to construct the scratch config via
        ``model_copy``.  The caller's config is never mutated.

    Returns
    -------
    dict
        Transformed dataset produced by ``scratch_factory.augment(dataset)``.
    """
    augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
    scratch_cfg = config.model_copy(update={"augmentation_params": augment_params})
    scratch_factory = DataFactory(scratch_cfg)
    return scratch_factory.augment(dataset)


def _mpi_world_comm():
    """Return mpi4py's ``COMM_WORLD`` when running on more than one MPI rank.

    Returns
    -------
    mpi4py.MPI.Comm or None
        ``MPI.COMM_WORLD`` if mpi4py imports and ``COMM_WORLD.Get_size() > 1``;
        ``None`` for single-process runs (no mpi4py, or a world of size 1), in
        which case the outcome reduction uses local data without collectives.

    Notes
    -----
    Same probe pattern as ``HyperparamOptimizer._detect_backend``: an
    ``ImportError`` means "not an MPI environment" and is silent; other MPI
    initialisation errors are logged at DEBUG level (Pitfall 6).
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            return comm
    except ImportError:
        pass  # not an MPI environment
    except Exception as exc:  # Pitfall 6: MPI init failures are not always ImportError
        _log.debug("MPI communicator probe failed: %s", exc)
    return None


def _merge_trial_histories(
    per_rank_histories: list[list[dict[str, Any]]],
    start_rank: int = 0,
) -> tuple[list[Trial], list[dict[str, Any]]]:
    """Merge per-rank successful-trial histories, keeping finite scores only.

    Parameters
    ----------
    per_rank_histories : list[list[dict[str, Any]]]
        One list per rank (in rank order) of ``Trial.model_dump()`` dicts —
        the pickle-safe transport form used by ``_reduce_trial_outcomes``.
    start_rank : int, optional
        Rank number of the first entry in ``per_rank_histories``.  Default 0.

    Returns
    -------
    tuple[list[Trial], list[dict[str, Any]]]
        ``(merged, bad)``: the finite-score trials of all ranks in rank order,
        and one failure record per non-finite trial with keys ``params``,
        ``tier``, ``error``, ``error_type`` (``"NonFiniteScore"``) and ``rank``.

    Notes
    -----
    Phase 59 NUM-05: ``_objective`` records a trial whatever its score, so a
    non-finite score (for example a NaN composite) would otherwise reach best
    selection.  Filtering here guarantees for every strategy, Propulate
    included (its evaluating ranks contribute their real trials since Phase 62
    RD-5), that no ``-inf``/``NaN`` trial is ever selected as best or written
    to ``search_history.json``.  Propulate placeholders are added after this
    merge by ``_propulate_placeholders``, which skips non-finite scores too.
    """
    merged: list[Trial] = []
    bad: list[dict[str, Any]] = []
    for offset, history in enumerate(per_rank_histories):
        rank = start_rank + offset
        for entry in history:
            trial = Trial.model_validate(entry)
            if math.isfinite(trial.score):
                merged.append(trial)
                continue
            _log.warning(
                "Dropping trial with non-finite score %r from rank %d (params=%s)",
                trial.score,
                rank,
                trial.params,
            )
            bad.append({
                "params": dict(trial.params),
                "tier": trial.tier,
                "error": f"non-finite score {trial.score!r}",
                "error_type": "NonFiniteScore",
                "rank": rank,
            })
    return merged, bad


def _trial_key(params: dict[str, Any], score: float) -> tuple[str, float]:
    """Identity of one evaluation for Propulate reconciliation (Phase 62 RD-6)."""
    return json.dumps(params, sort_keys=True, default=repr), score


def _propulate_placeholders(
    merged: list[Trial],
    returned: list[tuple[str, dict[str, Any], float]],
) -> list[Trial]:
    """Placeholder trials for Propulate individuals without an evaluation record.

    Parameters
    ----------
    merged : list[Trial]
        The real, finite-score trials of every rank (``_merge_trial_histories``).
    returned : list[tuple[str, dict[str, Any], float]]
        ``(tier, params, score)`` for every pair ``PropulateSearch.search``
        returned on this rank, in return order (empty on non-zero ranks).

    Returns
    -------
    list[Trial]
        One ``Trial`` per returned pair whose ``(params, score)`` matches no
        trial in ``merged``, in return order, with zero-filled metrics, the
        returned tier and ``flags=[PROPULATE_PLACEHOLDER_FLAG]``.

    Notes
    -----
    Phase 62 RD-6.  The key is ``(json.dumps(params, sort_keys=True,
    default=repr), score)`` and ignores the tier: the next tier's Propulator
    reloads the checkpoint the previous tier wrote to the same ``output_dir``,
    so a pair of an earlier tier would otherwise be duplicated under the wrong
    tier.  The returned score is the exact negation of the loss Propulate
    stored, i.e. exactly the objective's return value, so equality matching
    is reliable.  Placeholders are de-duplicated by the same key and
    non-finite scores are skipped.  Placeholders keep checkpoint-restored
    individuals selectable as best (Phase 63 HPC-02 resume).
    """
    seen = {_trial_key(t.params, t.score) for t in merged}
    out: list[Trial] = []
    for tier, params, score in returned:
        if not math.isfinite(score):
            continue
        key = _trial_key(params, score)
        if key in seen:
            continue
        seen.add(key)
        out.append(Trial(
            params=dict(params),
            score=score,
            metrics=StageMetrics(
                chamfer_distance=0.0,
                hausdorff_distance=0.0,
                path_smoothness=0.0,
                temporal_stability=0.0,
                f1_score=0.0,
                knn_consistency=0.0,
            ),
            tier=tier,
            flags=[PROPULATE_PLACEHOLDER_FLAG],
        ))
    return out


def _resolve_subsample_pair_seeds(transform_spec: dict, tier_name: str) -> list[int]:
    """Resolve ``transform_spec['seed']`` into the list of seeds a trial should use (D-07).

    Phase 57 GT-04/GT-06: ``transform_spec['seed']`` for ``subsample_pair``
    mode is normally a single ``int`` (D-06's default, unchanged fixed-seed
    path). It MAY optionally be a ``list`` of ints, in which case D-07's
    opt-in multi-seed averaging applies — but ONLY on the ``full`` tier
    (``SUBSAMPLE_PAIR_MULTISEED_TIER``); sanity/dev tiers gracefully fall
    back to the list's first entry so they never silently pay the full
    N-seed cost.

    Parameters
    ----------
    transform_spec : dict
        The current ``subsample_pair`` transform spec. ``"seed"`` defaults to
        ``42`` when absent, matching :meth:`DataFactory.generate_subsample_pair`.
    tier_name : str
        Current tier ("sanity" / "dev" / "full").

    Returns
    -------
    list[int]
        ``[seed]`` for the default int-seed case (every non-list-seed call,
        including every call Plan 57-03's code already makes);
        ``[seed[0]]`` for a list seed on a non-``full`` tier (graceful
        fallback); ``list(seed)`` unchanged for a list seed on the ``full``
        tier.

    Raises
    ------
    ValueError
        If ``seed`` is a list and empty, or if ``seed`` is neither an ``int``
        nor a ``list``.

    Notes
    -----
    **Never raises for a list seed on a non-full tier** — deliberately a
    graceful fallback (logged warning), not a raised exception, since
    ``_objective``'s broad ``except Exception`` handler would otherwise turn
    a raised validation error into a failed trial (``-inf``, recorded in
    ``SearchResult.failed_trials``) instead of a clear configuration signal
    (D-07).
    """
    seed = transform_spec.get("seed", 42)
    if isinstance(seed, int):
        return [seed]
    if isinstance(seed, list):
        if len(seed) == 0:
            raise ValueError(
                "_resolve_subsample_pair_seeds: transform_spec['seed'] list "
                "must be non-empty."
            )
        if tier_name != SUBSAMPLE_PAIR_MULTISEED_TIER:
            _log.warning(
                "_resolve_subsample_pair_seeds: multi-seed averaging (D-07) is "
                "gated to the %r tier; tier %r will use only seed[0]=%r instead "
                "of averaging across all %d seeds.",
                SUBSAMPLE_PAIR_MULTISEED_TIER,
                tier_name,
                seed[0],
                len(seed),
            )
            return [seed[0]]
        return list(seed)
    raise ValueError(
        "_resolve_subsample_pair_seeds: transform_spec['seed'] must be an "
        f"int or a list of ints, got {type(seed).__name__}."
    )


class HyperparamOptimizer:
    """Tiered hyperparameter search over AlignmentStage + LabelTransferStage.

    Orchestrates one or more search tiers (sanity / dev / full) in sequence,
    passing pruned top-k candidates from each tier as warm-start seeds for the
    next.  Writes ``best_params.json`` and ``search_history.json`` to
    ``config.output_dir`` at the end of ``run()``.

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration.  Must have non-empty list values in
        ``config.search_space`` (T-22-01 validation enforced at construction).

    Raises
    ------
    ValueError
        If ``config.search_space`` contains any non-list or empty-list value
        (T-22-01 security guard).
    ValueError
        If both ``config.run_alignment`` and ``config.run_label_transfer`` are
        ``False`` (fail-fast guard — at least one stage required).
    """

    def __init__(
        self,
        config: EvalConfig,
        warm_start: list[dict] | None = None,
    ) -> None:
        # T-22-01: validate search_space — each value must be a non-empty list
        for k, v in config.search_space.items():
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(
                    f"search_space[{k!r}] must be a non-empty list, got {v!r}"
                )

        # Fail-fast guard (mirrors eval_runner.py)
        if not config.run_alignment and not config.run_label_transfer:
            raise ValueError("At least one stage must be enabled")

        self.config = config
        self._engine = MetricsEngine(config)
        self._factory = DataFactory(config)
        # Seed from a prior run's best_params (cross-job forwarding, EXT-04).
        # Stored here; injected as initial warm_start in run() before the tier loop.
        self._initial_warm_start = warm_start
        # Phase 59 NUM-04: the F1-unavailable reason is logged once per instance.
        self._f1_unavailable_logged = False
        # Phase 59 NUM-05: per-run trial outcome bookkeeping (reset in run()).
        self._failed_trials: list[dict[str, Any]] = []
        self._n_succeeded: int = 0
        # Phase 62 RD-6: (tier, params, score) pairs PropulateSearch returned on
        # this rank (rank 0 only; [] elsewhere), reconciled in the reduction.
        self._propulate_returned: list[tuple[str, dict[str, Any], float]] = []
        # MPI communicator for the outcome reduction. None => resolved lazily in
        # run() via _mpi_world_comm(); tests inject a communicator here.
        self._comm = None

        # _default_params: fallback for all 9 required stage keys (Pitfall 4)
        # These fill any gaps when search_space covers only a subset of required keys.
        self._default_params: dict[str, Any] = {
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

    def run(self) -> SearchResult:
        """Execute tiered hyperparameter search and return the best result.

        Runs tiers up to (and including) ``self.config.tier`` in sequence.  Each
        tier uses a tier-specific dataset slice and trial count.  Top-k pruned
        candidates from each tier are seeded as warm-start for the next tier.

        Returns
        -------
        SearchResult
            Frozen pydantic model with ``best_params``, ``best_score``,
            ``history`` (the finite-score successful ``Trial`` objects of all
            tiers and, on rank 0, of all MPI ranks), ``tier`` (the ceiling tier
            from ``config.tier``) and ``failed_trials`` (failure records; on
            rank 0 those of every rank).  Non-zero MPI ranks return an empty
            result and never write to disk.

        Raises
        ------
        RuntimeError
            On every rank, when at least one trial was attempted on some rank
            but the merged (all-rank) history holds no successful trial.
            Rank 0 writes ``failed_trials.json`` before raising.

        Notes
        -----
        **T-22-02:** ``output_dir`` is resolved via ``Path(...).resolve()`` as
        the first line to prevent path traversal attacks.

        **Phase 59 NUM-05 — global outcome reduction:** after the tier loop
        every rank calls ``_reduce_trial_outcomes`` on its history (gather to
        rank 0, then broadcast of the global counts) before any
        rank-dependent branch.  The persisted result on rank 0 is built from
        the merged history of all ranks, so rank 0 writes another rank's best
        params even if all of its own trials failed.  Under Propulate every
        rank keeps the real trials it evaluated (Phase 62 RD-5) and the
        gather brings them to rank 0; each individual is evaluated on exactly
        one rank, so no duplicates arise.  Pairs ``PropulateSearch.search``
        returns on rank 0 without an evaluation record of this run (e.g.
        checkpoint-restored) are added as flagged placeholders (RD-6).

        **Rank abort (WR-03):** the pre-populate step and the tier loop run
        in ``_run_tiers`` inside ``try/except BaseException``.  With a
        communicator, a rank that raises still joins the reduction and passes
        its abort reason; afterwards it re-raises its own exception and every
        other rank raises ``RuntimeError`` naming the aborted rank(s), instead
        of blocking in ``gather`` until walltime.  Collectives *inside* a
        strategy (Propulate) cannot be rescued this way.

        **Known limitation (Phase 61, MPI robustness):** grid / random /
        sobol / bayesian under ``mpirun -n N`` run the same search on every
        rank, so the merged history can contain the same params evaluated by
        several ranks.  Each entry is a real evaluation and best selection is
        unaffected.
        """
        # T-22-02: resolve output_dir to prevent path traversal
        output_dir = Path(self.config.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 59 NUM-05: per-run bookkeeping — a second run() on the same
        # instance reports only its own outcomes.
        self._failed_trials = []
        self._n_succeeded = 0
        self._propulate_returned = []
        self._aborted_ranks: list[tuple[int, str]] = []
        # Resolved lazily so constructing an optimizer does not initialise MPI.
        if self._comm is None:
            self._comm = _mpi_world_comm()

        # WR-07: a positive F1 weight is dead weight when F1 is zero-filled in
        # every trial; it caps the composite score below 1.0 and makes
        # best_score values incomparable with runs where F1 is available.
        f1_weight = self.config.metric_weights.get("f1", 0.0)
        f1_reason = f1_unavailable_reason(self.config)
        if f1_reason is not None and f1_weight > 0:
            _log.warning(
                "metric_weights['f1']=%s but F1 is zero-filled in every trial (%s); "
                "the composite score is capped below 1.0. Set metric_weights.f1 to 0.0 "
                "(ranking-neutral) to remove the dead weight.",
                f1_weight,
                f1_reason,
            )

        # WR-03: an exception raised on one rank before the outcome reduction
        # (pre-populate / load_real() I/O, _tier_dataset, an Optuna storage
        # error, ...) must not leave the other ranks blocked in gather. With a
        # communicator, every rank still reaches _reduce_trial_outcomes and
        # reports whether it aborted; afterwards every rank raises.
        all_history: list[Trial] = []
        abort_exc: BaseException | None = None
        try:
            all_history = self._run_tiers(output_dir)
        except BaseException as exc:
            if self._comm is None:
                raise
            abort_exc = exc
            _log.error(
                "HPO aborted on MPI rank %d before the outcome reduction",
                self._comm.Get_rank(),
                exc_info=True,
            )

        # Phase 59 NUM-05: global outcome reduction. Called unconditionally by
        # every rank, before any rank-dependent branch (collective-order
        # contract, see _reduce_trial_outcomes).
        merged_history, failed_records, n_ok, n_fail, n_hist = self._reduce_trial_outcomes(
            all_history, aborted=None if abort_exc is None else repr(abort_exc)
        )
        if abort_exc is not None:
            raise abort_exc
        if self._aborted_ranks:
            detail = "; ".join(f"rank {r}: {msg}" for r, msg in self._aborted_ranks)
            raise RuntimeError(f"HPO aborted on other MPI rank(s): {detail}")

        # Construct SearchResult from the merged all-rank history. Every trial
        # in merged_history has a finite score (_merge_trial_histories filter),
        # so a failed (-inf) or NaN trial can never be selected as best.
        if not merged_history:
            result = SearchResult(
                best_params={},
                best_score=0.0,
                history=[],
                tier=self.config.tier,
                failed_trials=list(failed_records),
            )
        else:
            best_trial = max(merged_history, key=lambda t: t.score)
            result = SearchResult(
                best_params=dict(best_trial.params),
                best_score=best_trial.score,
                history=list(merged_history),
                tier=self.config.tier,
                failed_trials=list(failed_records),
            )

        # Raise rule — identical on every rank because n_ok / n_fail / n_hist
        # come from rank 0's broadcast. Zero attempted trials (n_ok + n_fail
        # == 0) keeps the old empty-result behaviour.
        if n_hist == 0 and (n_ok + n_fail) > 0:
            if self._is_rank_zero():
                with open(output_dir / "failed_trials.json", "w") as f:
                    json.dump(failed_records, f, indent=2)
            if n_fail > 0:
                if self._failed_trials:
                    first = self._failed_trials[0]
                    detail = f"first local failure: {first['error_type']}: {first['error']}"
                else:
                    detail = f"see {output_dir / 'failed_trials.json'} on rank 0"
                raise RuntimeError(
                    f"All {n_fail} HPO trials failed across all ranks "
                    f"(no successful trial to select best params from); {detail}"
                )
            raise RuntimeError(
                f"No successful HPO trial reached rank 0 although {n_ok} trial(s) "
                "succeeded; the merged history is empty."
            )

        # Only rank 0 writes results: it holds the merged history of every
        # rank (non-zero ranks get an empty merged_history from the reduction)
        # and concurrent writes from many ranks would corrupt best_params.json.
        if self._is_rank_zero():
            self.save_best_params(result, output_dir)
        return result

    def _run_tiers(self, output_dir: Path) -> list[Trial]:
        """Pre-populate the synthetic target and run every tier up to the ceiling.

        Extracted from ``run()`` (WR-03) so ``run()`` can catch an exception
        raised here and still take part in the MPI outcome reduction.

        Parameters
        ----------
        output_dir : Path
            Resolved output directory (Bayesian / Propulate storage).

        Returns
        -------
        list[Trial]
            This rank's successful trials (``all_history``).
        """

        # Synthetic mode: pre-populate _synthetic_target so dev/full tiers can access it.
        # Must be done before the tier loop because _objective reads _factory._synthetic_target
        # directly for tier_name != "sanity" (it cannot call generate_target() per-trial
        # without overwriting shared state — Pitfall 3 from Phase 31 RESEARCH.md).
        #
        # Gate on a non-sanity ceiling (D-11): when config.tier == "sanity" only the sanity
        # tier runs, and it must use the isolated scratch factory (_apply_transform_to_dataset)
        # — calling the main factory's generate_target() there would violate sanity isolation.
        if (
            self.config.pipeline_mode == "synthetic"
            and self.config.transform_spec is not None
            and self.config.tier != "sanity"
        ):
            # Phase 57 GT-06/D-06: subsample_pair mode pre-populates via
            # generate_subsample_pair() (single fixed seed for every trial)
            # instead of generate_target(); the rigid/noise path below is
            # otherwise completely unchanged.
            if self._is_subsample_pair_mode():
                # Phase 59 NUM-03 (D-04): a list-valued seed pre-populates with
                # seed[0] so the dev/full tier datasets exist. Sanity/dev tiers
                # score seed[0] only (graceful fallback in
                # _resolve_subsample_pair_seeds); the full tier averages all
                # seeds via _score_subsample_pair_multiseed, which builds its
                # own per-seed pairs. Before Phase 59 this branch skipped the
                # pre-populate, so _tier_dataset returned None and every
                # sanity/dev trial failed silently.
                seed = self.config.transform_spec.get("seed", 42)
                prepopulate_spec = self.config.transform_spec
                if isinstance(seed, list):
                    if len(seed) == 0:
                        raise ValueError(
                            "_resolve_subsample_pair_seeds: transform_spec['seed'] list "
                            "must be non-empty."
                        )
                    first_seed = seed[0]
                    prepopulate_spec = {**self.config.transform_spec, "seed": first_seed}
                base = (
                    None
                    if self.config.transform_spec.get("synthesize", False)
                    else self._factory.load_real()
                )
                self._factory.generate_subsample_pair(base, prepopulate_spec)
            else:
                real_source = self._factory.load_real()
                self._factory.generate_target(real_source, self.config.transform_spec)

        # Tier execution sequence — stop at config.tier ceiling (D-01)
        TIER_SEQUENCE = ["sanity", "dev", "full"]
        tiers_to_run = TIER_SEQUENCE[: TIER_SEQUENCE.index(self.config.tier) + 1]

        all_history: list[Trial] = []
        warm_start: list[dict] | None = self._initial_warm_start

        for tier_name in tiers_to_run:
            tier_dataset = self._tier_dataset(tier_name)

            if tier_name == "sanity":
                n_trials = SANITY_N_TRIALS
            elif tier_name == "dev":
                n_trials = DEV_N_TRIALS
            else:
                n_trials = self.config.n_trials

            strategy_name = self.config.search_strategy
            if strategy_name == "auto":
                strategy_name = self._detect_backend()  # D-03: resolved per-tier, not persisted

            # Closure captures tier_dataset and tier_name to avoid late-binding issues
            def make_objective(
                td: dict,
                tn: str,
                hist: list[Trial],
            ):
                def objective(params: dict) -> float:
                    return self._objective(params, td, tn, hist)
                return objective

            obj = make_objective(tier_dataset, tier_name, all_history)

            if strategy_name == "grid":
                GridSearch().search(
                    self.config.search_space, obj, warm_start=warm_start
                )
            elif strategy_name == "random":
                RandomSearch().search(
                    self.config.search_space,
                    obj,
                    n_trials=n_trials,
                    warm_start=warm_start,
                )
            elif strategy_name == "sobol":
                SobolSearch().search(
                    self.config.search_space,
                    obj,
                    n_trials=n_trials,
                    seed=self.config.sobol_seed,
                    randomize=self.config.sobol_randomize,
                    warm_start=warm_start,
                )
            elif strategy_name == "bayesian":
                BayesianSearch().search(
                    self.config.search_space,
                    obj,
                    n_trials=n_trials,
                    output_dir=str(output_dir),
                    warm_start=warm_start,
                )
            elif strategy_name == "propulate":
                # Phase 62 RD-5 (Review cycle 1 HIGH, IN-09b): the evaluating rank's
                # real Trial (metrics, flags) goes to all_history; _reduce_trial_outcomes
                # gathers them; returned pairs only back-fill individuals without an
                # evaluation record (RD-6).
                results = PropulateSearch().search(
                    self.config.search_space,
                    obj,
                    n_trials=n_trials,
                    output_dir=str(output_dir),
                    warm_start=warm_start,  # D-09: silently ignored by PropulateSearch
                )
                self._propulate_returned.extend(
                    (tier_name, dict(p), float(sc)) for p, sc in results
                )
            else:
                raise ValueError(f"Unknown search_strategy: {strategy_name!r}")

            # Note: _objective appends Trial objects directly to all_history
            # Prune for warm-start of next tier
            if all_history:
                warm_start = self.prune_candidates(all_history, keep_top_k=3)

        return all_history

    def _reduce_trial_outcomes(
        self, local_history: list[Trial], aborted: str | None = None
    ) -> tuple[list[Trial], list[dict[str, Any]], int, int, int]:
        """Merge trial outcomes of all MPI ranks on rank 0 (Phase 59 NUM-05).

        Parameters
        ----------
        local_history : list[Trial]
            This rank's successful trials (``run()``'s ``all_history``).
        aborted : str or None, optional
            ``repr`` of the exception that aborted this rank's search before
            the reduction (WR-03), or ``None``.  Every rank's abort reason is
            gathered and broadcast; afterwards ``self._aborted_ranks`` holds
            the global ``[(rank, reason), ...]`` list on every rank so
            ``run()`` raises everywhere instead of leaving ranks blocked.

        Returns
        -------
        tuple[list[Trial], list[dict[str, Any]], int, int, int]
            ``(merged_history, failure_records, n_ok, n_fail, n_hist)``.
            On rank 0 (or single-process) ``merged_history`` holds the
            finite-score trials of every rank and ``failure_records`` every
            rank's failures (each tagged with ``rank``) plus any non-finite
            trials dropped by ``_merge_trial_histories``.  On other ranks
            ``merged_history`` is empty and ``failure_records`` holds only the
            local failures.  ``n_ok`` (successful objective evaluations),
            ``n_fail`` (len of rank 0's failure records) and ``n_hist`` (len of
            rank 0's merged history) are global and identical on every rank.

        Notes
        -----
        **Collective-order contract:** with a communicator, every rank calls
        exactly ``comm.gather(payload, root=0)`` then ``comm.bcast(summary,
        root=0)`` — same collectives, same order, no early return or
        rank-dependent branch before them.  ``_objective`` runs on every rank
        under Propulate and under mpirun-launched grid / random / sobol /
        bayesian searches; a divergent call order would deadlock the job.

        **Transport:** the payload carries ``Trial.model_dump()`` dicts (plain,
        pickle-safe data), never ``Trial`` objects.

        **Propulate contract (Phase 62 RD-5..RD-7):** every rank contributes
        the real trials it evaluated (metrics, tier, flags), exactly like the
        other strategies; each individual is evaluated on one rank, so the
        merge has no duplicates.  On rank 0 (and single-process) the merged
        history is then extended by ``_propulate_placeholders`` for the pairs
        ``PropulateSearch.search`` returned (``self._propulate_returned``)
        that match no merged trial (e.g. checkpoint-restored individuals),
        before the counts are computed, so ``n_hist`` includes them and the
        raise rule is identical on every rank; ``n_ok`` counts only real
        successful evaluations.  This is local to rank 0 and adds no
        collective.

        Without a communicator (no mpi4py, or a world of size 1) the same
        merge runs on local data with no collectives.
        """
        comm = self._comm
        rank = comm.Get_rank() if comm is not None else 0
        payload = {
            "failures": [dict(r, rank=rank) for r in self._failed_trials],
            "history": [t.model_dump() for t in local_history],
            "n_ok": self._n_succeeded,
            "aborted": aborted,
        }
        if comm is None:
            self._aborted_ranks = [] if aborted is None else [(rank, aborted)]
            merged, bad = _merge_trial_histories([payload["history"]], start_rank=rank)
            merged = merged + _propulate_placeholders(merged, self._propulate_returned)
            failures = payload["failures"] + bad
            return merged, failures, self._n_succeeded, len(failures), len(merged)

        gathered = comm.gather(payload, root=0)
        if rank == 0:
            merged, bad = _merge_trial_histories([p["history"] for p in gathered], start_rank=0)
            merged = merged + _propulate_placeholders(merged, self._propulate_returned)
            failures = [rec for p in gathered for rec in p["failures"]] + bad
            aborted_ranks = [
                (r, p["aborted"]) for r, p in enumerate(gathered) if p.get("aborted") is not None
            ]
            summary = (sum(p["n_ok"] for p in gathered), len(failures), len(merged), aborted_ranks)
        else:
            merged = []
            failures = payload["failures"]
            summary = None
        n_ok, n_fail, n_hist, aborted_ranks = comm.bcast(summary, root=0)
        self._aborted_ranks = [tuple(a) for a in aborted_ranks]
        return merged, failures, n_ok, n_fail, n_hist

    def _objective(
        self,
        params: dict,
        tier_dataset: dict,
        tier_name: str,
        history_out: list[Trial],
    ) -> float:
        """Compute the score for a single trial by running stages directly (D-07).

        Parameters
        ----------
        params : dict
            Trial hyperparameters from the search strategy.
        tier_dataset : dict
            Tier-specific dataset slice from ``_tier_dataset``.
        tier_name : str
            Current tier name ("sanity" / "dev" / "full") — stored in Trial.
        history_out : list[Trial]
            Mutable list to append the constructed Trial on success.

        Returns
        -------
        float
            Score from ``MetricsEngine.compute_score``, or ``float("-inf")``
            when the trial raises (D-09, Phase 59 NUM-05).

        Notes
        -----
        **Pitfall 4 — default params merge:** ``merged = {**self._default_params,
        **params}`` ensures all required stage keys are present even when the
        search space covers only a subset.

        **D-09 / NUM-05:** a failed trial is logged with its traceback,
        appended to ``self._failed_trials`` as ``{params, tier, error,
        error_type}`` and returns ``-inf`` (worst for the maximised objective;
        every strategy tolerates it and Propulate's ``loss == inf`` filter
        drops it).  It does NOT append to ``history_out``.  A successful trial
        increments ``self._n_succeeded``.  Only ``Exception`` is caught, so
        ``KeyboardInterrupt`` / ``SystemExit`` still propagate.
        """
        try:
            # Pitfall 4: merge defaults first, trial params override
            merged = {**self._default_params, **params}

            # Phase 57 GT-04/D-07: opt-in multi-seed averaging early-exit.
            # This is the ONLY edit this plan makes to _objective's existing
            # body — when len(seeds) == 1 (the default int-seed case, or the
            # graceful sanity/dev fallback from _resolve_subsample_pair_seeds),
            # execution falls through unchanged into the existing logic below,
            # exactly satisfying D-06's "no changes needed inside _objective
            # for the default path."
            # Phase 59 NUM-03: ``seeds`` is defined whenever subsample_pair mode
            # is active so the sanity branch below can pin seed[0].
            seeds: list[int] = []
            if self._is_subsample_pair_mode():
                seeds = _resolve_subsample_pair_seeds(
                    self.config.transform_spec, tier_name
                )
                if len(seeds) > 1:
                    multiseed_score = self._score_subsample_pair_multiseed(
                        params, merged, tier_name, seeds, history_out
                    )
                    self._n_succeeded += 1
                    return multiseed_score

            # Phase 57 GT-06: carries the sanity-tier isolated scratch DataFactory
            # from site 2 (this block) to site 3 (GT-selection, below) within
            # this single _objective() call.
            scratch_factory = None

            # Phase 30/31 source/target dispatch — Pitfall 7 + CONTEXT D-03/D-10/D-11
            # Phase 31 MODE-02/MODE-03: synthetic mode branches added here
            if self.config.pipeline_mode == "synthetic":
                if self._is_subsample_pair_mode():
                    # Phase 57 GT-06/D-06/T-57-06: subsample_pair mode.
                    if tier_name == "sanity":
                        # Sanity tier ALWAYS synthesizes fresh small geometry,
                        # regardless of the original transform_spec["synthesize"]
                        # value — matching the existing convention that the
                        # sanity tier never touches real data. Uses an isolated
                        # scratch DataFactory, mirroring
                        # _apply_transform_to_dataset's isolation contract —
                        # self._factory is never touched here.
                        # Phase 59 NUM-03: pin the resolved single seed so a
                        # list-valued seed never reaches generate_subsample_pair.
                        sanity_spec = {**self.config.transform_spec, "synthesize": True, "seed": seeds[0]}
                        scratch_factory = DataFactory(self.config)
                        tier_dataset, tier_target = scratch_factory.generate_subsample_pair(
                            None, sanity_spec
                        )
                    else:  # dev / full
                        tier_target = {
                            k: self._factory._synthetic_target[k]
                            for k in tier_dataset
                            if k in self._factory._synthetic_target
                        }
                elif tier_name == "sanity":
                    # D-11: apply transform locally — NOT via self._factory.generate_target()
                    # to avoid overwriting _synthetic_target (Pitfall 3)
                    tier_target = _apply_transform_to_dataset(
                        tier_dataset, self.config.transform_spec, self.config
                    )
                else:  # dev / full
                    # D-10: use pre-computed _synthetic_target, sliced to tier keys (Pitfall 7)
                    tier_target = {
                        k: self._factory._synthetic_target[k]
                        for k in tier_dataset
                        if k in self._factory._synthetic_target
                    }
            else:  # paired mode — existing code preserved
                if tier_name == "sanity":
                    tier_target = tier_dataset  # Pitfall 7(a) — sanity reuses same dataset
                else:
                    tier_target = self._factory.load_target()  # Pitfall 7(b) — dev/full call load_target

            align_result = None
            label_result = None
            stage_input = tier_dataset

            if self.config.run_alignment:
                align_result = AlignmentStage(self.config).run(tier_dataset, tier_target, merged)
                stage_input = align_result.aligned_cloud

            # Phase 59 NUM-04 (D-01): the label-transfer direction follows
            # config.label_source via the shared helper (the same contract
            # EvaluationRunner uses), so paired Kobitski->Shah HPO transfers
            # Shah labels onto the aligned Kobitski instead of the reverse.
            lt_target = None
            if self.config.run_label_transfer:
                lt_source, lt_target = resolve_label_transfer_pair(self.config, stage_input, tier_target)
                # LT-02 (Phase 62): cpd_weighted needs the alignment posterior.
                label_result = LabelTransferStage(self.config).run(
                    lt_source, lt_target, merged, align_result=align_result
                )

            # Argument assembly — mirrors eval_runner._run_single exactly (Phase 30 Pitfall 4 rename)
            source_sorted_keys = sorted(tier_dataset.keys())
            target_sorted_keys = sorted(tier_target.keys())
            warp_path = align_result.warp_path if align_result else []
            transforms: list = []

            # Phase 59 NUM-04: F1 is undefined when the labels flow onto a side
            # without ground truth (paired + label_source="target"). Skip the
            # GT read entirely and zero-fill F1 below so every trial carries the
            # same constant F1 (does not bias ranking).
            f1_reason = f1_unavailable_reason(self.config)
            if f1_reason is not None and not self._f1_unavailable_logged:
                _log.warning("%s (HPO objective)", f1_reason)
                self._f1_unavailable_logged = True

            # GT selection — branches on pipeline_mode (Phase 31 MODE-03, D-08/D-09)
            if f1_reason is not None:
                y_true = None
            elif self.config.pipeline_mode == "synthetic":
                if self._is_subsample_pair_mode():
                    # Phase 57 GT-06: correspondence-based extraction (not
                    # positional) for both tiers — subsample_pair's
                    # source/target views do not share point-for-point
                    # positional correspondence the way rigid/noise
                    # identity-transform pairs do.
                    if tier_name == "sanity":
                        # Same scratch_factory instance created at site 2.
                        y_true = scratch_factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]
                    else:  # dev / full
                        y_true = self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]
                elif tier_name == "sanity":
                    # Sanity toy dataset has labels in pc["label"] (generate_labels contract,
                    # Pitfall 5 from Phase 31 RESEARCH.md). get_synthetic_ground_truth() reads
                    # _source_dataset (the real dataset), not the toy dataset — use label directly.
                    sample_pc = tier_dataset[source_sorted_keys[0]]
                    if sample_pc["label"] is not None:
                        y_true = tier_dataset[source_sorted_keys[-1]]["label"]
                    else:
                        last_pos = tier_dataset[source_sorted_keys[-1]]["pos"]
                        y_true = torch.arange(
                            last_pos.shape[0], dtype=torch.long, device=last_pos.device
                        )
                else:  # dev / full in synthetic mode — D-09
                    y_true = self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]
            else:
                # CR-04: sanity tier generates labels into pc["label"] (via generate_labels),
                # not pc["id"]. get_ground_truth() always reads pc["id"] which is None for
                # synthetic data — causing silent all-zero scores. Detect the right field
                # directly instead of delegating to get_ground_truth().
                sample_pc = tier_dataset[source_sorted_keys[0]]
                gt_key = "id" if sample_pc["id"] is not None else "label"
                y_true = tier_dataset[source_sorted_keys[-1]][gt_key]
                if y_true is None:
                    raise ValueError(
                        f"No ground-truth labels in field '{gt_key}' for sanity tier dataset."
                    )
            # kNN inputs are taken BEFORE the WR-01 truncation below: positions
            # and labels of the same receiver frame (U5-3, Phase 62).
            knn_points, labels_for_knn = self._knn_inputs(
                label_result, lt_target, tier_target[target_sorted_keys[-1]]["pos"]
            )
            if label_result is not None:
                y_pred = labels_for_knn
            elif y_true is not None:
                # F1 placeholder only; it no longer feeds kNN consistency.
                y_pred = torch.zeros_like(y_true)
            else:
                y_pred = labels_for_knn.clone()

            if y_true is None:
                # F1 unavailable: shape-compatible placeholder, zero-filled below.
                y_true = y_pred.clone()

            # WR-01: truncate to min length when source and target have different point counts
            # (heterogeneous paired datasets). compute_f1 validates shape equality strictly.
            if y_true.shape[0] != y_pred.shape[0]:
                min_len = min(y_true.shape[0], y_pred.shape[0])
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            metrics = self._engine.compute_stage_metrics(
                stage_input,
                tier_target,
                warp_path,
                transforms,
                y_true,
                y_pred,
                knn_points,
                labels_for_knn,
                k_neighbours=merged.get("k_neighbours", 10),
            )
            self._require_scorable_frames(metrics)
            if f1_reason is not None:
                metrics = metrics.model_copy(update={"f1_score": 0.0})
                metrics = metrics.model_copy(update={"normalized": self._engine.normalize(metrics)})
            score = self._engine.compute_score(metrics)

            trial_obj = Trial(
                params=dict(params),   # shallow copy — Pitfall 7 from types.py
                score=score,
                metrics=metrics,
                tier=tier_name,
                # 59-REVIEW IN-09b: label-transfer fallbacks reach the record.
                flags=list(label_result.flags) if label_result is not None else [],
            )
            history_out.append(trial_obj)
            self._n_succeeded += 1
            return score

        except Exception as exc:
            # D-09 / Phase 59 NUM-05: visible, counted, worst-scored failure.
            _log.warning("Trial failed (tier=%s, params=%s)", tier_name, params, exc_info=True)
            self._failed_trials.append({
                "params": dict(params),
                "tier": tier_name,
                "error": repr(exc),
                "error_type": type(exc).__name__,
            })
            return float("-inf")

    @staticmethod
    def _knn_inputs(
        label_result,
        lt_target: dict | None,
        fallback_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the positions and labels kNN consistency is scored on.

        U5-3 WR-01 (Phase 62): kNN labels and positions come from the same
        receiver frame; truncation applies to F1 only. Mirrors
        ``EvaluationRunner._run_single``.

        Parameters
        ----------
        label_result : LabelResult or None
            Output of ``LabelTransferStage.run``; ``None`` when label transfer
            is disabled.
        lt_target : dict or None
            Receiver frames handed to ``LabelTransferStage.run`` (keys match
            ``label_result.transferred_labels``); unused when
            ``label_result`` is ``None``.
        fallback_points : torch.Tensor
            Receiver positions of shape ``(N, 3)`` scored when label transfer
            is disabled.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(points, labels)`` of equal length. With a label result:
            the last transferred receiver frame's positions and its
            untruncated transferred labels. Without: ``fallback_points`` and
            zero labels of the same length on the same device.
        """
        if label_result is not None:
            # Label keys are RECEIVER frames per the LabelTransferStage
            # contract; the last key actually present guards against KeyError
            # when |provider| < |receiver| (CR-01).
            key = sorted(label_result.transferred_labels)[-1]
            return lt_target[key]["pos"], label_result.transferred_labels[key]
        return fallback_points, torch.zeros(
            fallback_points.shape[0], dtype=torch.long, device=fallback_points.device
        )

    def _require_scorable_frames(self, metrics: StageMetrics) -> None:
        """Raise when an alignment trial produced no scorable frame (WR-01).

        ``compute_stage_metrics`` sets chamfer/hausdorff to ``+inf`` when no
        frame could be scored (no shared frame keys, or every shared frame
        degenerate, e.g. NaN positions after a CPD divergence).  ``+inf``
        normalises to ``0.0`` for those two components only, so the composite
        score would stay finite and rankable.  Raising here routes the trial
        through ``_objective``'s ``except`` handler: it is recorded in
        ``self._failed_trials`` and scored ``-inf`` (D-05, NUM-05).

        Only enforced when alignment runs: without alignment chamfer/hausdorff
        are not what the search optimises.

        Parameters
        ----------
        metrics : StageMetrics
            Metrics of the trial (or of one seed of a multiseed trial).

        Raises
        ------
        ValueError
            If alignment ran and chamfer or hausdorff is non-finite.
        """
        if not self.config.run_alignment:
            return
        if not (
            math.isfinite(metrics.chamfer_distance)
            and math.isfinite(metrics.hausdorff_distance)
        ):
            raise ValueError(
                "no scorable frame: chamfer/hausdorff are non-finite "
                f"({metrics.chamfer_distance}, {metrics.hausdorff_distance}); "
                f"coverage flags: {metrics.coverage_flags}"
            )

    def _score_subsample_pair_multiseed(
        self,
        params: dict,
        merged: dict,
        tier_name: str,
        seeds: list[int],
        history_out: list[Trial],
    ) -> float:
        """Score one trial by averaging across N independent subsample pairs (D-07).

        Phase 57 GT-04/D-07: called from ``_objective``'s early-exit branch
        only when ``_resolve_subsample_pair_seeds`` returns more than one
        seed (opt-in multi-seed averaging, gated to the ``full`` tier). Each
        seed's subsample pair is generated and scored in full isolation via a
        FRESH ``DataFactory`` instance — never ``self._factory`` and never a
        single scratch instance reused across seeds — because
        ``_correspondence_idx``/``_synthetic_target``/``_source_dataset`` are
        single-slot instance attributes that would otherwise be clobbered
        between seeds within the same trial (T-57-11).

        Each seed is scored on the aligned per-frame dict (``stage_input``,
        i.e. ``AlignmentStage``'s ``aligned_cloud`` when alignment runs)
        against the per-frame ``target_view`` dict through
        ``MetricsEngine.compute_stage_metrics``, exactly like ``_objective``
        (Phase 59 NUM-02, U1-1/U5-1). The label transfer follows the shared
        ``resolve_label_transfer_pair`` direction contract.

        Parameters
        ----------
        params : dict
            Trial hyperparameters from the search strategy (unmerged —
            stored as-is on the resulting ``Trial``, mirroring
            ``_objective``).
        merged : dict
            ``{**self._default_params, **params}`` — already merged by the
            caller (``_objective``); passed through to
            ``AlignmentStage``/``LabelTransferStage``.
        tier_name : str
            Current tier name — always ``"full"`` in practice (the only tier
            this method is ever invoked from, per
            ``_resolve_subsample_pair_seeds``'s gating), stored on the
            resulting ``Trial``.
        seeds : list[int]
            Resolved seed list (``len(seeds) > 1`` — guaranteed by the
            caller).
        history_out : list[Trial]
            Mutable list to append the single averaged ``Trial`` to on
            success.

        Returns
        -------
        float
            The arithmetic mean of the N per-seed
            ``MetricsEngine.compute_score`` results.

        Notes
        -----
        Exceptions are NOT caught here — they propagate to the caller's
        (``_objective``'s) ``except Exception`` handler, so a failed multiseed
        trial is recorded as a failure and scores ``-inf`` exactly like any
        other failed trial (D-09 consistency, Phase 59 NUM-05).
        """
        # Resolved ONCE, reused across all N seeds — load_real() is
        # cached/idempotent per DataFactory's existing contract, so this is
        # not redundant I/O.
        base = (
            None
            if self.config.transform_spec.get("synthesize", False)
            else self._factory.load_real()
        )

        scores: list[float] = []
        last_metrics: StageMetrics | None = None
        # 59-REVIEW IN-09b: every seed's label-transfer flags, seed-prefixed.
        trial_flags: list[str] = []

        for s in seeds:
            per_seed_spec = {**self.config.transform_spec, "seed": s}
            # A FRESH DataFactory per seed — never reused across iterations
            # and never self._factory (T-57-11).
            scratch_factory = DataFactory(self.config)
            source_view, target_view = scratch_factory.generate_subsample_pair(
                base, per_seed_spec
            )

            align_result = None
            label_result = None
            stage_input = source_view

            if self.config.run_alignment:
                align_result = AlignmentStage(self.config).run(
                    source_view, target_view, merged
                )
                stage_input = align_result.aligned_cloud

            lt_target = None
            if self.config.run_label_transfer:
                # Phase 59 NUM-04: same direction contract as _objective. EvalConfig
                # rejects label_source="target" in synthetic mode, so this always
                # resolves to (stage_input, target_view) here.
                lt_source, lt_target = resolve_label_transfer_pair(self.config, stage_input, target_view)
                # LT-02 (Phase 62): cpd_weighted needs the alignment posterior.
                label_result = LabelTransferStage(self.config).run(
                    lt_source, lt_target, merged, align_result=align_result
                )
                trial_flags.extend(f"seed {s}: {flag}" for flag in label_result.flags)

            source_sorted_keys = sorted(source_view.keys())
            target_sorted_keys = sorted(target_view.keys())
            warp_path = align_result.warp_path if align_result else []

            y_true = scratch_factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]
            # U5-3 (Phase 62): kNN inputs from the shared helper, before the
            # WR-01 truncation (which applies to the F1 pair only).
            knn_points, labels_for_knn = self._knn_inputs(
                label_result, lt_target, target_view[target_sorted_keys[-1]]["pos"]
            )
            if label_result is not None:
                y_pred = labels_for_knn
            else:
                # F1 placeholder only; it no longer feeds kNN consistency.
                y_pred = torch.zeros_like(y_true)

            # WR-01: truncate to min length when source and target have
            # different point counts.
            if y_true.shape[0] != y_pred.shape[0]:
                min_len = min(y_true.shape[0], y_pred.shape[0])
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            # Phase 59 NUM-02 (U1-1/U5-1): score the ALIGNED per-frame dict
            # against the target dict. Passing raw tensors of the unaligned
            # source view made chamfer/hausdorff identical for aligned and
            # misaligned pairs (0.0 at 6c1c37f, inf after the NUM-05 guard).
            metrics = self._engine.compute_stage_metrics(
                stage_input,
                target_view,
                warp_path,
                [],
                y_true,
                y_pred,
                knn_points,
                labels_for_knn,
                k_neighbours=merged.get("k_neighbours", 10),
            )
            self._require_scorable_frames(metrics)
            scores.append(self._engine.compute_score(metrics))
            # Kept as a representative sample for search_history.json logging
            # only — NOT the source of the averaged score.
            last_metrics = metrics

        avg_score = sum(scores) / len(scores)
        trial_obj = Trial(
            params=dict(params),
            score=avg_score,
            metrics=last_metrics,
            tier=tier_name,
            flags=trial_flags,
        )
        history_out.append(trial_obj)
        return avg_score

    def _tier_dataset(self, tier: str) -> dict:
        """Return the dataset slice appropriate for the given tier.

        Parameters
        ----------
        tier : str
            One of "sanity", "dev", "full".

        Returns
        -------
        dict[int, zRegPointCloud]
            Dataset for the tier.

        Notes
        -----
        **Pitfall 5:** Sanity tier calls ``generate_labels(generate_trajectory(...))``
        so ``pc["id"]`` is populated and ``get_ground_truth()`` returns valid tensors.

        **D-13:** Sanity tier consults ``self.config.label_generation`` when
        set (forwarding ``n_labels``/``label_specs``/``mode``/``seed``),
        falling back to the hardcoded ``n_labels=4, seed=42`` behaviour
        otherwise — every existing config without ``label_generation`` set
        is unaffected.
        """
        if tier == "sanity":
            # Small labelled synthetic dataset — MUST include labels (Pitfall 5)
            traj = generate_trajectory(n_points=50, n_frames=3, seed=42)
            if self.config.label_generation is not None:
                lg = self.config.label_generation
                return generate_labels(
                    traj, n_labels=lg.n_labels, label_specs=lg.label_specs, mode=lg.mode, seed=lg.seed
                )
            return generate_labels(traj, n_labels=4, seed=42)
        elif tier == "dev":
            # Phase 57 GT-06: subsample_pair mode — _subsample_source_view is
            # only non-None after run()'s pre-populate step (T-57-07). Phase 59
            # NUM-03: never return None — raise naming the missing step, so a
            # skipped pre-populate cannot surface as a swallowed TypeError.
            if self._is_subsample_pair_mode():
                return self._prepopulated_subsample_view(tier)
            # Use real data if available, otherwise fall back to synthetic
            data_path = Path(self.config.data_path) if self.config.data_path else None
            if data_path and data_path.exists():
                return self._factory.load_real()
            return self._factory.generate_synthetic()
        else:
            # Phase 57 GT-06 / Phase 59 NUM-03: see "dev" branch comment above.
            if self._is_subsample_pair_mode():
                return self._prepopulated_subsample_view(tier)
            # full tier — full real dataset
            return self._factory.load_real()

    def _prepopulated_subsample_view(self, tier: str) -> dict:
        """Return the pre-populated subsample_pair source view or raise.

        Parameters
        ----------
        tier : str
            Tier name, used only in the error message.

        Returns
        -------
        dict[int, zRegPointCloud]
            ``self._factory._subsample_source_view``.

        Raises
        ------
        RuntimeError
            If ``run()``'s pre-populate step has not produced the view yet
            (Phase 59 NUM-03: previously this returned ``None``).
        """
        view = self._factory._subsample_source_view
        if view is None:
            raise RuntimeError(
                "subsample_pair source view not pre-populated: run() must call "
                f"generate_subsample_pair before tier '{tier}' (pre-populate step)"
            )
        return view

    def _is_subsample_pair_mode(self) -> bool:
        """Return True when the current config targets subsample_pair generation.

        Phase 57 GT-06 / D-06: the default single-fixed-seed subsample_pair
        path is dispatched on this predicate at all three
        ``pipeline_mode == "synthetic"`` call sites in this file (``run()``'s
        pre-populate step, ``_tier_dataset()``, ``_objective()``).

        Returns
        -------
        bool
            ``True`` iff ``pipeline_mode == "synthetic"`` AND
            ``transform_spec is not None`` AND
            ``transform_spec.get("type") == "subsample_pair"``.
        """
        return (
            self.config.pipeline_mode == "synthetic"
            and self.config.transform_spec is not None
            and self.config.transform_spec.get("type") == "subsample_pair"
        )

    @staticmethod
    def prune_candidates(history: list[Trial], keep_top_k: int) -> list[dict]:
        """Return param dicts for the top-k trials by score descending.

        Parameters
        ----------
        history : list[Trial]
            All trials from the current (or previous) tier.
        keep_top_k : int
            Number of top candidates to return.

        Returns
        -------
        list[dict]
            Param dicts for the ``keep_top_k`` highest-scoring trials.
        """
        sorted_trials = sorted(history, key=lambda t: t.score, reverse=True)
        return [t.params for t in sorted_trials[:keep_top_k]]

    def save_best_params(
        self, result: SearchResult, output_dir: str | Path
    ) -> None:
        """Write ``best_params.json``, ``search_history.json`` and ``failed_trials.json``.

        Parameters
        ----------
        result : SearchResult
            Completed search result with best params, the merged trial
            history and the failure records.
        output_dir : str or Path
            Directory where JSON files are written.

        Notes
        -----
        ``best_params.json`` stays a flat dict and ``search_history.json`` a
        list of ``Trial`` dicts (downstream scripts parse both); the history
        holds the finite-score successful trials of all ranks.  Failures
        (Phase 59 NUM-05) go to ``failed_trials.json`` as a list of records.

        Uses ``model_dump()`` + ``json.dump()``.  The JSON shortcut raises
        ``PydanticSerializationError`` for ``torch.Tensor`` fields and must not
        be used (RESEARCH Pitfall 2 / D-12).
        """
        out = Path(output_dir)

        # best_params.json — flat dict of best hyperparameters
        with open(out / "best_params.json", "w") as f:
            json.dump(result.best_params, f, indent=2)

        # search_history.json — list of Trial dicts (model_dump() only)
        history_dicts = [t.model_dump() for t in result.history]
        with open(out / "search_history.json", "w") as f:
            json.dump(history_dicts, f, indent=2)

        # failed_trials.json — failure records of every rank (NUM-05)
        with open(out / "failed_trials.json", "w") as f:
            json.dump(result.failed_trials, f, indent=2)

    def _is_rank_zero(self) -> bool:
        """Return True if running single-process or if this is MPI rank 0.

        An injected / resolved communicator (``self._comm``) takes precedence
        over ``MPI.COMM_WORLD``.
        """
        if self._comm is not None:
            return self._comm.Get_rank() == 0
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank() == 0
        except ImportError:
            return True

    def _detect_backend(self) -> str:
        """Resolve search_strategy='auto' to 'propulate' or 'bayesian' (D-04).

        Detection order (strict, D-04):
          1. mpi4py importable AND MPI.COMM_WORLD.Get_size() > 1  → "propulate"
          2. SLURM_JOB_ID set in os.environ                       → "propulate"
          3. Fallback                                              → "bayesian"

        Parameters
        ----------
        (none — instance method for logging access)

        Returns
        -------
        str
            One of ``"propulate"`` or ``"bayesian"``.

        Notes
        -----
        **D-05:** mpi4py ImportError is silently caught — it means the user is
        not in an MPI environment.  No log warning is emitted.

        **Pitfall 6:** mpi4py initialisation errors (e.g., MPI not available on
        a SLURM login node) are caught by the broad ``except Exception`` and
        logged at DEBUG level to avoid masking real failures.
        """
        # D-04 step 1: check mpi4py world size
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.Get_size() > 1:
                _log.info("Auto-detected backend: propulate (world_size > 1)")
                return "propulate"
        except ImportError:
            pass  # D-05: silent — not an MPI environment
        except Exception as exc:  # Pitfall 6: MPI init failures are not always ImportError
            _log.debug("MPI init probe failed: %s", exc)

        # D-04 step 2: check SLURM_JOB_ID (cluster intent)
        if "SLURM_JOB_ID" in os.environ:
            _log.info("Auto-detected backend: propulate (SLURM_JOB_ID set)")
            return "propulate"

        # D-04 step 3: fallback
        _log.info("Auto-detected backend: bayesian (fallback)")
        return "bayesian"
