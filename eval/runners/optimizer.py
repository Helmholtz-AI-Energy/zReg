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
- **D-09** If a trial raises an exception, ``_objective`` returns ``0.0`` and logs
  a warning.  The search continues.
- **D-10** Study name fixed as ``"zreg-hpo"``; re-running with the same
  ``output_dir`` resumes existing study (``load_if_exists=True``).
- **D-11** SQLite at ``{output_dir}/optuna.db``.
- **D-12** ``GridSearch`` and ``RandomSearch`` are stateless; ``save_best_params``
  writes JSON at the end of ``run()``.
- **D-04** ``_detect_backend()`` resolves ``search_strategy="auto"`` → ``"propulate"``
  (MPI world_size > 1 or SLURM_JOB_ID set) or ``"bayesian"`` (fallback). EXT-03.
- **D-11** Propulate dispatch branch constructs ``Trial`` objects from the
  ``(params, score)`` pairs returned by ``PropulateSearch.search()`` because
  Propulate's loss closure does not append to ``history_out``.

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
``DataFactory.get_ground_truth()`` returns valid tensors.

**JSON serialisation:** Always ``model_dump()`` + ``json.dump()``.  The JSON
shortcut raises for ``torch.Tensor`` fields (Pitfall 2 from eval/types.py).

**Phase 30 source/target dispatch (Pitfall 7):** sanity tier reuses
``tier_dataset`` as both source and target (D-03 smoke-test parity); dev/full
tiers call ``self._factory.load_target()`` to obtain the target trajectory from
``config.target_data_path``.
"""

import json
import logging
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
from eval.search_strategies import BayesianSearch, GridSearch, PropulateSearch, RandomSearch, SobolSearch
from eval.stages import AlignmentStage, LabelTransferStage
from eval.types import SearchResult, StageMetrics, Trial

__all__ = ["HyperparamOptimizer"]

_log = logging.getLogger(__name__)

# Module-level tier trial count constants (RESEARCH Q3 resolved)
SANITY_N_TRIALS: int = 5
DEV_N_TRIALS: int = 20


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

    def __init__(self, config: EvalConfig) -> None:
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
            ``history`` (all ``Trial`` objects from all tiers), and ``tier``
            (the ceiling tier from ``config.tier``).

        Notes
        -----
        **T-22-02:** ``output_dir`` is resolved via ``Path(...).resolve()`` as
        the first line to prevent path traversal attacks.
        """
        # T-22-02: resolve output_dir to prevent path traversal
        output_dir = Path(self.config.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

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
            real_source = self._factory.load_real()
            self._factory.generate_target(real_source, self.config.transform_spec)

        # Tier execution sequence — stop at config.tier ceiling (D-01)
        TIER_SEQUENCE = ["sanity", "dev", "full"]
        tiers_to_run = TIER_SEQUENCE[: TIER_SEQUENCE.index(self.config.tier) + 1]

        all_history: list[Trial] = []
        warm_start: list[dict] | None = None

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
                # CR-01 fix: use a throwaway list so _objective does not append to
                # all_history — propulate runs evaluations internally and returns the
                # survivors via results; the explicit loop below is the single writer.
                _propulate_sink: list[Trial] = []
                propulate_obj = make_objective(tier_dataset, tier_name, _propulate_sink)
                results = PropulateSearch().search(
                    self.config.search_space,
                    propulate_obj,
                    n_trials=n_trials,
                    output_dir=str(output_dir),
                    warm_start=warm_start,  # D-09: silently ignored by PropulateSearch
                )
                # D-11: PropulateSearch returns (params, score) pairs; construct Trials
                # here with zero-filled StageMetrics placeholder (Open Question 2 / option a).
                minimal_metrics = StageMetrics(
                    chamfer_distance=0.0,
                    hausdorff_distance=0.0,
                    path_smoothness=0.0,
                    temporal_stability=0.0,
                    f1_score=0.0,
                    knn_consistency=0.0,
                )
                for params, score in results:
                    all_history.append(Trial(
                        params=dict(params),
                        score=score,
                        metrics=minimal_metrics,
                        tier=tier_name,
                    ))
            else:
                raise ValueError(f"Unknown search_strategy: {strategy_name!r}")

            # Note: _objective appends Trial objects directly to all_history
            # Prune for warm-start of next tier
            if all_history:
                warm_start = self.prune_candidates(all_history, keep_top_k=3)

        # Construct SearchResult from full history
        if not all_history:
            result = SearchResult(
                best_params={},
                best_score=0.0,
                history=[],
                tier=self.config.tier,
            )
        else:
            best_trial = max(all_history, key=lambda t: t.score)
            result = SearchResult(
                best_params=dict(best_trial.params),
                best_score=best_trial.score,
                history=list(all_history),
                tier=self.config.tier,
            )

        self.save_best_params(result, output_dir)
        return result

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
            Score from ``MetricsEngine.compute_score``, or ``0.0`` on any
            exception (D-09).

        Notes
        -----
        **Pitfall 4 — default params merge:** ``merged = {**self._default_params,
        **params}`` ensures all required stage keys are present even when the
        search space covers only a subset.

        **D-09:** Failed trials return ``0.0`` and do NOT append to
        ``history_out``.
        """
        try:
            # Pitfall 4: merge defaults first, trial params override
            merged = {**self._default_params, **params}

            # Phase 30/31 source/target dispatch — Pitfall 7 + CONTEXT D-03/D-10/D-11
            # Phase 31 MODE-02/MODE-03: synthetic mode branches added here
            if self.config.pipeline_mode == "synthetic":
                if tier_name == "sanity":
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

            if self.config.run_label_transfer:
                label_result = LabelTransferStage(self.config).run(stage_input, tier_target, merged)

            # Argument assembly — mirrors eval_runner._run_single exactly (Phase 30 Pitfall 4 rename)
            # Pitfall 4 — source_pos/target_pos avoid shadowing tier_dataset/tier_target parameters
            source_sorted_keys = sorted(tier_dataset.keys())
            target_sorted_keys = sorted(tier_target.keys())
            source_pos = tier_dataset[source_sorted_keys[0]]["pos"]
            target_pos = tier_target[target_sorted_keys[-1]]["pos"]
            warp_path = align_result.warp_path if align_result else []
            transforms: list = []

            # GT selection — branches on pipeline_mode (Phase 31 MODE-03, D-08/D-09)
            if self.config.pipeline_mode == "synthetic":
                if tier_name == "sanity":
                    # Sanity toy dataset has labels in pc["label"] (generate_labels contract,
                    # Pitfall 5 from Phase 31 RESEARCH.md). get_synthetic_ground_truth() reads
                    # _source_dataset (the real dataset), not the toy dataset — use label directly.
                    sample_pc = tier_dataset[source_sorted_keys[0]]
                    if sample_pc["label"] is not None:
                        y_true = tier_dataset[source_sorted_keys[-1]]["label"]
                    else:
                        n = tier_dataset[source_sorted_keys[-1]]["pos"].shape[0]
                        y_true = torch.arange(n, dtype=torch.long)
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
            if label_result is not None:
                # Label keys are TARGET frames per Plan 30-01 LabelTransferStage contract
                # Use last key actually present in transferred_labels (= last paired target
                # frame) — guards against KeyError when |source| < |target| (CR-01).
                transferred_keys = sorted(label_result.transferred_labels.keys())
                y_pred = label_result.transferred_labels[transferred_keys[-1]]
            else:
                y_pred = torch.zeros_like(y_true)

            # WR-01: truncate to min length when source and target have different point counts
            # (heterogeneous paired datasets). compute_f1 validates shape equality strictly.
            if y_true.shape[0] != y_pred.shape[0]:
                min_len = min(y_true.shape[0], y_pred.shape[0])
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]

            metrics = self._engine.compute_stage_metrics(
                source_pos,
                target_pos,
                warp_path,
                transforms,
                y_true,
                y_pred,
                target_pos,
                y_pred,
                k_neighbours=merged.get("k_neighbours", 10),
            )
            score = self._engine.compute_score(metrics)

            trial_obj = Trial(
                params=dict(params),   # shallow copy — Pitfall 7 from types.py
                score=score,
                metrics=metrics,
                tier=tier_name,
            )
            history_out.append(trial_obj)
            return score

        except Exception as exc:
            _log.warning("Trial failed: %s", exc)
            return 0.0  # D-09: failed trial returns 0.0; search continues; no Trial appended

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
        """
        if tier == "sanity":
            # Small labelled synthetic dataset — MUST include labels (Pitfall 5)
            traj = generate_trajectory(n_points=50, n_frames=3, seed=42)
            return generate_labels(traj, n_labels=4, seed=42)
        elif tier == "dev":
            # Use real data if available, otherwise fall back to synthetic
            data_path = Path(self.config.data_path) if self.config.data_path else None
            if data_path and data_path.exists():
                return self._factory.load_real()
            return self._factory.generate_synthetic()
        else:
            # full tier — full real dataset
            return self._factory.load_real()

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
        """Write ``best_params.json`` and ``search_history.json`` to ``output_dir``.

        Parameters
        ----------
        result : SearchResult
            Completed search result with best params and full trial history.
        output_dir : str or Path
            Directory where JSON files are written.

        Notes
        -----
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
