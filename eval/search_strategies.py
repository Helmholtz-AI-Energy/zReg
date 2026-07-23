"""Search strategy classes for the zReg hyperparameter optimisation framework.

Key design decisions implemented here:

- **D-05** Unified search space format — list of values per param key across all three
  strategies.  GridSearch exhausts all Cartesian products; RandomSearch samples from
  the lists; BayesianSearch treats each list as categorical choices for Optuna TPE.
- **D-12** GridSearch and RandomSearch are stateless — all trials computed in memory,
  no SQLite.  BayesianSearch creates / resumes an Optuna study with SQLite storage.
- **D-10** BayesianSearch study name is fixed as ``"zreg-hpo"`` per ``output_dir``.
  Re-running with the same ``output_dir`` resumes the existing study
  (``load_if_exists=True``).
- **FRAME-10** TPE sampler with ``n_startup_trials >= 2 * N_params`` (clamped for
  sanity tier to avoid exhausting the entire trial budget in the random startup phase;
  Pitfall 3 from RESEARCH).
- **D-12 (PropulateSearch)** ``propulate`` library import is lazy inside
  ``PropulateSearch.search()`` — this module remains importable without propulate
  installed.  Raises ``ImportError`` with a friendly ``pip install zreg[propulate]``
  message when propulate is absent.
- **D-13 (corrected)** Propulate optional extra pin is ``propulate>=1.0,<2`` (not
  ``>=0.4`` as in the original CONTEXT — see Pitfall 3 in RESEARCH: no 0.4 release
  exists on PyPI).  EXT-03.

Notes
-----
**Import order — macOS-ARM SIGABRT (Pitfall 7):**
``zreg.*`` imports MUST precede ``torch``, and ``torch`` MUST precede ``optuna``.
Interleaving these on macOS ARM (Apple Silicon) triggers a libomp SIGABRT.
``propulate`` is imported lazily inside ``PropulateSearch.search()`` — after all
module-level imports are already done, so the order constraint is naturally satisfied.

**Anti-pattern:** Do NOT call ``EvaluationRunner`` inside ``objective_fn`` (D-07).
Keep each trial lightweight — instantiate stages directly.
"""

import itertools
import logging
import math
import random
from pathlib import Path
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35.
from zreg.dataset import zRegPointCloud

import torch

import optuna

from scipy.stats import qmc

from eval.types import Trial, SearchResult

__all__ = ["GridSearch", "RandomSearch", "BayesianSearch", "PropulateSearch", "SobolSearch"]

_log = logging.getLogger(__name__)

SOBOL_MIN_TRIALS: int = 8


class GridSearch:
    """Exhaustive grid search over all Cartesian products of the search space.

    Stateless (D-12) — no SQLite, no Optuna study.  All results held in memory.

    Warm-start params are prepended to the candidate list so they are evaluated
    first (D-04); duplicates are skipped via a ``seen`` set.
    """

    def search(
        self,
        search_space: dict[str, list],
        objective_fn,
        warm_start: list[dict] | None = None,
    ) -> list[tuple[dict, float]]:
        """Enumerate all combinations and evaluate each with ``objective_fn``.

        Parameters
        ----------
        search_space:
            Dict mapping param name → list of candidate values.
        objective_fn:
            Callable accepting a params dict and returning a float score.
        warm_start:
            Optional list of param dicts to evaluate first (D-04).

        Returns
        -------
        list[tuple[dict, float]]
            Each element is (params_dict, score) for every evaluated combination.
        """
        keys = list(search_space.keys())
        all_combos = [
            dict(zip(keys, combo))
            for combo in itertools.product(*search_space.values())
        ]

        seen: set[tuple] = set()
        candidates: list[dict] = []

        # Prepend warm-start seeds so they are evaluated first (D-04)
        if warm_start:
            for p in warm_start:
                key = tuple(sorted(p.items()))
                if key not in seen:
                    candidates.append(p)
                    seen.add(key)

        for combo in all_combos:
            key = tuple(sorted(combo.items()))
            if key not in seen:
                candidates.append(combo)
                seen.add(key)

        results: list[tuple[dict, float]] = []
        for params in candidates:
            score = objective_fn(params)
            results.append((params, score))

        return results


class RandomSearch:
    """Random sampling search over the search space.

    Stateless (D-12) — no SQLite, no Optuna study.  All results held in memory.

    Warm-start params are evaluated first, then ``n_trials`` random samples are
    drawn from the search space lists (duplicates vs warm-start are skipped).
    """

    def search(
        self,
        search_space: dict[str, list],
        objective_fn,
        n_trials: int,
        warm_start: list[dict] | None = None,
    ) -> list[tuple[dict, float]]:
        """Sample randomly from search space and evaluate each candidate.

        Parameters
        ----------
        search_space:
            Dict mapping param name → list of candidate values.
        objective_fn:
            Callable accepting a params dict and returning a float score.
        n_trials:
            Number of random candidates to draw (in addition to warm-start).
        warm_start:
            Optional list of param dicts to evaluate first (D-04).

        Returns
        -------
        list[tuple[dict, float]]
            Each element is (params_dict, score) for every evaluated candidate.
        """
        seen: set[tuple] = set()
        candidates: list[dict] = []

        # Prepend warm-start seeds (D-04)
        if warm_start:
            for p in warm_start:
                key = tuple(sorted(p.items()))
                if key not in seen:
                    candidates.append(p)
                    seen.add(key)

        # Sample n_trials random combinations
        for _ in range(n_trials):
            params = {k: random.choice(v) for k, v in search_space.items()}
            key = tuple(sorted(params.items()))
            # Allow duplicate random samples (they may differ in non-warm-start keys)
            candidates.append(params)

        results: list[tuple[dict, float]] = []
        for params in candidates:
            score = objective_fn(params)
            results.append((params, score))

        return results


class SobolSearch:
    """Quasi-random Sobol sequence search over the search space.

    Stateless (D-12) — no SQLite, no Optuna study.  All results held in memory.

    Warm-start params are evaluated first, then ``n_trials`` Sobol-sampled
    candidates are generated.  When ``n_trials < SOBOL_MIN_TRIALS`` (= 8),
    falls back to ``RandomSearch`` using ``seed`` for reproducibility (D-10).
    The sanity tier (``SANITY_N_TRIALS = 5``) always triggers the fallback.

    ``seed`` and ``randomize`` are call-args (not constructor args), consistent
    with how ``output_dir`` is a call-arg in ``BayesianSearch.search()`` (D-05).

    **Import order note:** ``SobolSearch`` uses only ``scipy.stats.qmc`` — no
    torch or optuna dependency.  The macOS-ARM SIGABRT import-order constraint
    (``zreg.*`` → ``torch`` → ``optuna``) does NOT apply to ``scipy.stats.qmc``;
    it can be imported at any position after the existing block without risk.
    """

    def search(
        self,
        search_space: dict[str, list],
        objective_fn,
        n_trials: int,
        seed: int = 42,
        randomize: bool = True,
        warm_start: list[dict] | None = None,
    ) -> list[tuple[dict, float]]:
        """Sample using a Sobol quasi-random sequence and evaluate each candidate.

        Parameters
        ----------
        search_space:
            Dict mapping param name → list of candidate values.
        objective_fn:
            Callable accepting a params dict and returning a float score.
        n_trials:
            Number of Sobol-sampled candidates to generate.  When below
            ``SOBOL_MIN_TRIALS`` (= 8), falls back to ``RandomSearch`` (D-10).
        seed:
            Seed for the scrambled Owen sequence (``randomize=True``); silently
            ignored for the classical Van der Corput sequence (``randomize=False``,
            D-04).  Default 42, consistent with ``BayesianSearch``
            (``TPESampler(seed=42)``).
        randomize:
            When ``True`` (default), uses scrambled Owen sequence (better
            uniformity, reproducible via ``seed``).  When ``False``, uses
            classical Van der Corput sequence and ``seed`` is silently ignored
            (D-04).
        warm_start:
            Optional list of param dicts to evaluate first (D-07).

        Returns
        -------
        list[tuple[dict, float]]
            Each element is (params_dict, score) for every evaluated candidate.
        """
        # D-08: empty search space → return [] silently
        if not search_space:
            return []

        # D-10: small budget fallback to RandomSearch
        if n_trials < SOBOL_MIN_TRIALS:
            _log.debug(
                "SobolSearch: n_trials=%d < 8, using RandomSearch fallback", n_trials
            )
            random.seed(seed)
            return RandomSearch().search(
                search_space, objective_fn, n_trials=n_trials, warm_start=warm_start
            )

        # Warm-start prepend (D-07) — copy deduplication loop from RandomSearch
        seen: set[tuple] = set()
        candidates: list[dict] = []

        if warm_start:
            for p in warm_start:
                key = tuple(sorted(p.items()))
                if key not in seen:
                    candidates.append(p)
                    seen.add(key)

        # Sobol sampling
        keys = list(search_space.keys())
        choices = [search_space[k] for k in keys]
        n_dims = len(search_space)
        sampler = qmc.Sobol(d=n_dims, scramble=randomize, seed=seed)
        samples = sampler.random(n_trials)

        for row in samples:
            params = {
                keys[i]: choices[i][
                    min(int(math.floor(row[i] * len(choices[i]))), len(choices[i]) - 1)
                ]
                for i in range(n_dims)
            }
            candidates.append(params)

        # Results accumulation
        results: list[tuple[dict, float]] = []
        for params in candidates:
            score = objective_fn(params)
            results.append((params, score))

        return results


class BayesianSearch:
    """Bayesian hyperparameter search using Optuna TPE sampler with SQLite storage.

    Creates or resumes an Optuna study (D-10: ``load_if_exists=True``) at
    ``{output_dir}/optuna.db``.  All params are treated as categorical choices
    (FRAME-10 / D-05).

    TPE startup trials are clamped for sanity tier to avoid exhausting the entire
    trial budget in the random phase (Pitfall 3 from RESEARCH):
    ``n_startup = min(max(10, 2 * n_params), n_trials - 1)``.

    Warm-start params are seeded via ``study.enqueue_trial(skip_if_exists=True)``
    (D-04; Pitfall 6 — ``skip_if_exists`` REQUIRED to avoid duplicates on re-run).
    """

    def search(
        self,
        search_space: dict[str, list],
        objective_fn,
        n_trials: int,
        output_dir: str | Path,
        warm_start: list[dict] | None = None,
    ) -> list[tuple[dict, float]]:
        """Run Bayesian optimisation and return all completed trial results.

        Parameters
        ----------
        search_space:
            Dict mapping param name → list of candidate values.
        objective_fn:
            Callable accepting a params dict and returning a float score.
        n_trials:
            Total number of trials (including warm-start).
        output_dir:
            Directory where ``optuna.db`` will be written.
        warm_start:
            Optional list of param dicts to enqueue as fixed warm-start trials.

        Returns
        -------
        list[tuple[dict, float]]
            Each element is (params_dict, score) for every completed trial.
        """
        n_params = len(search_space)
        n_startup = max(10, 2 * n_params)  # FRAME-10: n_startup_trials >= 2 * N_params

        # Clamp n_startup for small trial budgets (Pitfall 3 — sanity tier)
        if n_trials <= 1:
            n_startup = 1
        else:
            n_startup = min(n_startup, n_trials - 1)

        sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=42)
        study = optuna.create_study(
            study_name="zreg-hpo",
            storage=f"sqlite:///{Path(output_dir) / 'optuna.db'}",
            load_if_exists=True,   # D-10: resume on re-run
            direction="maximize",  # compute_score is higher-is-better
            sampler=sampler,
        )

        # Warm-start: seed pruned top-k from previous tier (D-04, Pitfall 6)
        if warm_start:
            for p in warm_start:
                study.enqueue_trial(p, skip_if_exists=True)

        def _optuna_obj(trial: optuna.Trial) -> float:
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in search_space.items()
            }
            return objective_fn(params)

        study.optimize(_optuna_obj, n_trials=n_trials)

        # Guard (Pitfall 1): t.value is None for enqueued-but-not-evaluated trials
        return [(t.params, t.value) for t in study.trials if t.value is not None]


class PropulateSearch:
    """MPI-parallel evolutionary search via the propulate library.

    Mirrors BayesianSearch.search() signature.  Internally runs a single-island
    ``Propulator`` with ``island_comm=MPI.COMM_WORLD`` and gathers all evaluated
    ``Individual`` objects on rank 0.

    Approximate trial count: total evaluations = generations × world_size, so
    actual count may exceed ``n_trials`` by up to (world_size - 1).  Use
    ``mpirun -n N`` to control parallelism.

    ``warm_start`` is accepted but silently ignored — Propulate's evolutionary
    model manages its own population and does not accept warm-start seeds in
    the same way (D-09).

    Checkpoints are written to ``checkpoint_path=Path(output_dir)`` (same
    directory passed as ``output_dir``).

    .. warning::
        Do not load checkpoints from untrusted sources — Propulate uses pickle
        internally (T-26-03).
    """

    def search(
        self,
        search_space: dict[str, list],
        objective_fn,
        n_trials: int,
        output_dir: str | Path,
        warm_start: list[dict] | None = None,
    ) -> list[tuple[dict, float]]:
        """Run MPI-parallel evolutionary optimisation and return all evaluated results.

        Parameters
        ----------
        search_space:
            Dict mapping param name → list of candidate values.  Values are
            converted to tuples internally for Propulate's limits format (D-07).
        objective_fn:
            Callable accepting a params dict and returning a float score.
            Score inversion is handled internally (D-08 — Propulate minimises).
        n_trials:
            Approximate total number of trials.  Converted to per-worker
            generations via ``max(1, ceil(n_trials / world_size))`` (Pitfall 4).
        output_dir:
            Directory where Propulate checkpoint files are written.
        warm_start:
            Accepted but silently ignored (D-09 — log only).

        Returns
        -------
        list[tuple[dict, float]]
            On rank 0: list of (params_dict, score) for each evaluated
            ``Individual`` with finite loss.  On all other ranks: empty list.
        """
        # D-12: lazy import with friendly ImportError (T-26-05 / Pitfall 8)
        try:
            from mpi4py import MPI
            from propulate import Propulator
            from propulate.utils import get_default_propagator, set_logger_config
        except ImportError as e:
            raise ImportError(
                "propulate is not installed. Install it with:\n"
                "  pip install zreg[propulate]"
            ) from e

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # D-09: warm_start silently ignored — log and continue
        if warm_start:
            _log.info("PropulateSearch: warm_start ignored (D-09)")

        # D-07: convert lists → tuples for Propulate's limits format.
        # Propulate infers parameter type from the first element: str→categorical,
        # int→ordinal range (min, max), float→continuous interval (min, max).
        # All our params are discrete choice lists, so force categorical by encoding
        # every value as a string. Decode by index-lookup against the original list
        # to restore the original Python type (int, float, None, str) before the
        # objective function receives the params.
        _originals: dict[str, list] = {k: list(vals) for k, vals in search_space.items()}

        def _encode(v) -> str:
            return "__none__" if v is None else str(v)

        def _decode_param(key: str, encoded):
            # Some propulate versions return a float/int category index rather than
            # the string label. Fall back to positional lookup in that case.
            if isinstance(encoded, (int, float)):
                return _originals[key][int(encoded)]
            encoded_list = [_encode(x) for x in _originals[key]]
            return _originals[key][encoded_list.index(encoded)]

        limits = {k: tuple(_encode(v) for v in vals) for k, vals in search_space.items()}

        # D-08: closure inverts sign because Propulate minimises; framework maximises
        def _loss(ind) -> float:
            # Use explicit comprehension — Individual is not a dict subclass (Pitfall 1)
            params = {k: _decode_param(k, ind[k]) for k in search_space}
            return -objective_fn(params)

        # Per-rank reproducibility: deterministic seed offset keeps ranks independent
        rng = random.Random(42 + rank)

        # Pitfall 4: convert n_trials → per-worker generations so total ≈ n_trials
        generations = max(1, math.ceil(n_trials / world_size))

        # T-26-05: configure propulate logger on rank 0 only (avoid duplicate files)
        if rank == 0:
            set_logger_config(level=logging.WARNING, log_to_stdout=False)

        propagator = get_default_propagator(
            pop_size=max(4, len(search_space)),
            limits=limits,
            rng=rng,
        )

        propulator = Propulator(
            loss_fn=_loss,
            propagator=propagator,
            rng=rng,
            island_comm=comm,           # single-island per RESEARCH Finding 3
            generations=generations,
            checkpoint_path=Path(output_dir),
        )

        try:
            propulator.propulate(logging_interval=max(1, generations // 5))
        finally:
            # Pitfall 5: keep all ranks in sync even on exception (avoid deadlock)
            comm.Barrier()

        # D-10: rank 0 only — non-rank-0 processes return empty list
        if rank != 0:
            return []

        # Iterate propulator.population directly (not summarize() — see RESEARCH
        # Finding 3 + anti-pattern note: summarize() performs allgather and returns
        # only top-N; population holds every evaluated individual on the current rank)
        results: list[tuple[dict, float]] = []
        for ind in propulator.population:
            # Pitfall 2: skip unevaluated stragglers (loss defaults to float("inf"))
            if ind.loss == float("inf"):
                continue
            try:
                params = {k: _decode_param(k, ind[k]) for k in search_space}
            except (ValueError, IndexError):
                # Stale checkpoint individual from an older search space (e.g. a param
                # that previously allowed None/'__none__' but no longer does). Safe to
                # skip — these individuals were evaluated under a different config and
                # their scores are not meaningful for the current search space.
                _log.warning("Skipping stale checkpoint individual (undecodable params): %s", dict(ind))
                continue
            score = -ind.loss  # D-08: invert sign back to maximisation scale
            results.append((params, score))
        return results
