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

Notes
-----
**Import order — macOS-ARM SIGABRT (Pitfall 7):**
``zreg.*`` imports MUST precede ``torch``, and ``torch`` MUST precede ``optuna``.
Interleaving these on macOS ARM (Apple Silicon) triggers a libomp SIGABRT.

**Anti-pattern:** Do NOT call ``EvaluationRunner`` inside ``objective_fn`` (D-07).
Keep each trial lightweight — instantiate stages directly.
"""

import itertools
import logging
import random
from pathlib import Path
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35.
from zreg.dataset import zRegPointCloud

import torch

import optuna

from eval.types import Trial, SearchResult

__all__ = ["GridSearch", "RandomSearch", "BayesianSearch"]

_log = logging.getLogger(__name__)


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
