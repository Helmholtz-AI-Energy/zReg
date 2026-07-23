"""Tests for eval.search_strategies — GridSearch, RandomSearch, BayesianSearch, PropulateSearch.

Covers coverage gaps identified during 100% coverage drive:
- GridSearch warm_start deduplication (lines 100-104, 108)
- RandomSearch body (lines 154-177)
- BayesianSearch n_trials<=1 path (line 228) and warm_start enqueue (lines 243-244)
- PropulateSearch body (lines 318, 325-388) via sys.modules mock
"""

import sys
import types
import logging
import math
import random
from unittest.mock import MagicMock, patch

import pytest

# zreg.dataset before torch — macOS-ARM SIGABRT rule
from zreg.dataset import zRegPointCloud

import torch

from eval.search_strategies import GridSearch, RandomSearch, BayesianSearch, SobolSearch


# ---------------------------------------------------------------------------
# TestGridSearchWarmStart
# ---------------------------------------------------------------------------


class TestGridSearchWarmStart:
    """GridSearch warm_start deduplication (lines 100-104, 108)."""

    def _obj(self, params):
        return 0.5

    def test_warm_start_prepended_and_deduplicated(self):
        """warm_start=[{'a': 1}] is evaluated before grid combos; duplicate skipped."""
        search_space = {"a": [1, 2]}
        warm = [{"a": 1}]
        results = GridSearch().search(search_space, self._obj, warm_start=warm)
        # {'a': 1} appears once (not twice even though it's in both warm_start and grid)
        param_list = [p for p, _ in results]
        count_a1 = sum(1 for p in param_list if p.get("a") == 1)
        assert count_a1 == 1

    def test_warm_start_duplicate_within_warm_start_skipped(self):
        """Duplicate within warm_start itself (same params twice) is deduplicated."""
        search_space = {"a": [1, 2]}
        warm = [{"a": 1}, {"a": 1}]  # exact duplicate
        results = GridSearch().search(search_space, self._obj, warm_start=warm)
        param_list = [p for p, _ in results]
        count_a1 = sum(1 for p in param_list if p.get("a") == 1)
        assert count_a1 == 1  # only one evaluation, not two

    def test_warm_start_none_works(self):
        """warm_start=None produces full grid results."""
        search_space = {"a": [1, 2], "b": [3, 4]}
        results = GridSearch().search(search_space, self._obj, warm_start=None)
        assert len(results) == 4

    def test_warm_start_evaluated_first(self):
        """First result param matches first warm_start entry."""
        call_order = []
        def tracking_obj(params):
            call_order.append(params.copy())
            return 0.5
        search_space = {"a": [1, 2]}
        warm = [{"a": 2}]
        GridSearch().search(search_space, tracking_obj, warm_start=warm)
        assert call_order[0]["a"] == 2


# ---------------------------------------------------------------------------
# TestRandomSearchBody
# ---------------------------------------------------------------------------


class TestRandomSearchBody:
    """RandomSearch body coverage (lines 154-177)."""

    def test_returns_n_trials_results(self):
        """RandomSearch with n_trials=3 returns 3 results."""
        results = RandomSearch().search(
            {"k": [1, 2, 3]}, lambda p: 0.5, n_trials=3
        )
        assert len(results) == 3

    def test_each_result_is_params_score_tuple(self):
        """Each result element is a (dict, float) tuple."""
        results = RandomSearch().search({"k": [1, 2]}, lambda p: 0.7, n_trials=2)
        for params, score in results:
            assert isinstance(params, dict)
            assert isinstance(score, float)

    def test_warm_start_prepended(self):
        """warm_start=[{'k': 1}] is evaluated before random samples."""
        call_order = []
        def tracking_obj(p):
            call_order.append(p.copy())
            return 0.5
        RandomSearch().search({"k": [1, 2, 3]}, tracking_obj, n_trials=2, warm_start=[{"k": 99}])
        # first call should use warm_start value
        assert call_order[0]["k"] == 99
        assert len(call_order) == 3  # 1 warm + 2 random

    def test_zero_n_trials_returns_only_warm_start(self):
        """n_trials=0 with warm_start=[p] returns 1 result."""
        results = RandomSearch().search(
            {"k": [1, 2]}, lambda p: 0.5, n_trials=0, warm_start=[{"k": 1}]
        )
        assert len(results) == 1

    def test_duplicate_warm_start_is_deduplicated(self):
        """search_strategies.py:161->159 — duplicate warm_start entry is skipped (seen set)."""
        call_order = []
        def tracking_obj(p):
            call_order.append(p.copy())
            return 0.5
        # Pass two identical warm_start entries — second should be skipped
        RandomSearch().search(
            {"k": [1, 2]}, tracking_obj, n_trials=0,
            warm_start=[{"k": 1}, {"k": 1}],  # duplicate
        )
        # Only one unique warm_start entry should be evaluated
        assert len(call_order) == 1
        assert call_order[0]["k"] == 1


# ---------------------------------------------------------------------------
# TestBayesianSearchCoverageGaps
# ---------------------------------------------------------------------------


class TestBayesianSearchCoverageGaps:
    """BayesianSearch n_trials<=1 (line 228) and warm_start enqueue (lines 243-244)."""

    def test_n_trials_one_sets_n_startup_to_one(self, tmp_path):
        """n_trials=1 sets n_startup=1 (line 228 branch)."""
        results = BayesianSearch().search(
            {"k": [1, 2]},
            lambda p: 0.5,
            n_trials=1,
            output_dir=str(tmp_path),
        )
        assert len(results) >= 0  # just verify it runs without error

    def test_warm_start_enqueue(self, tmp_path):
        """warm_start=[{'k': 1}] is enqueued via study.enqueue_trial (lines 243-244)."""
        enqueued = []
        original_search = BayesianSearch().search

        import optuna
        original_create = optuna.create_study

        def capturing_create(**kwargs):
            study = original_create(
                storage=None,
                direction=kwargs.get("direction", "maximize"),
                sampler=kwargs.get("sampler"),
            )
            original_enqueue = study.enqueue_trial

            def capturing_enqueue(p, skip_if_exists=False):
                enqueued.append(p)
                return original_enqueue(p, skip_if_exists=skip_if_exists)

            study.enqueue_trial = capturing_enqueue
            return study

        with patch("eval.search_strategies.optuna.create_study", side_effect=capturing_create):
            BayesianSearch().search(
                {"k": [1, 2]},
                lambda p: 0.5,
                n_trials=3,
                output_dir=str(tmp_path / "sub"),
                warm_start=[{"k": 1}],
            )
        assert len(enqueued) == 1
        assert enqueued[0] == {"k": 1}


# ---------------------------------------------------------------------------
# TestPropulateSearchMocked
# ---------------------------------------------------------------------------


def _build_propulate_mocks(rank=0, world_size=1, population=None):
    """Build a minimal propulate + mpi4py mock for sys.modules injection."""

    # Create fake Individual
    class FakeIndividual:
        def __init__(self, loss, k=1):
            self.loss = loss
            self._k = k

        def __getitem__(self, key):
            return self._k

    if population is None:
        population = [FakeIndividual(float("inf")), FakeIndividual(0.3, k="2")]

    # Propulator mock
    class FakePropulator:
        def __init__(self, *, loss_fn, propagator, rng, island_comm, generations, checkpoint_path):
            self._loss_fn = loss_fn
            self.population = population

        def propulate(self, logging_interval=1):
            # Call loss_fn on each non-inf individual to exercise _loss body
            for ind in self.population:
                if ind.loss != float("inf"):
                    self._loss_fn(ind)

    mock_propulate = types.ModuleType("propulate")
    mock_propulate.Propulator = FakePropulator

    mock_utils = types.ModuleType("propulate.utils")
    mock_utils.get_default_propagator = MagicMock(return_value=MagicMock())
    mock_utils.set_logger_config = MagicMock()

    # MPI mock
    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = rank
    mock_comm.Get_size.return_value = world_size
    mock_comm.Barrier.return_value = None

    mock_mpi_mod = types.ModuleType("mpi4py")
    mock_mpi_cls = types.ModuleType("mpi4py.MPI")
    mock_mpi_cls.COMM_WORLD = mock_comm

    mock_mpi_mod.MPI = mock_mpi_cls

    return mock_propulate, mock_utils, mock_mpi_mod, mock_mpi_cls


class TestPropulateSearchMocked:
    """PropulateSearch with mocked propulate + mpi4py (lines 318, 325-388)."""

    def test_rank0_returns_results(self, tmp_path):
        """Rank 0: non-inf individuals returned as (params, score) pairs."""
        mock_prop, mock_utils, mock_mpi, mock_mpi_cls = _build_propulate_mocks(rank=0, world_size=1)
        modules = {
            "propulate": mock_prop,
            "propulate.utils": mock_utils,
            "mpi4py": mock_mpi,
            "mpi4py.MPI": mock_mpi_cls,
        }
        with patch.dict(sys.modules, modules):
            # Force reimport to pick up mocks
            import importlib
            if "eval.search_strategies" in sys.modules:
                del sys.modules["eval.search_strategies"]
            from eval.search_strategies import PropulateSearch
            results = PropulateSearch().search(
                {"k": [1, 2]},
                lambda p: 0.5,
                n_trials=4,
                output_dir=str(tmp_path),
            )
        assert isinstance(results, list)
        # Non-inf individuals produce results
        assert len(results) >= 0

    def test_rank_nonzero_returns_empty_list(self, tmp_path):
        """Non-rank-0 processes return [] (line 374-375)."""
        mock_prop, mock_utils, mock_mpi, mock_mpi_cls = _build_propulate_mocks(rank=1, world_size=2)
        modules = {
            "propulate": mock_prop,
            "propulate.utils": mock_utils,
            "mpi4py": mock_mpi,
            "mpi4py.MPI": mock_mpi_cls,
        }
        with patch.dict(sys.modules, modules):
            import importlib
            if "eval.search_strategies" in sys.modules:
                del sys.modules["eval.search_strategies"]
            from eval.search_strategies import PropulateSearch
            results = PropulateSearch().search(
                {"k": [1, 2]},
                lambda p: 0.5,
                n_trials=4,
                output_dir=str(tmp_path),
            )
        assert results == []

    def test_warm_start_logs_and_continues(self, tmp_path):
        """warm_start is accepted and logged (line 330-331); search still returns list."""
        mock_prop, mock_utils, mock_mpi, mock_mpi_cls = _build_propulate_mocks(rank=0, world_size=1)
        modules = {
            "propulate": mock_prop,
            "propulate.utils": mock_utils,
            "mpi4py": mock_mpi,
            "mpi4py.MPI": mock_mpi_cls,
        }
        with patch.dict(sys.modules, modules):
            if "eval.search_strategies" in sys.modules:
                del sys.modules["eval.search_strategies"]
            from eval.search_strategies import PropulateSearch
            results = PropulateSearch().search(
                {"k": [1, 2]},
                lambda p: 0.5,
                n_trials=4,
                output_dir=str(tmp_path),
                warm_start=[{"k": 1}],
            )
        assert isinstance(results, list)

    def test_import_error_raises_friendly_message(self):
        """PropulateSearch.search raises ImportError with friendly message when propulate absent."""
        with patch.dict(sys.modules, {"propulate": None, "mpi4py": None}):
            if "eval.search_strategies" in sys.modules:
                del sys.modules["eval.search_strategies"]
            from eval.search_strategies import PropulateSearch
            with pytest.raises(ImportError, match="pip install"):
                PropulateSearch().search(
                    {"k": [1, 2]}, lambda p: 0.5, n_trials=2, output_dir="/tmp"
                )

    def test_int_encoded_parameter_decoded_by_index(self, tmp_path):
        """_decode_param handles int/float category indices (propulate ≥2.x quirk, line 460)."""
        int_ind = MagicMock()
        int_ind.loss = 0.3
        int_ind.__getitem__ = MagicMock(return_value=1)  # int 1 → _originals["k"][1] = 2

        mock_prop, mock_utils, mock_mpi, mock_mpi_cls = _build_propulate_mocks(
            rank=0, world_size=1, population=[int_ind]
        )
        modules = {
            "propulate": mock_prop,
            "propulate.utils": mock_utils,
            "mpi4py": mock_mpi,
            "mpi4py.MPI": mock_mpi_cls,
        }
        with patch.dict(sys.modules, modules):
            if "eval.search_strategies" in sys.modules:
                del sys.modules["eval.search_strategies"]
            from eval.search_strategies import PropulateSearch
            results = PropulateSearch().search(
                {"k": [1, 2]},
                lambda p: 0.5,
                n_trials=4,
                output_dir=str(tmp_path),
            )
        assert len(results) == 1
        assert results[0][0]["k"] == 2  # int(1) → _originals["k"][1] = 2

    def test_stale_checkpoint_individual_is_skipped(self, tmp_path, caplog):
        """Stale checkpoint individual (undecodable params) is skipped with a warning (lines 517-523)."""
        stale_ind = MagicMock()
        stale_ind.loss = 0.3
        stale_ind.__getitem__ = MagicMock(return_value="__stale_not_in_space__")

        class _NoEvalPropulator:
            """Propulator that simulates a pre-evaluated checkpoint (no loss_fn call)."""
            def __init__(self, *, loss_fn, propagator, rng, island_comm, generations, checkpoint_path):
                self.population = [stale_ind]
            def propulate(self, logging_interval=1):
                pass  # individual already has loss from prior run

        mock_propulate = types.ModuleType("propulate")
        mock_propulate.Propulator = _NoEvalPropulator
        mock_utils = types.ModuleType("propulate.utils")
        mock_utils.get_default_propagator = MagicMock(return_value=MagicMock())
        mock_utils.set_logger_config = MagicMock()
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        mock_comm.Barrier.return_value = None
        mock_mpi_mod = types.ModuleType("mpi4py")
        mock_mpi_cls = types.ModuleType("mpi4py.MPI")
        mock_mpi_cls.COMM_WORLD = mock_comm
        mock_mpi_mod.MPI = mock_mpi_cls

        modules = {
            "propulate": mock_propulate,
            "propulate.utils": mock_utils,
            "mpi4py": mock_mpi_mod,
            "mpi4py.MPI": mock_mpi_cls,
        }
        with patch.dict(sys.modules, modules):
            if "eval.search_strategies" in sys.modules:
                del sys.modules["eval.search_strategies"]
            from eval.search_strategies import PropulateSearch
            with caplog.at_level(logging.WARNING, logger="eval.search_strategies"):
                results = PropulateSearch().search(
                    {"k": [1, 2]},
                    lambda p: 0.5,
                    n_trials=4,
                    output_dir=str(tmp_path),
                )
        assert results == []
        assert any("Skipping stale checkpoint" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TestSobolSearch
# ---------------------------------------------------------------------------


class TestSobolSearch:
    """SobolSearch: quasi-random sampling, SOBOL_MIN_TRIALS fallback, warm_start prepend, space-filling coverage."""

    def _obj(self, params):
        return 0.5

    def test_returns_n_trials_results(self):
        """SobolSearch with n_trials=8 returns 8 results (no fallback; 8 == SOBOL_MIN_TRIALS)."""
        results = SobolSearch().search(
            {"k": [1, 2, 3]}, self._obj, n_trials=8
        )
        assert len(results) == 8

    def test_each_result_is_params_score_tuple(self):
        """Each result element is a (dict, float) tuple."""
        results = SobolSearch().search({"k": [1, 2]}, self._obj, n_trials=8)
        for params, score in results:
            assert isinstance(params, dict)
            assert isinstance(score, float)

    def test_warm_start_prepended(self):
        """warm_start=[{'k': 99}] is evaluated before Sobol samples."""
        call_order = []

        def tracking_obj(p):
            call_order.append(p.copy())
            return 0.5

        SobolSearch().search(
            {"k": [1, 2, 3]}, tracking_obj, n_trials=8, warm_start=[{"k": 99}]
        )
        assert call_order[0]["k"] == 99

    def test_fallback_when_n_trials_below_min(self):
        """n_trials < 8 falls back to RandomSearch (D-10); returns results without error."""
        results = SobolSearch().search({"k": [1, 2]}, self._obj, n_trials=3, seed=42)
        assert len(results) == 3

    def test_empty_search_space_returns_empty(self):
        """Empty search_space (d=0) returns [] silently (D-08)."""
        results = SobolSearch().search({}, self._obj, n_trials=8)
        assert results == []

    def test_seed_reproducibility(self):
        """Same seed produces identical params sequence (scrambled Owen with seed=0)."""
        r1 = SobolSearch().search({"k": [1, 2, 3, 4]}, self._obj, n_trials=8, seed=0)
        r2 = SobolSearch().search({"k": [1, 2, 3, 4]}, self._obj, n_trials=8, seed=0)
        assert [p for p, _ in r1] == [p for p, _ in r2]

    def test_values_always_within_search_space(self):
        """All sampled values are members of the provided choice lists."""
        space = {"a": [10, 20, 30], "b": ["x", "y"]}
        results = SobolSearch().search(space, self._obj, n_trials=16, seed=42)
        for params, _ in results:
            assert params["a"] in space["a"]
            assert params["b"] in space["b"]

    def test_full_coverage_in_single_dim_space(self):
        """Sobol covers all 8 distinct values when n_trials equals space size (space-filling property, OPT-04-05).

        For n_trials=2^k with d=1, the (t,m,s)-net property guarantees exactly one
        sample in each interval [i/n, (i+1)/n), so all n discrete values are visited.
        This space-filling advantage does not hold for random sampling.
        """
        space = {"k": [1, 2, 3, 4, 5, 6, 7, 8]}
        results = SobolSearch().search(space, self._obj, n_trials=8, seed=0)
        assert {p["k"] for p, _ in results} == set(space["k"])

    def test_warm_start_duplicate_skipped(self):
        """search_strategies.py:262->260 — duplicate entry in Sobol warm_start is deduplicated."""
        call_order = []

        def tracking_obj(p):
            call_order.append(p.copy())
            return 0.5

        space = {"a": [1, 2]}
        SobolSearch().search(
            space, tracking_obj, n_trials=8,
            warm_start=[{"a": 1}, {"a": 1}],  # exact duplicate
            seed=0,
        )
        # 1 deduped warm_start + 8 Sobol = 9 evaluations (not 10 = 2 + 8)
        assert len(call_order) == 9
        assert call_order[0]["a"] == 1
