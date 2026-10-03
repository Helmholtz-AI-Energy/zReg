"""Tests for PropulateSearch and _detect_backend (Phase 26 / EXT-03).

Covers decisions:
  - D-04: _detect_backend() resolves auto → propulate/bayesian by env
  - D-05: mpi4py ImportError silently caught in _detect_backend
  - D-06: PropulateSearch.search() outer signature mirrors BayesianSearch
  - D-10: PropulateSearch.search() return contract (rank-gated, inf-filtered)
  - D-12: lazy ImportError raises 'pip install zreg[propulate]' message
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.core.dataset import zRegPointCloud  # noqa: F401

import torch  # noqa: F401

from eval.config import EvalConfig
from eval.runners.optimizer import HyperparamOptimizer
from eval.types import StageMetrics, Trial


# ---------------------------------------------------------------------------
# Module-level fixture for _detect_backend tests
# ---------------------------------------------------------------------------


@pytest.fixture
def optimizer_auto(tmp_path):
    """HyperparamOptimizer with search_strategy='auto' for _detect_backend tests."""
    cfg = EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path),
        search_strategy="auto",
        search_space={"window_size": [3, 5]},
    )
    return HyperparamOptimizer(cfg)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPropulateImportGuard:
    """D-12: lazy ImportError raises friendly ImportError when propulate is missing."""

    def test_propulate_missing_raises_helpful_error(self, monkeypatch, tmp_path):
        # Pitfall 8: monkeypatch.setitem with None → next `import propulate` fails.
        # Must patch BEFORE importing PropulateSearch so the lazy import sees None.
        monkeypatch.setitem(sys.modules, "propulate", None)
        # Also patch propulate.utils which is imported alongside propulate
        monkeypatch.setitem(sys.modules, "propulate.utils", None)

        from eval.search_strategies import PropulateSearch

        with pytest.raises(ImportError, match=r"pip install zreg\[propulate\]"):
            PropulateSearch().search(
                search_space={"x": [1, 2]},
                objective_fn=lambda p: 0.5,
                n_trials=2,
                output_dir=str(tmp_path),
            )


class TestEvalConfigAcceptsPropulate:
    """D-01: EvalConfig accepts 'propulate' and 'auto' as valid search_strategy values."""

    def test_propulate_value_accepted(self):
        cfg = EvalConfig(data_path="unused.mat", search_strategy="propulate")
        assert cfg.search_strategy == "propulate"

    def test_auto_value_accepted(self):
        cfg = EvalConfig(data_path="unused.mat", search_strategy="auto")
        assert cfg.search_strategy == "auto"


class TestDetectBackend:
    """D-04 / D-05: backend auto-detection from environment."""

    def test_fallback_to_bayesian_when_no_mpi_no_slurm(self, optimizer_auto, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with patch("mpi4py.MPI.COMM_WORLD") as mock_comm:
            mock_comm.Get_size.return_value = 1
            assert optimizer_auto._detect_backend() == "bayesian"

    def test_slurm_job_id_returns_propulate(self, optimizer_auto, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        with patch("mpi4py.MPI.COMM_WORLD") as mock_comm:
            mock_comm.Get_size.return_value = 1
            assert optimizer_auto._detect_backend() == "propulate"

    def test_mpi_world_size_gt_one_returns_propulate(self, optimizer_auto, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with patch("mpi4py.MPI.COMM_WORLD") as mock_comm:
            mock_comm.Get_size.return_value = 2
            assert optimizer_auto._detect_backend() == "propulate"

    def test_mpi4py_missing_falls_through(self, optimizer_auto, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setitem(sys.modules, "mpi4py", None)
        # Should not raise; falls through to bayesian because mpi4py import fails
        assert optimizer_auto._detect_backend() == "bayesian"


class TestRunDispatchPropulate:
    """D-11: propulate dispatch branch constructs Trial objects with minimal StageMetrics."""

    def test_propulate_dispatch_writes_best_params(self, tmp_path):
        # Build EvalConfig with search_strategy="propulate" and minimal search_space.
        # run_alignment=True, run_label_transfer=True keeps __init__ guard satisfied;
        # PropulateSearch is mocked away; the fake search evaluates each
        # individual through the objective (a stubbed _objective records a real
        # Trial), as Propulate does on the evaluating rank. Since 62-REVIEW
        # CR-01 only real trials can become best_params; returned pairs without
        # an evaluation record are flagged placeholders.
        cfg = EvalConfig(
            data_path=str(tmp_path / "unused.mat"),
            output_dir=str(tmp_path / "out"),
            search_strategy="propulate",
            tier="sanity",
            n_trials=2,
            run_alignment=True,
            run_label_transfer=True,
            search_space={"window_size": [3, 5]},
        )

        fake_results = [({"window_size": 3}, 0.7), ({"window_size": 5}, 0.5)]

        scores = {p["window_size"]: sc for p, sc in fake_results}

        def fake_objective(self, params, tier_dataset, tier_name, history):
            score = scores[params["window_size"]]
            history.append(Trial(
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
                tier=tier_name,
            ))
            self._n_succeeded += 1
            return score

        def fake_search(search_space, objective, **kwargs):
            for params, _ in fake_results:
                objective(dict(params))
            return fake_results

        with patch("eval.runners.optimizer.PropulateSearch") as mock_cls, \
                patch.object(HyperparamOptimizer, "_objective", fake_objective):
            mock_cls.return_value.search.side_effect = fake_search
            result = HyperparamOptimizer(cfg).run()

        # Both returned pairs match a real trial: no placeholders.
        assert len(result.history) == 2
        assert all(not t.flags for t in result.history)

        # Best result is the one with highest score (window_size=3, score=0.7)
        assert result.best_params == {"window_size": 3}, (
            f"Expected best_params={{'window_size': 3}}, got {result.best_params}"
        )
        assert result.best_score == pytest.approx(0.7), (
            f"Expected best_score=0.7, got {result.best_score}"
        )

        # Output files must exist
        out_dir = tmp_path / "out"
        assert (out_dir / "best_params.json").exists(), "best_params.json not written"
        assert (out_dir / "search_history.json").exists(), "search_history.json not written"

        # Validate best_params.json content
        with open(out_dir / "best_params.json") as f:
            bp = json.load(f)
        assert bp == {"window_size": 3}, f"best_params.json content mismatch: {bp}"

        # Mock was called with expected kwargs
        mock_cls.return_value.search.assert_called_once()
        call_kwargs = mock_cls.return_value.search.call_args
        assert "n_trials" in call_kwargs.kwargs or len(call_kwargs.args) >= 3, (
            "search() must be called with n_trials"
        )
        assert "output_dir" in call_kwargs.kwargs or len(call_kwargs.args) >= 4, (
            "search() must be called with output_dir"
        )
        assert "warm_start" in call_kwargs.kwargs or len(call_kwargs.args) >= 5, (
            "search() must be called with warm_start"
        )


class TestPropulateMPIIntegration:
    """End-to-end: mpirun -n 2 produces non-empty results on rank 0."""

    def test_mpirun_n2_returns_results(self, tmp_path):  # pragma: no cover
        pytest.importorskip("propulate")
        pytest.importorskip("mpi4py")

        helper = Path(__file__).parent / "_propulate_mwe.py"
        assert helper.exists(), f"MPI helper script not found at {helper}"

        mpirun = shutil.which("mpirun")
        if mpirun is None:
            pytest.skip("mpirun not found on PATH")

        out_file = tmp_path / "mpi_result.json"

        result = subprocess.run(
            [mpirun, "-n", "2", sys.executable, str(helper), str(out_file)],
            capture_output=True,
            timeout=60,  # T-26-06: bounded subprocess prevents DoS from runaway MPI
        )

        assert result.returncode == 0, (
            f"mpirun failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert out_file.exists(), "rank 0 did not write results.json"

        results = json.loads(out_file.read_text())
        assert len(results) > 0, "PropulateSearch returned empty results on rank 0"
        # JSON round-trip: tuples → lists; each element is [params_dict, score]
        assert all(
            isinstance(r, list) and len(r) == 2 for r in results
        ), f"Expected list of [params, score] pairs, got: {results[:2]}"
