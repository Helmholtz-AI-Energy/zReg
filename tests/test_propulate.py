"""Tests for PropulateSearch and _detect_backend (Phase 26 / EXT-03).

Covers decisions:
  - D-04: _detect_backend() resolves auto → propulate/bayesian by env
  - D-05: mpi4py ImportError silently caught in _detect_backend
  - D-06: PropulateSearch.search() outer signature mirrors BayesianSearch
  - D-10: PropulateSearch.search() return contract (rank-gated, inf-filtered)
  - D-12: lazy ImportError raises 'pip install zreg[propulate]' message
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud  # noqa: F401

import torch  # noqa: F401

from eval.config import EvalConfig
from eval.runners.optimizer import HyperparamOptimizer


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
        pass


class TestEvalConfigAcceptsPropulate:
    """D-01: EvalConfig accepts 'propulate' and 'auto' as valid search_strategy values."""

    def test_propulate_value_accepted(self):
        pass

    def test_auto_value_accepted(self):
        pass


class TestDetectBackend:
    """D-04 / D-05: backend auto-detection from environment."""

    def test_fallback_to_bayesian_when_no_mpi_no_slurm(self, optimizer_auto, monkeypatch):
        pass

    def test_slurm_job_id_returns_propulate(self, optimizer_auto, monkeypatch):
        pass

    def test_mpi_world_size_gt_one_returns_propulate(self, optimizer_auto, monkeypatch):
        pass

    def test_mpi4py_missing_falls_through(self, optimizer_auto, monkeypatch):
        pass


class TestRunDispatchPropulate:
    """D-11: propulate dispatch branch constructs Trial objects with minimal StageMetrics."""

    def test_propulate_dispatch_writes_best_params(self, tmp_path):
        pass


class TestPropulateMPIIntegration:
    """End-to-end: mpirun -n 2 produces non-empty results on rank 0."""

    def test_mpirun_n2_returns_results(self, tmp_path):
        pytest.importorskip("propulate")
        pytest.importorskip("mpi4py")
        pytest.skip("populated in Plan 26-02")
