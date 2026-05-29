"""Tests for eval.runners.optimizer.HyperparamOptimizer (FRAME-09) and
eval.search_strategies (FRAME-10).

Covers:
  - FRAME-09-SC1: sanity tier completes in under 2 minutes
  - FRAME-09-SC2: best_params.json and search_history.json written to output_dir
  - FRAME-09-SC4: prune_candidates() reduces candidate count
  - FRAME-10-SC3: best params from optimizer improve score vs default params
  - FRAME-10: BayesianSearch uses TPE sampler with n_startup_trials >= 2*N_params
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_labels, generate_trajectory

import torch

from eval.config import EvalConfig

try:
    from eval.runners import HyperparamOptimizer
except ImportError:
    HyperparamOptimizer = None  # type: ignore[assignment,misc]

try:
    from eval.search_strategies import GridSearch, RandomSearch, BayesianSearch
except ImportError:
    GridSearch = None  # type: ignore[assignment,misc]
    RandomSearch = None  # type: ignore[assignment,misc]
    BayesianSearch = None  # type: ignore[assignment,misc]

from eval.types import Trial, SearchResult, StageMetrics


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optimizer_config(tmp_path) -> EvalConfig:
    """EvalConfig with search_space and output_dir inside tmp_path."""
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "output"),
        search_strategy="grid",
        tier="sanity",
        n_trials=3,
        run_alignment=True,
        run_label_transfer=True,
        search_space={
            "window_size": [3, 5],
            "k_neighbours": [3, 5],
        },
    )


@pytest.fixture
def synthetic_dataset() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with labels (id field populated).

    generate_labels() populates the 'id' field (torch.long, shape (N,))
    so DataFactory.get_ground_truth() returns valid id tensors.
    Mirror of test_eval_runner.py:51-58 fixture.
    """
    traj = generate_trajectory(n_points=20, n_frames=3, seed=0)
    return generate_labels(traj, n_classes=4, seed=0)


@pytest.fixture
def full_params() -> dict:
    """Complete params dict covering all 9 required keys."""
    return {
        "window_size": 10,
        "step": 1,
        "cpd_penalty": None,
        "dtw_dist_fn": "euclidean",
        "n_breakpoints": 5,
        "k_neighbours": 5,
        "dist_metric": "euclidean",
        "smoothing": 0.0,
        "threshold": 0.0,
    }


# ---------------------------------------------------------------------------
# Test classes (Wave 0 stubs — populated in Plan 22-02)
# ---------------------------------------------------------------------------


class TestHyperparamOptimizerSanityTier:
    """FRAME-09-SC1: sanity tier completes in under 2 minutes."""

    def test_sanity_tier_completes_under_two_minutes(
        self, optimizer_config, synthetic_dataset
    ) -> None:
        pytest.skip("populated in Plan 22-02")


class TestHyperparamOptimizerOutputFiles:
    """FRAME-09-SC2: best_params.json and search_history.json written."""

    def test_output_files_written(
        self, optimizer_config, synthetic_dataset
    ) -> None:
        pytest.skip("populated in Plan 22-02")


class TestPruneCandidates:
    """FRAME-09-SC4: prune_candidates() reduces candidate count."""

    def test_prune_reduces_count(self) -> None:
        pytest.skip("populated in Plan 22-02")


class TestBestParamsImproveDefault:
    """FRAME-10-SC3: optimizer best params improve score vs default params."""

    def test_best_params_improve_default_score(self) -> None:
        pytest.skip("populated in Plan 22-02")


class TestBayesianSearch:
    """FRAME-10: BayesianSearch uses TPE with n_startup_trials >= 2*N_params."""

    def test_tpe_sampler_n_startup_trials(self) -> None:
        pytest.skip("populated in Plan 22-02")
