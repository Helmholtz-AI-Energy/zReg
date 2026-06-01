"""Tests for run_eval CLI (FRAME-11 + FRAME-12).

Covers:
  - FRAME-11: mode dispatch (optimize/eval/full), EvalConfigError clean message,
    --help shows 4 flags, run_config.yaml written to output_dir before pipeline,
    --mode eval logs correct param source (best_params present / missing)
  - FRAME-12: all 5 scenario YAML configs in configs/ load without EvalConfigError

Plan 01 populates all 8 FRAME-11 test methods.
Plan 02 (Task 2) replaces TestScenarioConfigs.test_placeholder with FRAME-12
assertions for the 5 scenario YAML configs.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud

import torch

import run_eval
from eval.config import EvalConfig, EvalConfigError


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_cfg_path(tmp_path) -> Path:
    """Write a minimal valid YAML config and return its path.

    Uses the kobitski tracklets file (confirmed present at
    data/external/sample/kobitski_data/...) so EvalConfig.from_yaml
    accepts it without raising EvalConfigError.
    """
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "data_path: data/external/sample/kobitski_data/"
        "12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets\n"
    )
    return cfg


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestCLIModeDispatch:
    """FRAME-11: mode dispatch — optimize, eval, full."""

    def test_optimize_mode_calls_optimizer(self, minimal_cfg_path, tmp_path):
        raise NotImplementedError("populated in Task 2")

    def test_eval_mode_calls_runner(self, minimal_cfg_path, tmp_path):
        raise NotImplementedError("populated in Task 2")

    def test_full_mode_calls_optimizer_then_runner(self, minimal_cfg_path, tmp_path):
        raise NotImplementedError("populated in Task 2")


class TestCLIErrorHandling:
    """FRAME-11: EvalConfigError produces a readable one-line message without stacktrace."""

    def test_config_error_clean_message(self, tmp_path, capsys):
        raise NotImplementedError("populated in Task 2")


class TestCLIHelp:
    """FRAME-11: --help shows all 4 flags."""

    def test_help_shows_four_flags(self, capsys):
        raise NotImplementedError("populated in Task 2")


class TestCLIReproducibility:
    """FRAME-11: run_config.yaml is written to output_dir before the pipeline executes."""

    def test_run_config_yaml_written_before_pipeline(self, minimal_cfg_path, tmp_path):
        raise NotImplementedError("populated in Task 2")


class TestCLIParamSource:
    """FRAME-11: --mode eval logs which param source was used (D-01/D-02)."""

    def test_logs_loaded_params_when_file_exists(self, minimal_cfg_path, tmp_path, caplog):
        raise NotImplementedError("populated in Task 2")

    def test_logs_config_defaults_when_file_missing(self, minimal_cfg_path, tmp_path, caplog):
        raise NotImplementedError("populated in Task 2")


class TestScenarioConfigs:
    """FRAME-12: all 5 scenario YAML configs load without EvalConfigError.

    Populated by Plan 02 Task 2 — placeholder below keeps the test suite
    collection count stable during Plan 01 execution.
    """

    def test_placeholder(self):
        pytest.skip("populated in Plan 02 (FRAME-12)")
