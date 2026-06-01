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
import logging
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

    @patch("run_eval.EvaluationRunner")
    @patch("run_eval.HyperparamOptimizer")
    def test_optimize_mode_calls_optimizer(
        self, MockOpt, MockRunner, minimal_cfg_path, tmp_path
    ) -> None:
        """--mode optimize calls HyperparamOptimizer.run() exactly once; no EvaluationRunner."""
        mock_instance = MagicMock()
        MockOpt.return_value = mock_instance

        rc = run_eval.main([
            "--config", str(minimal_cfg_path),
            "--mode", "optimize",
            "--output-dir", str(tmp_path / "out"),
        ])

        assert rc == 0
        MockOpt.assert_called_once()
        mock_instance.run.assert_called_once()
        MockRunner.assert_not_called()

    @patch("run_eval.EvaluationRunner")
    @patch("run_eval.HyperparamOptimizer")
    def test_eval_mode_calls_runner(
        self, MockOpt, MockRunner, minimal_cfg_path, tmp_path
    ) -> None:
        """--mode eval calls EvaluationRunner.run() with loaded best_params; no Optimizer."""
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        best_params_path = out_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump({"window_size": 7}, f)

        mock_runner_instance = MagicMock()
        MockRunner.return_value = mock_runner_instance

        rc = run_eval.main([
            "--config", str(minimal_cfg_path),
            "--mode", "eval",
            "--output-dir", str(out_dir),
        ])

        assert rc == 0
        MockRunner.assert_called_once()
        # Verify params positional arg passed to EvaluationRunner.__init__
        assert MockRunner.call_args[0][1] == {"window_size": 7}
        mock_runner_instance.run.assert_called_once()
        MockOpt.assert_not_called()

    @patch("run_eval.EvaluationRunner")
    @patch("run_eval.HyperparamOptimizer")
    def test_full_mode_calls_optimizer_then_runner(
        self, MockOpt, MockRunner, minimal_cfg_path, tmp_path
    ) -> None:
        """--mode full calls Optimizer first (writes best_params.json), then EvaluationRunner."""
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer side-effect writes best_params.json so _load_best_params reads it
        def _write_best_params(*args, **kwargs):
            (out_dir / "best_params.json").write_text('{"window_size": 9}')

        mock_opt_instance = MagicMock()
        mock_opt_instance.run.side_effect = _write_best_params
        MockOpt.return_value = mock_opt_instance

        mock_runner_instance = MagicMock()
        MockRunner.return_value = mock_runner_instance

        rc = run_eval.main([
            "--config", str(minimal_cfg_path),
            "--mode", "full",
            "--output-dir", str(out_dir),
        ])

        assert rc == 0
        mock_opt_instance.run.assert_called_once()
        MockRunner.assert_called_once()
        # Proves runner saw the params written by optimizer (Pitfall 5 ordering)
        assert MockRunner.call_args[0][1] == {"window_size": 9}
        mock_runner_instance.run.assert_called_once()


class TestCLIErrorHandling:
    """FRAME-11: EvalConfigError produces a readable one-line message without stacktrace."""

    def test_config_error_clean_message(self, tmp_path, capsys) -> None:
        """Missing config file produces 'Error: EvalConfig: ...' on stderr, rc=1, no Traceback."""
        rc = run_eval.main([
            "--config", str(tmp_path / "nonexistent.yaml"),
            "--mode", "optimize",
        ])

        assert rc == 1
        captured = capsys.readouterr()
        assert "Error: EvalConfig:" in captured.err
        assert "Traceback" not in captured.err
        # Single line (strip to allow trailing newline from print)
        assert "\n" not in captured.err.strip()


class TestCLIHelp:
    """FRAME-11: --help shows all 4 flags."""

    def test_help_shows_four_flags(self, capsys) -> None:
        """--help exits 0 and prints all 4 argparse flags (Pitfall 7)."""
        with pytest.raises(SystemExit) as exc_info:
            run_eval.main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        for flag in ["--config", "--mode", "--output-dir", "--verbose"]:
            assert flag in captured.out


class TestCLIReproducibility:
    """FRAME-11: run_config.yaml is written to output_dir before the pipeline executes."""

    @patch("run_eval.HyperparamOptimizer")
    def test_run_config_yaml_written_before_pipeline(
        self, MockOpt, minimal_cfg_path, tmp_path
    ) -> None:
        """run_config.yaml is an exact copy of input YAML and exists even after pipeline fails.

        The optimizer raises RuntimeError (Pitfall 6) — proving the write
        happened BEFORE the pipeline was invoked.
        """
        MockOpt.return_value.run.side_effect = RuntimeError("pipeline failed")

        with pytest.raises(RuntimeError):
            run_eval.main([
                "--config", str(minimal_cfg_path),
                "--mode", "optimize",
                "--output-dir", str(tmp_path / "out"),
            ])

        run_config = tmp_path / "out" / "run_config.yaml"
        assert run_config.exists()
        assert run_config.read_bytes() == minimal_cfg_path.read_bytes()


class TestCLIParamSource:
    """FRAME-11: --mode eval logs which param source was used (D-01/D-02)."""

    @patch("run_eval.EvaluationRunner")
    def test_logs_loaded_params_when_file_exists(
        self, MockRunner, minimal_cfg_path, tmp_path, caplog
    ) -> None:
        """When best_params.json exists, CLI logs 'Loaded optimized params from ...'."""
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "best_params.json", "w") as f:
            json.dump({"k_neighbours": 4}, f)

        MockRunner.return_value = MagicMock()

        with caplog.at_level(logging.INFO, logger="run_eval"):
            rc = run_eval.main([
                "--config", str(minimal_cfg_path),
                "--mode", "eval",
                "--output-dir", str(out_dir),
                "--verbose",
            ])

        assert rc == 0
        assert any(
            "Loaded optimized params from" in rec.message for rec in caplog.records
        )

    @patch("run_eval.EvaluationRunner")
    def test_logs_config_defaults_when_file_missing(
        self, MockRunner, minimal_cfg_path, tmp_path, caplog
    ) -> None:
        """When best_params.json is absent, CLI logs 'No best_params.json found ...'."""
        out_dir = tmp_path / "out"
        # Do NOT create best_params.json

        MockRunner.return_value = MagicMock()

        with caplog.at_level(logging.INFO, logger="run_eval"):
            rc = run_eval.main([
                "--config", str(minimal_cfg_path),
                "--mode", "eval",
                "--output-dir", str(out_dir),
                "--verbose",
            ])

        assert rc == 0
        assert any(
            "No best_params.json found" in rec.message for rec in caplog.records
        )


_REPO_ROOT = Path(__file__).parent.parent

# D-04 scenario differentiation table: (filename, tier, n_trials, n_synthetic, run_alignment, run_label_transfer)
_SCENARIO_TABLE = [
    ("alignment_sanity.yaml", "sanity", 3, 20, True, False),
    ("alignment_dev.yaml", "dev", 10, 50, True, False),
    ("label_transfer_sanity.yaml", "sanity", 3, 20, False, True),
    ("label_transfer_dev.yaml", "dev", 10, 50, False, True),
    ("combined_full.yaml", "full", 20, 100, True, True),
]

_SCENARIO_IDS = [row[0].replace(".yaml", "") for row in _SCENARIO_TABLE]


class TestScenarioConfigs:
    """FRAME-12: all 5 scenario YAML configs load without EvalConfigError."""

    @pytest.mark.parametrize("scenario", _SCENARIO_TABLE, ids=_SCENARIO_IDS)
    def test_config_loads_without_error(self, scenario) -> None:
        """Each scenario YAML loads via EvalConfig.from_yaml without raising EvalConfigError.

        Also asserts cfg.data_path ends with the kobitski tracklets filename (D-03).
        """
        filename = scenario[0]
        cfg = EvalConfig.from_yaml(_REPO_ROOT / "configs" / filename)
        assert isinstance(cfg, EvalConfig)
        assert cfg.data_path.endswith(
            "12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets"
        )

    @pytest.mark.parametrize("scenario", _SCENARIO_TABLE, ids=_SCENARIO_IDS)
    def test_config_has_d04_values(self, scenario) -> None:
        """Each scenario config has exactly the D-04 tier/stage-flag/trial/synthetic values."""
        filename, expected_tier, expected_n_trials, expected_n_synthetic, expected_run_alignment, expected_run_label_transfer = scenario
        cfg = EvalConfig.from_yaml(_REPO_ROOT / "configs" / filename)
        assert cfg.tier == expected_tier
        assert cfg.n_trials == expected_n_trials
        assert cfg.n_synthetic == expected_n_synthetic
        assert cfg.run_alignment is expected_run_alignment
        assert cfg.run_label_transfer is expected_run_label_transfer
        assert cfg.data_format == "tracklets"
