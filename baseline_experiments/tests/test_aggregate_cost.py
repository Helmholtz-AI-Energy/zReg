"""Tests for aggregate_cost.py: --calibration, --configs-dir, --budget-hours, verdict placement.

Covers BUDG-02 (budget gate) and BUDG-03 (calibration loading).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# conftest.py inserts baseline_experiments/scripts into sys.path.
import aggregate_cost

# ── helpers ──────────────────────────────────────────────────────────────────


def _fresh_module():
    """Reload aggregate_cost to reset all module-level mutable state."""
    return importlib.reload(aggregate_cost)


def _calfile(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(content))
    return p


# ── calibration loading ───────────────────────────────────────────────────────


def test_calibration_loads_cpd_override(tmp_path):
    """--calibration JSON with cpd_trial_seconds overrides CPD_TRIAL_SECONDS."""
    cal_path = _calfile(tmp_path, {"cpd_trial_seconds": {"5": 999.0}})
    mod = _fresh_module()
    # Call main; it touches the filesystem for configs — use real configs dir,
    # which exists in the repo. Runs will be "pending" (no experiments/ data),
    # so no FileNotFoundError is expected from configs.
    try:
        mod.main(["--calibration", str(cal_path)])
    except SystemExit:
        pass  # budget gate exit is fine; we only care about state mutation

    assert mod.CPD_TRIAL_SECONDS[5] == 999.0, (
        f"CPD_TRIAL_SECONDS[5] should be 999.0 after calibration load, got {mod.CPD_TRIAL_SECONDS[5]}"
    )
    # Reload to not pollute other tests.
    _fresh_module()


def test_calibration_missing_key_uses_default(tmp_path):
    """--calibration JSON with only no_cpd_trial_seconds leaves CPD defaults intact."""
    cal_path = _calfile(tmp_path, {"no_cpd_trial_seconds": 1.5})
    mod = _fresh_module()
    original_cpd5 = mod.CPD_TRIAL_SECONDS[5]  # 130.5

    try:
        mod.main(["--calibration", str(cal_path)])
    except SystemExit:
        pass

    assert mod.NO_CPD_TRIAL_SECONDS == 1.5, (
        f"NO_CPD_TRIAL_SECONDS should be 1.5, got {mod.NO_CPD_TRIAL_SECONDS}"
    )
    assert mod.CPD_TRIAL_SECONDS[5] == original_cpd5, (
        f"CPD_TRIAL_SECONDS[5] should remain {original_cpd5}, got {mod.CPD_TRIAL_SECONDS[5]}"
    )
    _fresh_module()


def test_calibration_dict_form_accepted(tmp_path):
    """--calibration JSON with dict-form cpd_trial_seconds (from extract_calibration.py) is loaded correctly."""
    cal_path = _calfile(tmp_path, {
        "cpd_trial_seconds": {
            "5": {"seconds": 111.1, "extrapolated": False},
            "10": {"seconds": 222.2, "extrapolated": False},
            "20": {"seconds": 333.3, "extrapolated": True},
        }
    })
    mod = _fresh_module()

    try:
        mod.main(["--calibration", str(cal_path)])
    except SystemExit:
        pass

    assert abs(mod.CPD_TRIAL_SECONDS[5] - 111.1) < 0.01, (
        f"Dict-form value for window_size=5 not loaded correctly: {mod.CPD_TRIAL_SECONDS[5]}"
    )
    assert abs(mod.CPD_TRIAL_SECONDS[20] - 333.3) < 0.01, (
        f"Dict-form value for window_size=20 not loaded correctly: {mod.CPD_TRIAL_SECONDS[20]}"
    )
    _fresh_module()


# ── --configs-dir skip behaviour (D-07) ─────────────────────────────────────


def test_configs_dir_skip_missing(tmp_path, capsys):
    """--configs-dir pointing to an empty dir prints skip lines and doesn't crash."""
    # Create an empty configs dir — no YAML files exist.
    empty_configs = tmp_path / "empty_configs"
    empty_configs.mkdir()

    mod = _fresh_module()
    try:
        mod.main(["--configs-dir", str(empty_configs)])
    except SystemExit:
        pass  # budget gate may exit 1 since no runs have actual data

    captured = capsys.readouterr()
    # All runs should print "not found" / "excluded" since no YAMLs exist.
    assert "not found" in captured.out or "excluded" in captured.out, (
        f"Expected skip message in stdout but got: {captured.out!r}"
    )
    _fresh_module()


# ── budget gate (D-11/D-12/D-13) ────────────────────────────────────────────


def test_budget_gate_exit_0(tmp_path):
    """--budget-hours 999 exits 0 (well within cap)."""
    mod = _fresh_module()
    result = None
    try:
        result = mod.main(["--budget-hours", "999"])
    except SystemExit as exc:
        pytest.fail(f"main() should not exit with budget-hours=999 but got SystemExit({exc.code})")

    assert result == 0, f"main() should return 0 for budget-hours=999, got {result}"
    _fresh_module()


def test_budget_gate_exit_1(tmp_path):
    """--budget-hours 0.0001 exits 1 (exceeds cap)."""
    mod = _fresh_module()
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["--budget-hours", "0.0001"])

    assert exc_info.value.code == 1, (
        f"Expected SystemExit(1) for budget-hours=0.0001, got SystemExit({exc_info.value.code})"
    )
    _fresh_module()


def test_budget_gate_writes_markdown_before_exit(tmp_path, monkeypatch):
    """compute_cost.md is written even when budget gate triggers exit 1."""
    mod = _fresh_module()
    monkeypatch.setattr(mod, "EXPERIMENTS_ROOT", tmp_path)
    out_path = tmp_path / "summary" / "compute_cost.md"

    try:
        mod.main(["--budget-hours", "0.0001"])
    except SystemExit:
        pass

    assert out_path.exists(), (
        f"compute_cost.md should have been written before exit 1, but {out_path} does not exist"
    )
    _fresh_module()


def test_verdict_line_at_top(tmp_path, monkeypatch):
    """compute_cost.md first line starts with the verdict (WITHIN or EXCEEDS)."""
    mod = _fresh_module()
    monkeypatch.setattr(mod, "EXPERIMENTS_ROOT", tmp_path)
    out_path = tmp_path / "summary" / "compute_cost.md"

    try:
        mod.main(["--budget-hours", "999"])
    except SystemExit:
        pass

    assert out_path.exists(), f"compute_cost.md not created at {out_path}"
    first_line = out_path.read_text().splitlines()[0]
    assert first_line.startswith("WITHIN") or first_line.startswith("EXCEEDS"), (
        f"First line of compute_cost.md should start with WITHIN or EXCEEDS, got: {first_line!r}"
    )
    _fresh_module()


# ── CR-01 regression: null calibration values must not crash arithmetic ───────


def test_calibration_null_cpd_seconds_does_not_crash(tmp_path, monkeypatch):
    """Calibration with null seconds values (no CPD trials seen) must not crash.

    Regression guard for CR-01: a legacy calibration file (written before
    extract_calibration.py was fixed to omit null entries) may still contain
    {"seconds": null, "extrapolated": true} dict values. aggregate_cost.py
    must silently skip those entries rather than propagating None into
    _avg_trial_seconds() arithmetic.
    """
    cal_path = _calfile(tmp_path, {
        "cpd_trial_seconds": {
            "5": {"seconds": None, "extrapolated": True},
        },
        "no_cpd_trial_seconds": 4.0,
    })
    mod = _fresh_module()
    monkeypatch.setattr(mod, "EXPERIMENTS_ROOT", tmp_path)
    try:
        mod.main(["--calibration", str(cal_path)])
    except SystemExit:
        pass
    # Should not raise TypeError — the null entry must be skipped.
    _fresh_module()


def test_calibration_null_no_cpd_seconds_does_not_crash(tmp_path, monkeypatch):
    """Calibration with null no_cpd_trial_seconds must not crash.

    Regression guard for CR-01: when no no-CPD trials were observed,
    extract_calibration.py emits "no_cpd_trial_seconds": null. aggregate_cost.py
    must skip the assignment rather than storing None into NO_CPD_TRIAL_SECONDS
    and crashing at (1 - frac_cpd) * None.
    """
    cal_path = _calfile(tmp_path, {"no_cpd_trial_seconds": None})
    mod = _fresh_module()
    monkeypatch.setattr(mod, "EXPERIMENTS_ROOT", tmp_path)
    try:
        mod.main(["--calibration", str(cal_path)])
    except SystemExit:
        pass
    # Should not raise TypeError — the null value must be ignored.
    _fresh_module()
