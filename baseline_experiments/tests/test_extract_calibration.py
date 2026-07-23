"""Tests for extract_calibration.py: grouping, means, interpolation, multi-file, errors.

Covers BUDG-02 test coverage for extract_calibration.py output reliability.
Uses the `extract()` callable directly for fast unit tests, plus subprocess
for the exit-code / stdout-as-JSON contract tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# conftest.py inserts baseline_experiments/scripts into sys.path.
from extract_calibration import extract


# ── fixtures / helpers ────────────────────────────────────────────────────────


def _write_history(tmp_path: Path, name: str, trials: list[dict]) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(trials))
    return p


def _cpd_trial(window_size: int, duration: float, penalty: str = "rigid") -> dict:
    return {"params": {"window_size": window_size, "cpd_penalty": penalty}, "duration_seconds": duration}


def _no_cpd_trial(duration: float) -> dict:
    return {"params": {"cpd_penalty": None}, "duration_seconds": duration}


# ── grouping and extrapolation ────────────────────────────────────────────────


def test_two_window_sizes_extrapolates_third(tmp_path):
    """window_size 5 and 10 observed → 20 extrapolated."""
    history = [
        _cpd_trial(5, 120.0),
        _cpd_trial(5, 140.0),
        _cpd_trial(10, 160.0),
        _cpd_trial(10, 180.0),
    ]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])
    cpd = output["cpd_trial_seconds"]

    assert set(cpd.keys()) >= {"5", "10", "20"}, f"Expected all three keys, got {set(cpd.keys())}"
    assert cpd["20"]["extrapolated"] is True, "window_size=20 must be marked extrapolated"
    assert cpd["5"]["extrapolated"] is False, "window_size=5 must not be extrapolated"
    assert cpd["10"]["extrapolated"] is False, "window_size=10 must not be extrapolated"


def test_all_window_sizes_no_extrapolation(tmp_path):
    """All three window sizes observed → none extrapolated."""
    history = [
        _cpd_trial(5, 130.0),
        _cpd_trial(10, 176.0),
        _cpd_trial(20, 278.0),
    ]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])
    cpd = output["cpd_trial_seconds"]

    assert cpd["5"]["extrapolated"] is False
    assert cpd["10"]["extrapolated"] is False
    assert cpd["20"]["extrapolated"] is False


def test_mean_computation(tmp_path):
    """Three window_size=5 trials with durations 100, 120, 140 → mean 120.0."""
    history = [
        _cpd_trial(5, 100.0),
        _cpd_trial(5, 120.0),
        _cpd_trial(5, 140.0),
    ]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])
    cpd = output["cpd_trial_seconds"]

    assert abs(cpd["5"]["seconds"] - 120.0) < 0.01, (
        f"Expected mean 120.0 for window_size=5, got {cpd['5']['seconds']}"
    )


# ── no-CPD trials separation ──────────────────────────────────────────────────


def test_no_cpd_trials_excluded_from_cpd_timing(tmp_path):
    """No-CPD trials must NOT bleed into CPD window_size=10 mean."""
    history = [
        _cpd_trial(10, 200.0),
        _cpd_trial(10, 200.0),
        _no_cpd_trial(3.0),
    ]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])
    cpd = output["cpd_trial_seconds"]

    assert abs(cpd["10"]["seconds"] - 200.0) < 0.01, (
        f"No-CPD trial should not contaminate CPD mean: got {cpd['10']['seconds']}"
    )


def test_no_cpd_mean(tmp_path):
    """Two no-CPD trials (3.0s, 5.0s) → no_cpd_trial_seconds ≈ 4.0."""
    history = [
        _no_cpd_trial(3.0),
        _no_cpd_trial(5.0),
    ]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])

    assert output["no_cpd_trial_seconds"] is not None, "no_cpd_trial_seconds should not be null"
    assert abs(output["no_cpd_trial_seconds"] - 4.0) < 0.01, (
        f"Expected no_cpd_trial_seconds ≈ 4.0, got {output['no_cpd_trial_seconds']}"
    )


def test_no_cpd_null_when_no_no_cpd_trials(tmp_path):
    """no_cpd_trial_seconds should be null when no no-CPD trials appear."""
    history = [_cpd_trial(10, 180.0)]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])

    assert output["no_cpd_trial_seconds"] is None, (
        f"Expected null for no_cpd_trial_seconds, got {output['no_cpd_trial_seconds']}"
    )


# ── multi-file merge ──────────────────────────────────────────────────────────


def test_two_files_merged(tmp_path):
    """Trials split across two files produce the same result as one merged file."""
    all_trials = [
        _cpd_trial(5, 100.0),
        _cpd_trial(5, 140.0),
        _cpd_trial(10, 160.0),
    ]
    # Merged reference
    merged = _write_history(tmp_path, "merged.json", all_trials)
    out_merged = extract([str(merged)])

    # Split across two files
    f1 = _write_history(tmp_path, "f1.json", all_trials[:2])
    f2 = _write_history(tmp_path, "f2.json", all_trials[2:])
    out_split = extract([str(f1), str(f2)])

    assert abs(out_merged["cpd_trial_seconds"]["5"]["seconds"] - out_split["cpd_trial_seconds"]["5"]["seconds"]) < 0.01
    assert abs(out_merged["cpd_trial_seconds"]["10"]["seconds"] - out_split["cpd_trial_seconds"]["10"]["seconds"]) < 0.01


# ── field name acceptance ─────────────────────────────────────────────────────


def test_elapsed_seconds_field_accepted(tmp_path):
    """Trials using 'elapsed_seconds' instead of 'duration_seconds' are accepted."""
    history = [{"params": {"window_size": 5, "cpd_penalty": "rigid"}, "elapsed_seconds": 130.0}]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])

    assert abs(output["cpd_trial_seconds"]["5"]["seconds"] - 130.0) < 0.01, (
        f"elapsed_seconds not accepted: got {output['cpd_trial_seconds']['5']['seconds']}"
    )


def test_missing_duration_field_skipped(tmp_path, capsys):
    """Trial with no duration field is skipped without crashing (T-54-06)."""
    history = [
        {"params": {"window_size": 5, "cpd_penalty": "rigid"}},  # no duration field
        _cpd_trial(10, 160.0),
    ]
    f = _write_history(tmp_path, "h.json", history)
    # Should not raise; skips the malformed trial and processes the valid one.
    output = extract([str(f)])

    assert "10" in output["cpd_trial_seconds"], "Valid trial should still be processed"


# ── output format ─────────────────────────────────────────────────────────────


def test_output_json_schema(tmp_path):
    """Output JSON has the required top-level keys and cpd_trial_seconds dict form."""
    history = [_cpd_trial(5, 130.0)]
    f = _write_history(tmp_path, "h.json", history)
    output = extract([str(f)])

    assert "cpd_trial_seconds" in output
    assert "no_cpd_trial_seconds" in output
    assert "label_transfer_overhead_seconds" in output
    assert "_note" in output
    assert "_source" in output

    # cpd values must always be dict form
    for key in ["5", "10", "20"]:
        val = output["cpd_trial_seconds"][key]
        assert isinstance(val, dict), f"cpd_trial_seconds['{key}'] must be a dict, got {type(val)}"
        assert "seconds" in val and "extrapolated" in val, f"Missing 'seconds'/'extrapolated' in {val}"

    # label_transfer_overhead_seconds must be null (not computable from history)
    assert output["label_transfer_overhead_seconds"] is None


# ── exit code behaviour (via subprocess) ─────────────────────────────────────


def test_empty_no_readable_trials_exits_1(tmp_path):
    """Empty JSON array → exit code 1."""
    f = _write_history(tmp_path, "empty.json", [])
    result = subprocess.run(
        [sys.executable, "-m", "extract_calibration", str(f)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1] / "scripts"),
    )
    # subprocess -m extract_calibration may not work; fall back to direct path
    if result.returncode not in (0, 1):
        script = Path(__file__).resolve().parents[1] / "scripts" / "extract_calibration.py"
        result = subprocess.run(
            [sys.executable, str(script), str(f)],
            capture_output=True, text=True,
        )
    assert result.returncode == 1, (
        f"Expected exit 1 for empty trial list, got {result.returncode}. "
        f"stderr: {result.stderr!r}"
    )


def test_nonexistent_file_exits_1(tmp_path):
    """Nonexistent file path → exit code 1 with error on stderr."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "extract_calibration.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "does_not_exist.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, f"Expected exit 1, got {result.returncode}"
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower(), (
        f"Expected error message on stderr, got: {result.stderr!r}"
    )


def test_stdout_is_valid_json_and_pasteable(tmp_path):
    """Script stdout is valid JSON parseable by json.loads (ready for horeka.json)."""
    history = [_cpd_trial(5, 120.0), _cpd_trial(10, 160.0)]
    f = _write_history(tmp_path, "h.json", history)
    script = Path(__file__).resolve().parents[1] / "scripts" / "extract_calibration.py"
    result = subprocess.run(
        [sys.executable, str(script), str(f)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}. stderr: {result.stderr!r}"
    parsed = json.loads(result.stdout)
    assert "cpd_trial_seconds" in parsed, "Output JSON must contain cpd_trial_seconds"
