"""Tests for eval.tracking.log_run().

Single test class TestLogRun covers all observable behaviours of log_run()
per decisions D-08 and D-09 in 15-CONTEXT.md:

- Return value contract
- File creation (JSON and CSV)
- All 9 EVAL-04 required fields present in both output files
- Caller-supplied field values round-trip correctly through JSON
- Auto-captured fields (git_hash, zreg_version) are injected via mock
- Fallback paths exercised (subprocess.run raises, PackageNotFoundError raised)
- Auto-created output_dir (including nested paths)
- CSV single data row
- frame_indices CSV serialisation as string

Tests use tmp_path (built-in pytest fixture) as output_dir so no
experiments/runs/ directory is created on disk during test runs.
"""

import csv
import importlib.metadata
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.tracking import log_run

# ---------------------------------------------------------------------------
# Module-level constants for caller args (reused across all tests)
# ---------------------------------------------------------------------------

RUN_ID = "run-001"
DATASET_PATH = "data/test.pkl"
FRAME_INDICES = [0, 1, 2]
SEED = 42
N_POINTS_BEFORE = 200
N_POINTS_AFTER = 190

EXPECTED_FIELDS = {
    "run_id",
    "dataset_path",
    "frame_indices",
    "seed",
    "n_points_before",
    "n_points_after",
    "git_hash",
    "zreg_version",
    "timestamp",
}

# Deterministic patch targets used in tests that only care about file structure
_MOCK_GIT_RESULT = MagicMock(stdout="deadbeef\n", returncode=0)


# ---------------------------------------------------------------------------
# TestLogRun
# ---------------------------------------------------------------------------


class TestLogRun:
    """Unit tests for log_run() covering all 9 EVAL-04 fields, fallback
    behaviour, return value, and file output."""

    # ------------------------------------------------------------------
    # Return value
    # ------------------------------------------------------------------

    def test_returns_run_id(self, tmp_path):
        """log_run() returns the exact run_id string passed by the caller."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            result = log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        assert result == RUN_ID

    # ------------------------------------------------------------------
    # File creation
    # ------------------------------------------------------------------

    def test_json_file_created(self, tmp_path):
        """After log_run(), the JSON file {run_id}.json exists in output_dir."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        assert (tmp_path / f"{RUN_ID}.json").exists()

    def test_csv_file_created(self, tmp_path):
        """After log_run(), the CSV file {run_id}.csv exists in output_dir."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        assert (tmp_path / f"{RUN_ID}.csv").exists()

    # ------------------------------------------------------------------
    # Field presence in JSON
    # ------------------------------------------------------------------

    def test_json_contains_all_nine_fields(self, tmp_path):
        """The parsed JSON dict has exactly the 9 EVAL-04 required keys."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.json") as f:
            data = json.load(f)
        assert set(data.keys()) == EXPECTED_FIELDS

    # ------------------------------------------------------------------
    # Field presence in CSV
    # ------------------------------------------------------------------

    def test_csv_contains_all_nine_fields(self, tmp_path):
        """The CSV header row contains exactly the 9 EVAL-04 field names."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.csv", newline="") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == EXPECTED_FIELDS

    # ------------------------------------------------------------------
    # CSV row count
    # ------------------------------------------------------------------

    def test_csv_has_one_data_row(self, tmp_path):
        """csv.DictReader on the output CSV yields exactly one row."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    # ------------------------------------------------------------------
    # Caller field round-trip
    # ------------------------------------------------------------------

    def test_json_caller_fields_correct(self, tmp_path):
        """JSON values for caller-supplied fields match what was passed in."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.json") as f:
            data = json.load(f)
        assert data["run_id"] == RUN_ID
        assert data["dataset_path"] == DATASET_PATH
        assert data["seed"] == SEED
        assert data["n_points_before"] == N_POINTS_BEFORE
        assert data["n_points_after"] == N_POINTS_AFTER

    # ------------------------------------------------------------------
    # git_hash auto-capture
    # ------------------------------------------------------------------

    def test_git_hash_captured(self, tmp_path):
        """When subprocess.run returns stdout='abc123\\n' and returncode=0,
        git_hash in JSON equals 'abc123'."""
        mock_result = MagicMock(stdout="abc123\n", returncode=0)
        with patch("eval.tracking.tracking.subprocess.run", return_value=mock_result), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.json") as f:
            data = json.load(f)
        assert data["git_hash"] == "abc123"

    def test_git_hash_fallback_on_exception(self, tmp_path):
        """When subprocess.run raises Exception, git_hash in JSON equals 'unknown'."""
        with patch("eval.tracking.tracking.subprocess.run", side_effect=Exception("git not found")), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.json") as f:
            data = json.load(f)
        assert data["git_hash"] == "unknown"

    # ------------------------------------------------------------------
    # zreg_version auto-capture
    # ------------------------------------------------------------------

    def test_zreg_version_captured(self, tmp_path):
        """When importlib.metadata.version is patched to return '1.2.0',
        zreg_version in JSON equals '1.2.0'."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="1.2.0"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.json") as f:
            data = json.load(f)
        assert data["zreg_version"] == "1.2.0"

    def test_zreg_version_fallback(self, tmp_path):
        """When importlib.metadata.version raises PackageNotFoundError,
        zreg_version in JSON equals 'unknown'."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch(
                 "eval.tracking.tracking.importlib.metadata.version",
                 side_effect=importlib.metadata.PackageNotFoundError("zreg"),
             ):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.json") as f:
            data = json.load(f)
        assert data["zreg_version"] == "unknown"

    # ------------------------------------------------------------------
    # Auto-created output_dir
    # ------------------------------------------------------------------

    def test_output_dir_autocreated(self, tmp_path):
        """log_run() creates a nested output_dir that does not yet exist."""
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(nested),
            )
        assert nested.exists()

    # ------------------------------------------------------------------
    # frame_indices CSV serialisation
    # ------------------------------------------------------------------

    def test_frame_indices_in_csv_as_string(self, tmp_path):
        """For frame_indices=[0, 1, 2], the CSV row value is the string '[0, 1, 2]'."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=[0, 1, 2],
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["frame_indices"] == "[0, 1, 2]"

    def test_git_hash_fallback_when_returncode_nonzero(self, tmp_path):
        """tracking.py:98 — git_hash='unknown' when subprocess returncode != 0."""
        mock_result = MagicMock(returncode=1, stdout="")
        with patch("eval.tracking.tracking.subprocess.run", return_value=mock_result), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with open(tmp_path / f"{RUN_ID}.json") as f:
            data = json.load(f)
        assert data["git_hash"] == "unknown"


# ---------------------------------------------------------------------------
# TestValidateRunId — coverage for lines 32, 34, 36, 163
# ---------------------------------------------------------------------------


class TestValidateRunId:
    """Unit tests for the _validate_run_id guard (called by log_run)."""

    def test_empty_run_id_raises(self, tmp_path):
        """tracking.py:32 — empty run_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            log_run(
                run_id="",
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )

    def test_run_id_with_path_separator_raises(self, tmp_path):
        """tracking.py:34 — run_id containing os.sep raises ValueError."""
        import os
        bad_id = f"dir{os.sep}run"
        with pytest.raises(ValueError, match="path separator"):
            log_run(
                run_id=bad_id,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )

    def test_run_id_starting_with_dotdot_raises(self, tmp_path):
        """tracking.py:36 — run_id beginning with '..' raises ValueError (no sep so line 34 skipped)."""
        with pytest.raises(ValueError):
            log_run(
                run_id="..escape",  # starts with '..' but no path separator → hits line 36
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )

    def test_run_id_starting_with_slash_raises(self, tmp_path):
        """tracking.py:34 — run_id beginning with '/' is caught as path separator."""
        with pytest.raises(ValueError):
            log_run(
                run_id="/absolute",
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )

    def test_duplicate_run_id_raises_file_exists_error(self, tmp_path):
        """tracking.py:163 — calling log_run twice with same run_id raises FileExistsError."""
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            log_run(
                run_id=RUN_ID,
                dataset_path=DATASET_PATH,
                frame_indices=FRAME_INDICES,
                seed=SEED,
                n_points_before=N_POINTS_BEFORE,
                n_points_after=N_POINTS_AFTER,
                output_dir=str(tmp_path),
            )
        with patch("eval.tracking.tracking.subprocess.run", return_value=_MOCK_GIT_RESULT), \
             patch("eval.tracking.tracking.importlib.metadata.version", return_value="0.0.1"):
            with pytest.raises(FileExistsError, match=RUN_ID):
                log_run(
                    run_id=RUN_ID,
                    dataset_path=DATASET_PATH,
                    frame_indices=FRAME_INDICES,
                    seed=SEED,
                    n_points_before=N_POINTS_BEFORE,
                    n_points_after=N_POINTS_AFTER,
                    output_dir=str(tmp_path),
                )
