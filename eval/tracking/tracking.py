"""Per-run experiment metadata writer for the zReg evaluation framework.

Writes one JSON file and one single-row CSV file per call to ``log_run()``.
Both files are persisted to ``output_dir`` (default ``evaluation/runs/``)
using stdlib modules only — no third-party dependencies are required.
Auto-captured fields (``git_hash``, ``zreg_version``, ``timestamp``) are
resolved at call time and silently fall back to ``"unknown"`` when the
corresponding subprocess or package-metadata lookup fails.
"""

import csv
import datetime
import importlib.metadata
import json
import subprocess
from pathlib import Path

__all__ = ["log_run"]


def log_run(
    run_id: str,
    dataset_path,
    frame_indices,
    seed,
    n_points_before: int,
    n_points_after: int,
    output_dir: str = "evaluation/runs",
) -> str:
    """Persist all 9 EVAL-04 required fields for a single experiment run.

    Writes two files to *output_dir*:

    - ``{run_id}.json`` — all 9 fields as a JSON object.
    - ``{run_id}.csv`` — header row followed by one data row with the same 9
      fields.

    Three fields are auto-captured internally:

    - ``git_hash``: current HEAD commit hash via ``subprocess.run``; falls
      back to ``"unknown"`` on any failure.
    - ``zreg_version``: installed package version via
      ``importlib.metadata.version``; falls back to ``"unknown"`` when the
      package is not found.
    - ``timestamp``: current UTC time as an ISO 8601 string.

    Parameters
    ----------
    run_id : str
        Caller-provided identifier for this run.  Used as the stem of the
        output filenames (``{run_id}.json``, ``{run_id}.csv``).
    dataset_path : str or Path
        Path to the dataset file used in this experiment.
    frame_indices : list of int
        Frame indices selected from the dataset for this run.
    seed : int
        Random seed used to initialise stochastic operations.
    n_points_before : int
        Number of points in each point cloud before any downsampling.
    n_points_after : int
        Number of points in each point cloud after downsampling.
    output_dir : str, optional
        Directory where output files are written.  Created automatically if
        it does not exist.  Default is ``"evaluation/runs"``.

    Returns
    -------
    run_id : str
        The same ``run_id`` string that was passed in, so callers can
        reference the logged entry (e.g. for sweep correlation).

    Raises
    ------
    OSError
        If ``output_dir`` cannot be created or the output files cannot be
        written (e.g. permission denied, disk full).  No exception is raised
        for ``git_hash`` or ``zreg_version`` lookup failures — those silently
        fall back to ``"unknown"``.
    """
    # --- auto-capture git_hash ---
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
        else:
            git_hash = "unknown"
    except Exception:
        git_hash = "unknown"

    # --- auto-capture zreg_version ---
    try:
        zreg_version = importlib.metadata.version("zreg")
    except importlib.metadata.PackageNotFoundError:
        zreg_version = "unknown"

    # --- auto-capture timestamp ---
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # --- build record (field order matches EVAL-04 spec) ---
    record = {
        "run_id": run_id,
        "dataset_path": dataset_path,
        "frame_indices": frame_indices,
        "seed": seed,
        "n_points_before": n_points_before,
        "n_points_after": n_points_after,
        "git_hash": git_hash,
        "zreg_version": zreg_version,
        "timestamp": timestamp,
    }

    # --- ensure output directory exists ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- write JSON ---
    json_path = Path(output_dir) / f"{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    # --- write CSV ---
    csv_path = Path(output_dir) / f"{run_id}.csv"
    csv_record = {**record, "frame_indices": str(frame_indices)}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(csv_record)

    return run_id
