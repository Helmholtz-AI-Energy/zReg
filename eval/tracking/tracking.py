"""Per-run experiment metadata writer for the zReg evaluation framework.

Writes one JSON file and one single-row CSV file per call to ``log_run()``.
Both files are persisted to ``output_dir`` (default ``experiments/runs/``)
using stdlib modules only — no third-party dependencies are required.
Auto-captured fields (``git_hash``, ``zreg_version``, ``timestamp``) are
resolved at call time and silently fall back to ``"unknown"`` when the
corresponding subprocess or package-metadata lookup fails.
"""

import csv
import datetime
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

__all__ = ["log_run"]


def _validate_run_id(run_id: str) -> None:
    """Validate that *run_id* is safe to use as a filename stem.

    Raises
    ------
    ValueError
        If *run_id* is empty, contains a path separator, or starts with
        ``".."`` or ``"/"``.
    """
    if not run_id:
        raise ValueError("run_id must be a non-empty string")
    if os.sep in run_id or (os.altsep and os.altsep in run_id):
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")
    if run_id.startswith("..") or run_id.startswith("/"):
        raise ValueError(f"run_id must not begin with '..' or '/': {run_id!r}")


def log_run(
    run_id: str,
    dataset_path,
    frame_indices,
    seed,
    n_points_before: int,
    n_points_after: int,
    output_dir: str = "experiments/runs",
    **extra_fields,
) -> str:
    """Persist all 9 EVAL-04 required fields for a single experiment run.

    Writes two files to *output_dir*:

    - ``{run_id}.json`` — all 9 required fields plus any *extra_fields* as a
      JSON object.
    - ``{run_id}.csv`` — header row followed by one data row with the same
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
        output filenames (``{run_id}.json``, ``{run_id}.csv``).  Must be a
        non-empty string that does not contain path separators or begin with
        ``".."`` or ``"/"``.
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
        it does not exist.  Default is ``"experiments/runs"``.
    **extra_fields
        Any additional keyword arguments (e.g. ``chamfer=0.003``,
        ``hausdorff=0.012``) are appended to the record after the 9 required
        fields.  Keys must not collide with the 9 required field names or the
        three auto-captured names (``git_hash``, ``zreg_version``,
        ``timestamp``).

    Returns
    -------
    run_id : str
        The same ``run_id`` string that was passed in, so callers can
        reference the logged entry (e.g. for sweep correlation).

    Raises
    ------
    ValueError
        If ``run_id`` is empty, contains a path separator, or begins with
        ``".."`` or ``"/"``.
    FileExistsError
        If a file named ``{run_id}.json`` or ``{run_id}.csv`` already exists
        in ``output_dir``.  Use a unique ``run_id`` or remove the existing
        files before calling this function.
    OSError
        If ``output_dir`` cannot be created or the output files cannot be
        written (e.g. permission denied, disk full).  No exception is raised
        for ``git_hash`` or ``zreg_version`` lookup failures — those silently
        fall back to ``"unknown"``.
    """
    # --- validate run_id to prevent path traversal ---
    _validate_run_id(run_id)

    # --- auto-capture git_hash ---
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
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

    # --- build record (field order: 9 EVAL-04 required fields, then extras) ---
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
        **extra_fields,
    }

    # --- ensure output directory exists ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- guard against run_id reuse ---
    json_path = Path(output_dir) / f"{run_id}.json"
    csv_path = Path(output_dir) / f"{run_id}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            f"Run ID {run_id!r} already exists in {output_dir!r}. "
            "Use a unique run_id or remove the existing files."
        )

    # --- write JSON ---
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    # --- write CSV ---
    csv_record = {**record, "frame_indices": str(frame_indices)}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(csv_record)

    return run_id
