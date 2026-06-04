"""Trajectory CSV and metadata JSON export for the zReg evaluation framework.

Provides ``export_trajectory``, which writes per-point-per-frame CSV files and
accompanying metadata JSON files to a caller-specified output directory.

All file I/O uses stdlib modules only — no third-party dependencies are
required at the module level.  Torch tensors are accessed exclusively via
``.tolist()`` and ``.item()`` so that torch need not be imported here.

Auto-captured fields (``git_hash``, ``zreg_version``, ``timestamp``) are
resolved at call time and silently fall back to ``"unknown"`` when the
corresponding subprocess or package-metadata lookup fails — identical
pattern to ``eval.tracking.tracking.log_run``.
"""

import csv
import datetime
import importlib.metadata
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from eval.config import EvalConfig

__all__ = ["export_trajectory"]


def export_trajectory(
    result: dict,
    dataset: dict,
    config: EvalConfig,
    output_dir: Path,
) -> list[str]:
    """Export per-point trajectory CSVs and metadata JSON files (EXT-01).

    Writes up to four files into *output_dir*, depending on which stages
    produced results:

    - ``align_trajectory.csv``  — 5-column, one row per point per frame
    - ``align_metadata.json``   — 11-field run metadata for the align stage
    - ``label_trajectory.csv``  — 6-column (adds ``label``), one row per point per frame
    - ``label_metadata.json``   — 11-field run metadata for the label stage

    When ``result["align"]`` is ``None`` the align files are skipped; when
    ``result["label"]`` is ``None`` the label files are skipped.

    Parameters
    ----------
    result : dict
        A dict with keys ``"align"`` and ``"label"``.  Each value is either
        ``None`` or the corresponding stage result object (``AlignResult`` /
        ``LabelResult`` from ``eval.types``).
    dataset : dict
        Raw dataset mapping ``frame_idx`` (int) to a ``zRegPointCloud``-like
        dict.  At minimum each value must contain a ``"pos"`` tensor of shape
        ``(N, 3)``.
    config : EvalConfig
        Evaluation configuration; used to populate metadata fields
        (``data_path``, ``tier``, ``n_trials``, ``n_synthetic``).
    output_dir : Path
        Directory where output files are written.  Created automatically
        (including parent directories) if it does not exist.

    Returns
    -------
    list[str]
        Absolute path strings of the files actually written, in order:
        ``[align_csv, align_meta, label_csv, label_meta]`` (absent stages
        are omitted, so the list length is 0, 2, or 4).

    Notes
    -----
    The ``label_trajectory.csv`` position source follows decision D-07/D-08:
    when ``result["align"]`` is not ``None``, ``x/y/z`` are taken from
    ``result["align"].aligned_cloud[frame_idx]["pos"]``; otherwise they
    are taken from ``dataset[frame_idx]["pos"]``.

    A single UUID ``run_id`` is generated once per call; both metadata files
    share the same UUID when both stages ran.
    """
    # 1. Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate single run_id for this call
    run_id = str(uuid.uuid4())

    # 3. Auto-capture block — verbatim copy of tracking.py lines 88-109
    # --- auto-capture git_hash ---
    try:
        _git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,  # seconds — fall through to "unknown" on TimeoutExpired
        )
        if _git_result.returncode == 0:
            git_hash = _git_result.stdout.strip()
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

    # 4. Initialise return path list
    paths: list[str] = []

    # 5. Align stage output
    if result["align"] is not None:
        align_result = result["align"]

        # 5a. Write align_trajectory.csv
        align_csv_path = output_dir / "align_trajectory.csv"
        with open(align_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "point_idx", "x", "y", "z"])
            for frame_idx in sorted(dataset.keys()):
                pos = align_result.aligned_cloud[frame_idx]["pos"]
                for i in range(len(pos)):
                    x, y, z = pos[i].tolist()
                    writer.writerow([frame_idx, i, x, y, z])

        # 5b. Build align metadata
        align_meta: dict[str, Any] = {
            "run_id": run_id,
            "frame_count": len(dataset),
            "frame_indices": sorted(dataset.keys()),
            "data_path": config.data_path,
            "params_used": align_result.params_used,
            "tier": config.tier,
            "n_trials": config.n_trials,
            "n_synthetic": config.n_synthetic,
            "git_hash": git_hash,
            "zreg_version": zreg_version,
            "timestamp": timestamp,
        }

        # 5c. Write align_metadata.json
        align_meta_path = output_dir / "align_metadata.json"
        with open(align_meta_path, "w") as f:
            json.dump(align_meta, f, indent=2, default=str)

        # 5d. Append paths
        paths.extend([str(align_csv_path), str(align_meta_path)])

    # 6. Label stage output
    if result["label"] is not None:
        label_result = result["label"]

        # 6a. Write label_trajectory.csv
        label_csv_path = output_dir / "label_trajectory.csv"
        with open(label_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "point_idx", "x", "y", "z", "label"])
            for frame_idx in sorted(dataset.keys()):
                # D-07: use aligned pos when align stage ran; D-08: else use raw dataset
                if result["align"] is not None:
                    pos = result["align"].aligned_cloud[frame_idx]["pos"]
                else:
                    pos = dataset[frame_idx]["pos"]
                labels = label_result.transferred_labels[frame_idx]
                if len(labels) != len(pos):
                    raise ValueError(
                        f"frame {frame_idx}: transferred_labels length {len(labels)} "
                        f"!= pos length {len(pos)}"
                    )
                for i in range(len(pos)):
                    x, y, z = pos[i].tolist()
                    label = int(labels[i].item())
                    writer.writerow([frame_idx, i, x, y, z, label])

        # 6b. Build label metadata
        label_meta: dict[str, Any] = {
            "run_id": run_id,
            "frame_count": len(dataset),
            "frame_indices": sorted(dataset.keys()),
            "data_path": config.data_path,
            "params_used": label_result.params_used,
            "tier": config.tier,
            "n_trials": config.n_trials,
            "n_synthetic": config.n_synthetic,
            "git_hash": git_hash,
            "zreg_version": zreg_version,
            "timestamp": timestamp,
        }

        # 6c. Write label_metadata.json
        label_meta_path = output_dir / "label_metadata.json"
        with open(label_meta_path, "w") as f:
            json.dump(label_meta, f, indent=2, default=str)

        # 6d. Append paths
        paths.extend([str(label_csv_path), str(label_meta_path)])

    # 7. Return list of written paths
    return paths
