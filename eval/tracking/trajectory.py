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
    target: dict | None = None,
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
        ``LabelResult`` from ``eval.types``).  May also carry
        ``"label_receiver"`` (as returned by ``EvaluationRunner._run_single``),
        which is then the position source for the label CSV.
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
    The ``label_trajectory.csv`` position source (Phase 59 D-01): positions
    come from ``result["label_receiver"][frame_idx]["pos"]`` when the result
    carries a label receiver (the trajectory the labels were transferred
    onto — the aligned source when ``config.label_source == "target"``, else
    the target).  Without a receiver (legacy callers) the chain is
    ``target`` -> ``result["align"].aligned_cloud`` -> ``dataset``.  A length
    mismatch between positions and transferred labels raises ``ValueError``.

    ``label_metadata.json`` describes the receiver frames: ``frame_count`` /
    ``frame_indices`` are the ``transferred_labels`` keys (not the original
    ``dataset``'s frames), and ``label_source`` plus the provider and
    receiver data paths record the transfer direction.

    A single UUID ``run_id`` is generated once per call; both metadata files
    share the same UUID when both stages ran.

    When *dataset* is empty (``{}``), the write loops never execute and the
    resulting CSV files contain only the header row (no data rows).  The
    function still returns the path list and reports ``frame_count=0`` in the
    metadata JSON — this is intentional and not treated as an error.
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

    # 4a. Extract stage results safely — guards against callers that omit a key
    align_result_val = result.get("align")
    label_result_val = result.get("label")

    # 5. Align stage output
    if align_result_val is not None:
        align_result = align_result_val

        # 5a. Write align_trajectory.csv
        # aligned_cloud is keyed by target frame indices in paired mode, so
        # iterate its own keys rather than source dataset keys.
        align_cloud_keys = sorted(align_result.aligned_cloud.keys())
        align_csv_path = output_dir / "align_trajectory.csv"
        with open(align_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "point_idx", "x", "y", "z"])
            for frame_idx in align_cloud_keys:
                pos = align_result.aligned_cloud[frame_idx]["pos"]
                for i in range(len(pos)):
                    x, y, z = pos[i].tolist()
                    writer.writerow([frame_idx, i, x, y, z])

        # 5b. Build align metadata
        align_meta: dict[str, Any] = {
            "run_id": run_id,
            "frame_count": len(align_cloud_keys),
            "frame_indices": align_cloud_keys,
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
    if label_result_val is not None:
        label_result = label_result_val

        # 6a. Write label_trajectory.csv
        label_csv_path = output_dir / "label_trajectory.csv"
        with open(label_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "point_idx", "x", "y", "z", "label"])
            # Iterate transferred_labels keys (receiver space) so positions and labels
            # come from the same dataset. The receiver (Phase 59 D-01) is the aligned
            # source when label_source == "target", else the target; reading it first
            # keeps positions and labels consistent in both directions.
            label_receiver = result.get("label_receiver")
            for frame_idx in sorted(label_result.transferred_labels.keys()):
                labels = label_result.transferred_labels[frame_idx]
                if label_receiver is not None and frame_idx in label_receiver:
                    pos = label_receiver[frame_idx]["pos"]
                elif target is not None and frame_idx in target:
                    pos = target[frame_idx]["pos"]
                elif align_result_val is not None and frame_idx in align_result_val.aligned_cloud:
                    pos = align_result_val.aligned_cloud[frame_idx]["pos"]
                else:
                    pos = dataset[frame_idx]["pos"]
                if len(pos) != len(labels):
                    raise ValueError(
                        f"frame {frame_idx}: pos has {len(pos)} points but "
                        f"transferred_labels has {len(labels)} — pipeline state is inconsistent"
                    )
                for i in range(len(labels)):
                    x, y, z = pos[i].tolist()
                    label = int(labels[i].item())
                    writer.writerow([frame_idx, i, x, y, z, label])

        # 6b. Build label metadata — describes the receiver frames that carry the
        # transferred labels, plus the transfer direction (Phase 59 D-01).
        transferred_keys = sorted(label_result.transferred_labels.keys())
        if config.label_source == "target":
            provider_path = config.target_data_path
            receiver_path = config.data_path
        else:
            provider_path = config.data_path
            receiver_path = config.target_data_path or config.data_path
        label_meta: dict[str, Any] = {
            "run_id": run_id,
            "frame_count": len(transferred_keys),
            "frame_indices": transferred_keys,
            "label_source": config.label_source,
            "label_provider_data_path": provider_path,
            "label_receiver_data_path": receiver_path,
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
