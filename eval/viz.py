"""Matplotlib rendering helpers for FRAME-08.

Two public functions produce figures:

- ``plot_trajectory`` — two independent 1×3 figure pairs (alignment_trajectory
  and label_trajectory) written only when the corresponding stage result is
  provided. Returns a list of written path strings.
- ``plot_metrics`` — horizontal bar chart of 6 normalised metrics per D-09.

Notes
-----
1. ``matplotlib.use("Agg")`` is called once at module import before
   ``import matplotlib.pyplot as plt``.  This permanently sets the Agg
   (non-interactive) backend for the process, which is CI-safe and requires
   no display connection.  All public functions in this module write figures
   to disk only; interactive display is never needed.
2. ``plt.close(fig)`` is mandatory after every ``fig.savefig`` call
   (RESEARCH Matplotlib Agg Patterns).  Without it, matplotlib accumulates
   open figure handles that are never freed.
3. ``bbox_inches="tight"`` is enforced on every ``fig.savefig`` call per
   FRAME-08 mandatory rules.
4. PDF + PNG output; both files are written per figure.
5. mathtext only — no system TeX dependencies.
"""

from pathlib import Path
from typing import Union

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
from zreg.dataset import zRegPointCloud  # noqa: F401 — ensures import order

import torch  # noqa: F401 — must follow zreg.* (libomp SIGABRT rule)

import numpy as np

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import — file-export module only
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from eval.types import AlignResult, EvalReport, LabelResult

__all__ = ["plot_trajectory", "plot_metrics", "render_dataset_triptych"]


def plot_trajectory(
    align_result: "AlignResult | None",
    label_result: "LabelResult | None",
    dataset: "dict[int, zRegPointCloud]",
    label_names: "dict[int, str] | None",
    output_dir: Path,
    target: "dict[int, zRegPointCloud] | None" = None,
) -> list[str]:
    """Render trajectory figures for alignment and/or label-transfer stages (EXT-02).

    Produces up to two independent 1×3 figure pairs — each pair is a PDF and
    a PNG — depending on which stage results are provided:

    - ``alignment_trajectory.pdf`` / ``alignment_trajectory.png`` — written when
      ``align_result is not None``.  Each of the 3 subplots superimposes the
      pre-alignment source frame (blue) and the aligned frame (orange).
    - ``label_trajectory.pdf`` / ``label_trajectory.png`` — written when
      ``label_result is not None``.  Each of the 3 subplots shows the point
      cloud coloured by transferred label ID.

    When both inputs are ``None``, the function returns ``[]`` and writes no
    files (D-03).

    Parameters
    ----------
    align_result : AlignResult or None
        Alignment result carrying ``aligned_cloud: dict[int, zRegPointCloud]``.
        Pass ``None`` to skip the alignment figure.
    label_result : LabelResult or None
        Label-transfer result carrying ``transferred_labels: dict[int, torch.Tensor]``.
        Pass ``None`` to skip the label figure.
    dataset : dict[int, zRegPointCloud]
        Raw dataset mapping integer frame indices to ``zRegPointCloud``
        dicts.  Must contain ``pos`` tensors of shape ``(N, 3)``.
    label_names : dict[int, str] or None
        Optional mapping of integer label IDs to descriptive names (D-06).
        Used for legend labels in the label-trajectory figure.  When ``None``,
        raw integer IDs are rendered as strings.
    output_dir : Path
        Directory where output files are written.  The directory must already
        exist; this function does NOT call ``mkdir``.

    Returns
    -------
    list[str]
        Absolute path strings of the files actually written, in order:
        ``[alignment_trajectory.pdf, alignment_trajectory.png,
           label_trajectory.pdf, label_trajectory.png]``
        (absent stages are omitted, so the list length is 0, 2, or 4).

    Notes
    -----
    D-04: Each figure uses first, middle, and last frame from
    ``sorted(dataset.keys())``, mirroring the ``export_trajectory`` frame
    selection pattern from Phase 24.

    D-07: Positions for the label figure use ``align_result.aligned_cloud``
    when ``align_result is not None`` (post-alignment coordinate space,
    consistent with label assignment); otherwise ``dataset[frame]["pos"]`` is
    used (label-only run).

    FRAME-08 mandatory rules apply: ``plt.close(fig)`` after every
    ``fig.savefig``, ``bbox_inches="tight"`` on every savefig call.
    """
    paths: list[str] = []

    # D-04: first / middle / last frame selection
    sorted_keys = sorted(dataset.keys())
    if not sorted_keys:
        return paths  # no frames to plot
    candidates = [sorted_keys[0], sorted_keys[len(sorted_keys) // 2], sorted_keys[-1]]
    # Preserve order but deduplicate (handles 1- and 2-frame datasets)
    seen: set = set()
    frame_indices = [k for k in candidates if not (k in seen or seen.add(k))]

    # ------------------------------------------------------------------
    # ALIGNMENT FIGURE — written only when align_result is not None
    # ------------------------------------------------------------------
    if align_result is not None:
        # aligned_cloud is keyed by target frame indices, which differ from
        # source keys in paired mode. Derive frame selection from aligned_cloud
        # so both lookups (aligned_cloud[fk] and dataset[fk]) are valid.
        align_sorted = sorted(align_result.aligned_cloud.keys())
        align_candidates = [
            align_sorted[0],
            align_sorted[len(align_sorted) // 2],
            align_sorted[-1],
        ]
        seen_a: set = set()
        align_frame_indices = [k for k in align_candidates if not (k in seen_a or seen_a.add(k))]
        fig = plt.figure(figsize=(12, 4))
        source_h = None
        aligned_h = None
        for idx, fk in enumerate(align_frame_indices):
            ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
            source_pos = dataset[fk]["pos"].detach().cpu().numpy()
            aligned_pos = align_result.aligned_cloud[fk]["pos"].detach().cpu().numpy()
            # Per-cloud subsampling — each cloud gets its own default_rng(0) instance (D-01)
            rng = np.random.default_rng(0)
            if len(source_pos) > 4000:
                source_pos = source_pos[rng.choice(len(source_pos), 4000, replace=False)]
            rng = np.random.default_rng(0)
            if len(aligned_pos) > 4000:
                aligned_pos = aligned_pos[rng.choice(len(aligned_pos), 4000, replace=False)]
            s = ax.scatter(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2],
                           c="blue", s=1.5, alpha=1.0, linewidths=0)
            a = ax.scatter(aligned_pos[:, 0], aligned_pos[:, 1], aligned_pos[:, 2],
                           c="red", s=1.5, alpha=1.0, linewidths=0)
            if idx == 0:
                source_h, aligned_h = s, a
            ax.set_title(f"Frame {fk}", fontsize=9, pad=4)
            for lbl in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
                lbl.set_fontsize(6)
            ax.set_xlabel("x", fontsize=7, labelpad=2)
            ax.set_ylabel("y", fontsize=7, labelpad=2)
            ax.set_zlabel("z", fontsize=7, labelpad=2)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        fig.legend([source_h, aligned_h], ["Source", "Aligned"], loc="center right", bbox_to_anchor=(1.12, 0.5))
        base = Path(output_dir) / "alignment_trajectory"
        try:
            fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        paths.extend([str(base.with_suffix(".pdf")), str(base.with_suffix(".png"))])

    # ------------------------------------------------------------------
    # LABEL FIGURE — written only when label_result is not None
    # ------------------------------------------------------------------
    if label_result is not None:
        # Frame indices for the label figure come from transferred_labels keys
        # (target space) so positions and labels stay in the same dataset.
        label_sorted_keys = sorted(label_result.transferred_labels.keys())
        label_candidates = [
            label_sorted_keys[0],
            label_sorted_keys[len(label_sorted_keys) // 2],
            label_sorted_keys[-1],
        ]
        seen_lbl: set = set()
        label_frame_indices = [
            k for k in label_candidates if not (k in seen_lbl or seen_lbl.add(k))
        ]

        # Collect union of label IDs across the 3 plotted frames
        union_labels: set[int] = set()
        for fk in label_frame_indices:
            t = label_result.transferred_labels[fk]
            union_labels.update(int(v) for v in torch.unique(t).tolist())
        sorted_labels = sorted(union_labels)

        _label_palette = ["red", "green", "blue"]
        color_for_label = {lab: _label_palette[i % 3] for i, lab in enumerate(sorted_labels)}

        fig = plt.figure(figsize=(12, 4))
        for idx, fk in enumerate(label_frame_indices):
            ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
            # Use target positions when available so they match transferred_labels
            # (which are keyed by target frame). Fall back to aligned source or
            # raw source for label-only or same-dataset runs.
            if target is not None and fk in target:
                pos = target[fk]["pos"].detach().cpu().numpy()
            elif align_result is not None and fk in align_result.aligned_cloud:
                pos = align_result.aligned_cloud[fk]["pos"].detach().cpu().numpy()
            else:
                pos = dataset[fk]["pos"].detach().cpu().numpy()
            t = label_result.transferred_labels[fk]
            c_vals = [color_for_label[int(v)] for v in t.tolist()]
            # Per-cloud subsampling — subsample pos and c_vals in sync (D-03)
            n_pts = min(len(pos), len(c_vals))
            rng = np.random.default_rng(0)
            if n_pts > 4000:
                keep = rng.choice(n_pts, 4000, replace=False)
                pos = pos[keep]
                c_vals = [c_vals[i] for i in keep]
            else:
                pos = pos[:n_pts]
                c_vals = c_vals[:n_pts]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=c_vals, s=1.5, alpha=1.0, linewidths=0)
            ax.set_title(f"Frame {fk}", fontsize=9, pad=4)
            for lbl in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
                lbl.set_fontsize(6)
            ax.set_xlabel("x", fontsize=7, labelpad=2)
            ax.set_ylabel("y", fontsize=7, labelpad=2)
            ax.set_zlabel("z", fontsize=7, labelpad=2)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        # Build legend patches
        patches = [
            mpatches.Patch(
                color=color_for_label[lab],
                label=(label_names.get(lab, str(lab)) if label_names else str(lab)),
            )
            for lab in sorted_labels
        ]
        fig.legend(handles=patches, loc="center right", bbox_to_anchor=(1.15, 0.5))
        base2 = Path(output_dir) / "label_trajectory"
        try:
            fig.savefig(base2.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(base2.with_suffix(".png"), dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        paths.extend([str(base2.with_suffix(".pdf")), str(base2.with_suffix(".png"))])

    return paths


def plot_metrics(report: EvalReport, path: Union[str, Path]) -> None:
    """Render a horizontal bar chart of 6 normalised metric scores to PDF.

    Produces a single horizontal bar chart with 6 bars, one per canonical
    short-name metric key from ``StageMetrics.normalized``.  The x-axis spans
    ``[0, 1]`` (normalised score range per D-09).

    Parameters
    ----------
    report : EvalReport
        Evaluation report whose ``metrics.normalized`` field is a dict with
        6 canonical short-name keys (Pitfall 4 from RESEARCH.md):
        ``"chamfer"``, ``"hausdorff"``, ``"path_smoothness"``,
        ``"temporal_stability"``, ``"f1"``, ``"knn_consistency"``.

        Missing keys default to ``0.0`` via ``.get(k, 0.0)`` — the function
        will always render 6 bars regardless of dict completeness.

    path : str or Path
        Destination PDF file path.  The parent directory must already exist;
        this function does NOT call ``mkdir``.

    Returns
    -------
    None

    Notes
    -----
    D-09: Horizontal bar chart chosen for immediate readability of 6
    normalised metric scores.  Values from ``report.metrics.normalized``
    are already in ``[0, 1]`` after ``MetricsEngine.normalize``.

    The 6 canonical short-name keys (Pitfall 4 from RESEARCH.md) are
    hardcoded in this function to guarantee consistent bar order regardless
    of dict insertion order.  They match the keys in ``EvalConfig.metric_weights``.

    FRAME-08 mandatory rules:
    - ``plt.close(fig)`` is called after ``fig.savefig``.
    - ``fig.savefig(path, bbox_inches="tight")`` is used.

    The x-axis limit is set to ``[0, 1]`` via ``ax.set_xlim(0, 1)`` per D-09.
    """
    labels = [
        "chamfer",
        "hausdorff",
        "path_smoothness",
        "temporal_stability",
        "f1",
        "knn_consistency",
    ]
    values = [report.metrics.normalized.get(k, 0.0) for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised Score")
    try:
        fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)


def render_dataset_triptych(
    csv_path: "str | Path",
    name: str,
    output_dir: "str | Path",
    dpi: int = 150,
) -> Path:
    """Render a 1×3 triptych PNG for a synthetic dataset CSV (VIZ-01).

    Self-contains all CSV I/O and rendering. Returns the Path to the
    written PNG.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV with columns x, y, z, t.
    name : str
        Dataset name — used as figure suptitle and PNG filename stem.
    output_dir : str or Path
        Directory where <name>.png is written. Created if absent (D-08).
    dpi : int
        Raster resolution of the PNG (default 150).

    Returns
    -------
    Path
        Path to the written PNG (D-06).
    """
    import pandas as pd  # D-05: lazy import — keeps module lightweight

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # D-08

    # --- _frame_indices logic ---
    frames = sorted(
        pd.read_csv(csv_path, usecols=["t"])["t"].unique()
    )
    if not frames:
        raise ValueError(f"CSV contains no time frames: {csv_path}")
    n = len(frames)
    t_first, t_mid, t_last = frames[0], frames[n // 2], frames[-1]

    # --- _load_three_frames logic (D-09) ---
    wanted = {t_first, t_mid, t_last}
    parts = []
    for chunk in pd.read_csv(
        csv_path, usecols=["x", "y", "z", "t"], chunksize=60_000
    ):
        sub = chunk[chunk["t"].isin(wanted)]
        if len(sub):
            parts.append(sub)
    if not parts:
        raise ValueError(f"CSV has 't' column but no rows match {wanted}: {csv_path}")
    df = pd.concat(parts, ignore_index=True)
    data = {
        t: df.loc[df["t"] == t, ["x", "y", "z"]].values for t in wanted
    }

    COLOR = "#2a6496"  # D-07: hardcoded colour
    MAX_PTS = 4_000

    fig = plt.figure(figsize=(13, 4.2))
    fig.suptitle(name, fontsize=12, fontweight="bold", y=1.01)

    for col, (t, label) in enumerate(
        [(t_first, "first"), (t_mid, "mid"), (t_last, "last")]
    ):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        pts = data.get(t, np.empty((0, 3)))

        # Subsampling (D-01)
        n_orig = len(pts)
        rng = np.random.default_rng(0)
        if n_orig > MAX_PTS:
            idx = rng.choice(n_orig, MAX_PTS, replace=False)
            pts = pts[idx]

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=1.5, alpha=0.45, c=COLOR, linewidths=0)

        # Style block
        ax.set_title(f"t = {t}  ({label})\nn = {n_orig:,}",
                     fontsize=9, pad=4)
        for lbl in (
            ax.get_xticklabels()
            + ax.get_yticklabels()
            + ax.get_zticklabels()
        ):
            lbl.set_fontsize(6)
        ax.set_xlabel("x", fontsize=7, labelpad=2)
        ax.set_ylabel("y", fontsize=7, labelpad=2)
        ax.set_zlabel("z", fontsize=7, labelpad=2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    out_path = output_dir / f"{name}.png"
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)  # D-12: mandatory

    return out_path  # D-06: returns Path
