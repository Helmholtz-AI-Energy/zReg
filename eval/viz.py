"""Matplotlib rendering helpers for FRAME-08.

Two public functions produce figures:

- ``plot_trajectory`` — up to four independent 1×3 figure pairs for the
  alignment branch (source, target, aligned, superposed) plus two pairs for
  the label branch (source labels, transferred labels).  Returns a list of
  written path strings.
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _deduplicate_frames(candidates: list[int]) -> list[int]:
    """Return *candidates* with duplicates removed, preserving order.

    Used by both the alignment and label branches to handle 1- and 2-frame
    datasets where first/middle/last indices collapse.

    Parameters
    ----------
    candidates : list[int]
        Ordered list of frame indices (may contain duplicates).

    Returns
    -------
    list[int]
        Deduplicated list in insertion order.
    """
    seen: set[int] = set()
    return [k for k in candidates if not (k in seen or seen.add(k))]


def _warp_source_map(warp_path: list[tuple[int, int]]) -> dict[int, int]:
    """Build a ``target_idx -> source_idx`` mapping from *warp_path*.

    Takes the **first** occurrence per target index (DTW paths may repeat
    a target index).

    Parameters
    ----------
    warp_path : list of (src_idx, tgt_idx) tuples
        DTW warp path as stored in ``AlignResult.warp_path``.

    Returns
    -------
    dict[int, int]
        Mapping from target frame index to source frame index.
    """
    mapping: dict[int, int] = {}
    for src, tgt in warp_path:
        if tgt not in mapping:
            mapping[tgt] = src
    return mapping


def _subsample(arr: np.ndarray, max_pts: int = 4000) -> np.ndarray:
    """Subsample *arr* to at most *max_pts* rows using a fixed RNG seed.

    Parameters
    ----------
    arr : np.ndarray
        Point array of shape ``(N, 3)``.
    max_pts : int
        Maximum number of points to retain (default 4 000).

    Returns
    -------
    np.ndarray
        Subsampled array; unchanged when ``len(arr) <= max_pts``.
    """
    rng = np.random.default_rng(0)
    if len(arr) > max_pts:
        return arr[rng.choice(len(arr), max_pts, replace=False)]
    return arr


def _get_source_labels(pc: "zRegPointCloud") -> "torch.Tensor | None":
    """Return source labels from *pc*.

    Reads ``pc["label"]``; returns ``None`` when the field is absent or None
    (grey render fallback is acceptable in the viz layer).
    """
    lbl = pc.get("label")
    if lbl is not None:
        return lbl.long()
    return None


def _ax_style(ax, title: str) -> None:
    """Apply standard tick/label/pane style to a 3-D axes."""
    ax.set_title(title, fontsize=9, pad=4)
    for lbl in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        lbl.set_fontsize(6)
    ax.set_xlabel("x", fontsize=7, labelpad=2)
    ax.set_ylabel("y", fontsize=7, labelpad=2)
    ax.set_zlabel("z", fontsize=7, labelpad=2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def _save_fig(fig, base: Path, paths: list[str]) -> None:
    """Save *fig* as PDF + PNG at *base* (no suffix), close fig, extend *paths*.

    ``paths.extend`` is placed after both ``savefig`` calls so that paths are
    only recorded when the full pair succeeds (atomic pair semantics).  If the
    PNG save raises, neither path is appended.  ``plt.close`` always runs via
    the ``finally`` block to prevent figure handle leaks.
    """
    try:
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
        paths.extend([str(base.with_suffix(".pdf")), str(base.with_suffix(".png"))])
    finally:
        plt.close(fig)


def _write_single_cloud_figure(
    frame_indices: list[int],
    per_frame_pos: dict[int, np.ndarray],
    color: str,
    stem: str,
    output_dir: Path,
    paths: list[str],
) -> None:
    """Write a 1×3 figure showing a single point cloud per frame.

    Parameters
    ----------
    frame_indices : list[int]
        Ordered list of frame indices to plot (at most 3).
    per_frame_pos : dict[int, np.ndarray]
        Pre-computed (possibly subsampled) position arrays keyed by frame index.
    color : str
        Matplotlib colour string (hex or named colour).
    stem : str
        Output filename stem (without extension); e.g. ``"alignment_source_trajectory"``.
    output_dir : Path
        Directory where the PDF and PNG are written.
    paths : list[str]
        Accumulator for written path strings — extended in place.
    """
    fig = plt.figure(figsize=(12, 4))
    for idx, fk in enumerate(frame_indices):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        pos = per_frame_pos[fk]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=color, s=1.5, alpha=1.0, linewidths=0)
        _ax_style(ax, f"Frame {fk}")
    _save_fig(fig, Path(output_dir) / stem, paths)


def _write_superposed_figure(
    frame_indices: list[int],
    src_pos_map: dict[int, np.ndarray],
    aligned_pos_map: dict[int, np.ndarray],
    tgt_pos_map: "dict[int, np.ndarray] | None",
    output_dir: Path,
    paths: list[str],
) -> None:
    """Write a 1×3 figure superposing source, aligned, and optionally target clouds.

    Parameters
    ----------
    frame_indices : list[int]
        Ordered list of frame indices to plot (at most 3).
    src_pos_map : dict[int, np.ndarray]
        Pre-computed source position arrays keyed by frame index.
    aligned_pos_map : dict[int, np.ndarray]
        Pre-computed aligned position arrays keyed by frame index.
    tgt_pos_map : dict[int, np.ndarray] or None
        Pre-computed target position arrays keyed by frame index; ``None``
        when no target was provided.
    output_dir : Path
        Directory where the PDF and PNG are written.
    paths : list[str]
        Accumulator for written path strings — extended in place.
    """
    fig = plt.figure(figsize=(12, 4))
    for idx, fk in enumerate(frame_indices):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        src = src_pos_map[fk]
        aln = aligned_pos_map[fk]
        ax.scatter(src[:, 0], src[:, 1], src[:, 2],
                   c="#3a7abf", s=1.5, alpha=1.0, linewidths=0)
        ax.scatter(aln[:, 0], aln[:, 1], aln[:, 2],
                   c="#e07b39", s=1.5, alpha=1.0, linewidths=0)
        if tgt_pos_map is not None and fk in tgt_pos_map:
            tgt = tgt_pos_map[fk]
            ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2],
                       c="#38a058", s=1.5, alpha=1.0, linewidths=0)
        _ax_style(ax, f"Frame {fk}")
    # WR-02: derive legend from whether any target data was actually rendered.
    # tgt_pos_map may be non-None but still cover fewer keys than frame_indices
    # (e.g. paired run where target keys are a strict subset). Only include
    # "Target" when at least one subplot actually drew target scatter points.
    has_target = tgt_pos_map is not None and any(fk in tgt_pos_map for fk in frame_indices)
    legend_labels = ["Source", "Aligned"] + (["Target"] if has_target else [])
    fig.legend(legend_labels, loc="center right", bbox_to_anchor=(1.12, 0.5))
    _save_fig(fig, Path(output_dir) / "alignment_superposed_trajectory", paths)


def _write_label_figure(
    frame_indices: "list[int]",
    per_frame_pos: "dict[int, np.ndarray]",
    per_frame_labels: "dict[int, list[str] | None]",
    color_for_label: "dict[int, str]",
    label_names: "dict[int, str] | None",
    stem: str,
    output_dir: Path,
    paths: "list[str]",
) -> None:
    """Write a 1×3 label figure (PDF + PNG) showing coloured point clouds.

    Parameters
    ----------
    frame_indices : list[int]
        Ordered frame keys to plot (at most 3).
    per_frame_pos : dict[int, np.ndarray]
        Pre-subsampled position arrays keyed by frame index.
    per_frame_labels : dict[int, list[str] or None]
        Per-point colour strings aligned to *per_frame_pos*; ``None`` signals
        the all-unknown fallback (renders a single grey scatter, no legend).
    color_for_label : dict[int, str]
        Mapping of integer label ID → colour string; used for legend patches.
    label_names : dict[int, str] or None
        Human-readable names for label IDs (legend text).
    stem : str
        Output filename stem without extension.
    output_dir : Path
        Directory where PDF and PNG are written.
    paths : list[str]
        Accumulator extended in place with the written file paths.
    """
    fig = plt.figure(figsize=(12, 4))
    for idx, fk in enumerate(frame_indices):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        pos = per_frame_pos[fk]
        c_vals = per_frame_labels[fk]
        if c_vals is None:
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c="#aaaaaa", s=1.5, alpha=1.0, linewidths=0)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=c_vals, s=1.5, alpha=1.0, linewidths=0)
        _ax_style(ax, f"Frame {fk}")
    patches = [
        mpatches.Patch(
            color=color_for_label[lab],
            label=(label_names.get(lab, str(lab)) if label_names else str(lab)),
        )
        for lab in sorted(color_for_label)
    ]
    if patches:
        fig.legend(handles=patches, loc="center right", bbox_to_anchor=(1.15, 0.5))
    _save_fig(fig, Path(output_dir) / stem, paths)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_trajectory(
    align_result: "AlignResult | None",
    label_result: "LabelResult | None",
    dataset: "dict[int, zRegPointCloud]",
    label_names: "dict[int, str] | None",
    output_dir: Path,
    target: "dict[int, zRegPointCloud] | None" = None,
) -> list[str]:
    """Render trajectory figures for alignment and/or label-transfer stages (EXT-02).

    Alignment branch (written when ``align_result is not None``):

    - ``alignment_source_trajectory.{pdf,png}`` — source cloud per frame (blue).
    - ``alignment_target_trajectory.{pdf,png}`` — target cloud per frame (green);
      **skipped** when ``target is None``.
    - ``alignment_aligned_trajectory.{pdf,png}`` — aligned source cloud per frame
      (orange).
    - ``alignment_superposed_trajectory.{pdf,png}`` — all available clouds
      superposed per frame, with legend.

    Label branch (written when ``label_result is not None``):

    - ``label_source_trajectory.{pdf,png}`` — source cloud coloured by source
      labels (``dataset[fk]["label"]``).
    - ``label_target_trajectory.{pdf,png}`` — target/aligned cloud coloured by
      transferred labels (``label_result.transferred_labels``).

    Return value length depends on which stages ran and whether *target* is
    provided:

    - No stages → 0
    - Align only, no target → 6
    - Align only, with target → 8
    - Label only → 4
    - Align + label, no target → 10
    - Align + label, with target → 12
    - (all combinations) → 0, 4, 6, 8, 10, or 12

    Parameters
    ----------
    align_result : AlignResult or None
        Alignment result carrying ``aligned_cloud: dict[int, zRegPointCloud]``
        and ``warp_path: list[tuple[int, int]]``.
        Pass ``None`` to skip the alignment figures.
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
    target : dict[int, zRegPointCloud] or None
        Optional target dataset.  When provided, a separate target figure
        is written and the superposed figure includes the target cloud.

    Returns
    -------
    list[str]
        Absolute path strings of the files actually written, in order:
        alignment figures first (source, [target], aligned, superposed),
        then label figure.

    Notes
    -----
    D-04: Each figure uses first, middle, and last frame from the relevant
    cloud's sorted keys, deduplicated via ``_deduplicate_frames``.

    D-07: Positions for the label figure use ``align_result.aligned_cloud``
    when ``align_result is not None`` (post-alignment coordinate space,
    consistent with label assignment); otherwise ``dataset[frame]["pos"]``
    is used (label-only run).

    FRAME-08 mandatory rules apply: ``plt.close(fig)`` after every
    ``fig.savefig``, ``bbox_inches="tight"`` on every savefig call.
    """
    paths: list[str] = []

    # D-04: first / middle / last frame selection
    sorted_keys = sorted(dataset.keys())
    if not sorted_keys:
        return paths  # no frames to plot

    # ------------------------------------------------------------------
    # ALIGNMENT FIGURES — written only when align_result is not None
    # ------------------------------------------------------------------
    if align_result is not None:
        # aligned_cloud is keyed by target frame indices (Phase 33/35 CPD output).
        # Derive frame selection from aligned_cloud keys so all lookups are valid.
        align_sorted = sorted(align_result.aligned_cloud.keys())
        align_frame_indices = _deduplicate_frames([
            align_sorted[0],
            align_sorted[len(align_sorted) // 2],
            align_sorted[-1],
        ])

        # Build source-frame mapping from warp path
        src_map = _warp_source_map(align_result.warp_path)

        # Pre-compute subsampled arrays once per cloud per frame (Task 4)
        per_frame_source: dict[int, np.ndarray] = {}
        per_frame_aligned: dict[int, np.ndarray] = {}
        per_frame_target: dict[int, np.ndarray] = {}

        for fk in align_frame_indices:
            source_fk = src_map.get(fk, fk)  # fallback: use fk if no warp mapping
            per_frame_source[fk] = _subsample(
                dataset[source_fk]["pos"].detach().cpu().numpy()
            )
            per_frame_aligned[fk] = _subsample(
                align_result.aligned_cloud[fk]["pos"].detach().cpu().numpy()
            )
            if target is not None and fk in target:
                per_frame_target[fk] = _subsample(
                    target[fk]["pos"].detach().cpu().numpy()
                )

        # Figure 1: source only (blue)
        _write_single_cloud_figure(
            align_frame_indices, per_frame_source,
            color="#3a7abf", stem="alignment_source_trajectory",
            output_dir=Path(output_dir), paths=paths,
        )

        # Figure 2: target only (green) — skipped when target is None.
        # Guard: only plot frames present in both align_frame_indices and
        # per_frame_target (CR-01: target keys may be a strict subset of
        # align_frame_indices when source and target have non-overlapping
        # frame sets in a paired-alignment run).
        if target is not None:
            target_plot_indices = [fk for fk in align_frame_indices if fk in per_frame_target]
            if target_plot_indices:
                _write_single_cloud_figure(
                    target_plot_indices, per_frame_target,
                    color="#38a058", stem="alignment_target_trajectory",
                    output_dir=Path(output_dir), paths=paths,
                )

        # Figure 3: aligned source only (orange)
        _write_single_cloud_figure(
            align_frame_indices, per_frame_aligned,
            color="#e07b39", stem="alignment_aligned_trajectory",
            output_dir=Path(output_dir), paths=paths,
        )

        # Figure 4: all available clouds superposed
        _write_superposed_figure(
            align_frame_indices,
            per_frame_source, per_frame_aligned,
            per_frame_target if target is not None else None,
            output_dir=Path(output_dir), paths=paths,
        )

    # ------------------------------------------------------------------
    # LABEL FIGURES — written only when label_result is not None (VIZ-03)
    # Two independent 1×3 figure pairs:
    #   label_source_trajectory — source cloud coloured by source labels
    #   label_target_trajectory — target/aligned cloud coloured by transferred labels
    # ------------------------------------------------------------------
    if label_result is not None:
        label_sorted = sorted(label_result.transferred_labels.keys())
        label_frame_indices = _deduplicate_frames([
            label_sorted[0],
            label_sorted[len(label_sorted) // 2],
            label_sorted[-1],
        ])

        # 10-colour tab10 palette — supports up to 10 distinct classes before wrapping.
        _tab10 = [matplotlib.colormaps["tab10"](i) for i in range(10)]
        _label_palette = [
            f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for r, g, b, _ in _tab10
        ]

        # Task 4: union of all label IDs across both figures for consistent palette
        union_labels: set[int] = set()
        for fk in label_frame_indices:
            if fk not in dataset:
                continue
            src = _get_source_labels(dataset[fk])
            if src is not None:
                union_labels.update(int(v) for v in torch.unique(src).tolist())
            union_labels.update(
                int(v) for v in torch.unique(label_result.transferred_labels[fk]).tolist()
            )
        color_for_label = {
            lab: _label_palette[i % len(_label_palette)]
            for i, lab in enumerate(sorted(union_labels))
        }

        # Task 3: pre-compute source figure data
        source_pos_map: dict[int, np.ndarray] = {}
        source_colors_map: dict[int, "list[str] | None"] = {}
        for fk in label_frame_indices:
            if fk not in dataset:
                continue
            src_labels = _get_source_labels(dataset[fk])
            raw_pos = dataset[fk]["pos"].detach().cpu().numpy()
            if src_labels is not None:
                raw_labels = src_labels.tolist()
                n = min(len(raw_pos), len(raw_labels))
                raw_pos = raw_pos[:n]
                raw_labels = raw_labels[:n]
                if n > 4000:
                    keep = np.random.default_rng(0).choice(n, 4000, replace=False)
                    raw_pos = raw_pos[keep]
                    raw_labels = [raw_labels[i] for i in keep]
                source_pos_map[fk] = raw_pos
                source_colors_map[fk] = [color_for_label[int(v)] for v in raw_labels]
            else:
                source_pos_map[fk] = _subsample(raw_pos)
                source_colors_map[fk] = None

        # Task 3: pre-compute target figure data (preserves D-07 position logic)
        target_pos_map: dict[int, np.ndarray] = {}
        target_colors_map: dict[int, "list[str]"] = {}
        for fk in label_frame_indices:
            if target is not None and fk in target:
                pos = target[fk]["pos"].detach().cpu().numpy()
            elif align_result is not None and fk in align_result.aligned_cloud:
                pos = align_result.aligned_cloud[fk]["pos"].detach().cpu().numpy()
            else:
                if fk not in dataset:
                    raise KeyError(
                        f"Label frame key {fk!r} not found in dataset. "
                        "For label-only runs, dataset must be the target trajectory."
                    )
                pos = dataset[fk]["pos"].detach().cpu().numpy()
            t_labels = label_result.transferred_labels[fk]
            c_vals = [color_for_label[int(v)] for v in t_labels.tolist()]
            n_pts = min(len(pos), len(c_vals))
            rng = np.random.default_rng(0)
            if n_pts > 4000:
                keep = rng.choice(n_pts, 4000, replace=False)
                pos = pos[keep]
                c_vals = [c_vals[i] for i in keep]
            else:
                pos = pos[:n_pts]
                c_vals = c_vals[:n_pts]
            target_pos_map[fk] = pos
            target_colors_map[fk] = c_vals

        # Figure 1: source cloud coloured by source labels
        # Only pass frames that ended up in source_pos_map (guard for mismatched keys)
        source_frame_indices = [fk for fk in label_frame_indices if fk in source_pos_map]
        # If all source frames lack labels, pass empty color_for_label to suppress legend
        all_none_source = all(source_colors_map.get(fk) is None for fk in source_frame_indices)
        _write_label_figure(
            source_frame_indices, source_pos_map, source_colors_map,
            {} if all_none_source else color_for_label, label_names,
            stem="label_source_trajectory",
            output_dir=Path(output_dir), paths=paths,
        )

        # Figure 2: target/aligned cloud coloured by transferred labels
        _write_label_figure(
            label_frame_indices, target_pos_map, target_colors_map,
            color_for_label, label_names,
            stem="label_target_trajectory",
            output_dir=Path(output_dir), paths=paths,
        )

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
