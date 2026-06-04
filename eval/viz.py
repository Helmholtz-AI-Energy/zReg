"""Matplotlib rendering helpers for FRAME-08.

Two public functions produce figures:

- ``plot_trajectory`` — two independent 1×3 figure pairs (alignment_trajectory
  and label_trajectory) written only when the corresponding stage result is
  provided. Returns a list of written path strings.
- ``plot_metrics`` — horizontal bar chart of 6 normalised metrics per D-09.

Notes
-----
1. All figure code runs inside ``matplotlib.rc_context({"backend": "Agg"})``
   per FRAME-08.  This switches the backend to the non-interactive Agg
   renderer, which is CI-safe and requires no display connection.
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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from eval.types import AlignResult, EvalReport, LabelResult

__all__ = ["plot_trajectory", "plot_metrics"]


def plot_trajectory(
    align_result: "AlignResult | None",
    label_result: "LabelResult | None",
    dataset: "dict[int, zRegPointCloud]",
    label_names: "dict[int, str] | None",
    output_dir: Path,
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

    FRAME-08 mandatory rules apply: all figure code inside
    ``matplotlib.rc_context({"backend": "Agg"})``, ``plt.close(fig)`` after
    every ``fig.savefig``, ``bbox_inches="tight"`` on every savefig call.
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
        with matplotlib.rc_context({"backend": "Agg"}):
            fig = plt.figure(figsize=(12, 4))
            source_h = None
            aligned_h = None
            for idx, fk in enumerate(frame_indices):
                ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
                source_pos = dataset[fk]["pos"].detach().cpu().numpy()
                aligned_pos = align_result.aligned_cloud[fk]["pos"].detach().cpu().numpy()
                s = ax.scatter(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2], c="blue", s=5)
                a = ax.scatter(aligned_pos[:, 0], aligned_pos[:, 1], aligned_pos[:, 2], c="orange", s=5)
                if idx == 0:
                    source_h, aligned_h = s, a
                ax.set_title(f"Frame {fk}")
            fig.legend([source_h, aligned_h], ["Source", "Aligned"], loc="center right", bbox_to_anchor=(1.12, 0.5))
            base = Path(output_dir) / "alignment_trajectory"
            try:
                fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
                fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
            finally:
                plt.close(fig)
        paths.extend([str(base.with_suffix(".pdf")), str(base.with_suffix(".png"))])

    # ------------------------------------------------------------------
    # LABEL FIGURE — written only when label_result is not None
    # ------------------------------------------------------------------
    if label_result is not None:
        # Collect union of label IDs across the 3 plotted frames
        union_labels: set[int] = set()
        for fk in frame_indices:
            t = label_result.transferred_labels[fk]
            union_labels.update(int(v) for v in torch.unique(t).tolist())
        sorted_labels = sorted(union_labels)

        # Build colormap — non-deprecated API (matplotlib 3.7+)
        cmap = matplotlib.colormaps.get_cmap("tab10")
        color_for_label = {lab: cmap(i % 10) for i, lab in enumerate(sorted_labels)}

        with matplotlib.rc_context({"backend": "Agg"}):
            fig = plt.figure(figsize=(12, 4))
            for idx, fk in enumerate(frame_indices):
                ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
                # D-07: use aligned positions when align stage ran
                pos = (
                    align_result.aligned_cloud[fk]["pos"].detach().cpu().numpy()
                    if align_result is not None
                    else dataset[fk]["pos"].detach().cpu().numpy()
                )
                t = label_result.transferred_labels[fk]
                c_vals = [color_for_label[int(v)] for v in t.tolist()]
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c_vals, s=5)
                ax.set_title(f"Frame {fk}")
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
                fig.savefig(base2.with_suffix(".png"), bbox_inches="tight")
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
    - All figure code is inside ``matplotlib.rc_context({"backend": "Agg"})``.
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

    with matplotlib.rc_context({"backend": "Agg"}):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(labels, values)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalised Score")
        try:
            fig.savefig(path, bbox_inches="tight")
        finally:
            plt.close(fig)
