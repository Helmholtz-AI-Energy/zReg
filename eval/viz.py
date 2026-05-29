"""Matplotlib rendering helpers for FRAME-08.

Two public functions produce PDF figures:

- ``plot_point_cloud`` — 3D scatter per frame, up to 4 frames per D-08,
  AlignResult-only per RESEARCH Pitfall 4 / Option 1.
- ``plot_metrics_summary`` — horizontal bar chart of 6 normalised metrics
  per D-09.

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
4. ``plot_point_cloud`` is restricted to ``AlignResult`` — ``LabelResult``
   lacks positional data (``aligned_cloud`` dict with per-frame ``pos``
   tensors), so it cannot be used to render 3D scatter plots (RESEARCH
   §plot_point_cloud Signature Gap Option 1).
5. PDF output only; no PNG, SVG, or display rendering.
6. mathtext only — no system TeX dependencies.
"""

from pathlib import Path
from typing import Union

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# See: tests/conftest.py:20-24, eval/data_factory.py:18-35, eval/metrics.py:53-67
from zreg.dataset import zRegPointCloud  # noqa: F401 — ensures import order

import torch  # noqa: F401 — must follow zreg.* (libomp SIGABRT rule)

import matplotlib
import matplotlib.pyplot as plt

from eval.types import AlignResult, EvalReport

__all__ = ["plot_point_cloud", "plot_metrics_summary"]


def plot_point_cloud(result: AlignResult, path: Union[str, Path]) -> None:
    """Render a 3D scatter of up to 4 frames from an AlignResult to PDF.

    Produces one subplot per frame (up to 4 frames; remaining frames are
    silently ignored per D-08).  Each subplot shows all points in that
    frame coloured by their label/class value from the ``color`` tensor.

    Parameters
    ----------
    result : AlignResult
        Alignment result carrying ``aligned_cloud: dict[int, zRegPointCloud]``.
        Each point cloud must have:

        - ``pos``: ``torch.Tensor`` of shape ``(N, 3)`` — 3D positions.
        - ``color``: ``torch.Tensor`` of shape ``(N,)`` (label indices, torch.long)
          or ``(N, 3)`` (RGB — reduced to ``[:, 0]`` before passing to scatter).

        Only ``AlignResult`` is supported; ``LabelResult`` has no positional
        data (RESEARCH.md Pitfall 4 / §plot_point_cloud Signature Gap Option 1).
        Passing a non-AlignResult that lacks an ``aligned_cloud`` attribute will
        raise ``AttributeError``, which is the intended failure mode.

    path : str or Path
        Destination PDF file path.  The parent directory must already exist;
        this function does NOT call ``mkdir`` — the caller (EvaluationRunner)
        is responsible for creating the output directory (D-12).

    Returns
    -------
    None

    Notes
    -----
    D-07: Points are coloured by the per-point ``color`` tensor (label index
    or class value), producing a visually interpretable scatter per frame.

    D-08: First 4 frames only.  Beyond 4, the figure would be unreadably small.
    The slice ``list(result.aligned_cloud.items())[:4]`` enforces this cap.

    AlignResult-only restriction: The function directly accesses
    ``result.aligned_cloud``.  If called with a ``LabelResult``, the
    ``AttributeError`` raised is the intended signal that this function is
    not suitable for that type.

    FRAME-08 mandatory rules:
    - All figure code is inside ``matplotlib.rc_context({"backend": "Agg"})``.
    - ``plt.close(fig)`` is called after ``fig.savefig``.
    - ``fig.savefig(path, bbox_inches="tight")`` is used.

    Color tensor shape robustness: shape ``(N,)`` is used directly; shape
    ``(N, k)`` (2-D) is reduced via ``[:, 0]`` before passing to
    ``ax.scatter``.  ``generate_labels`` from Phase 14 D-04 produces
    ``color`` tensors of shape ``(N, 3)``; the ``[:, 0]`` slice extracts the
    first column (the class index).
    """
    frames = list(result.aligned_cloud.items())[:4]  # D-08: cap at 4 frames
    n = len(frames)

    with matplotlib.rc_context({"backend": "Agg"}):
        fig = plt.figure(figsize=(4 * n, 4))
        for i, (frame_key, pc) in enumerate(frames):
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            pos = pc["pos"].detach().cpu().numpy()          # shape (N, 3)
            color = pc["color"].detach().cpu().numpy()      # shape (N,) or (N, k)
            # Robustness: generate_labels produces (N, 3); reduce to 1-D for scatter
            if color.ndim == 2 and color.shape[1] >= 1:
                color = color[:, 0]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=5)
            ax.set_title(f"Frame {frame_key}")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_metrics_summary(report: EvalReport, path: Union[str, Path]) -> None:
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
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
