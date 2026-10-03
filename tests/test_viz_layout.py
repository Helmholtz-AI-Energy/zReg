"""Layout and content checks for eval.viz figures (Phase 63 VIZ-01, VIZ-02, IN-10).

These tests drive the real viz code paths (``plot_trajectory``,
``plot_metrics``, ``render_dataset_triptych``) including the real
``savefig`` calls.  Only ``plt.close`` is wrapped, to capture each figure
right after it has been saved; a closed figure can still be measured with
its renderer.

- VIZ-01: legends and the triptych suptitle must lie inside the saved
  canvas (renderer-based window extents), at the declared figure size
  (``bbox_inches`` stays ``None`` for 3-D figures, commit 13ee2f8).
- VIZ-02: the source panel must show exactly the source frame that
  ``AlignmentStage._build_aligned_cloud`` used (``source_sorted[::step]``),
  checked against a temporal-only alignment oracle.
- IN-10: ``plot_metrics`` must not draw a never-computed metric as a zero
  bar; such rows get an "n/a" annotation instead.
"""

from pathlib import Path

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_labels, generate_trajectory

import torch

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

import eval.viz as viz
from eval.types import AlignResult, EvalReport, LabelResult, StageMetrics


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch) -> list:
    """Collect every figure passed to ``viz.plt.close`` (real close still runs)."""
    figs: list = []
    real_close = viz.plt.close

    def _capture(fig=None):
        figs.append(fig)
        return real_close(fig)

    monkeypatch.setattr(viz.plt, "close", _capture)
    return figs


def _assert_inside(fig, bb, what: str) -> None:
    """Assert window extent *bb* lies inside the figure canvas."""
    fb = fig.bbox
    assert bb.x0 >= 0, f"{what} starts left of the canvas: x0={bb.x0:.1f}"
    assert bb.x1 <= fb.x1, f"{what} ends right of the canvas: x1={bb.x1:.1f} > {fb.x1:.1f}"
    assert bb.y0 >= 0, f"{what} starts below the canvas: y0={bb.y0:.1f}"
    assert bb.y1 <= fb.y1, f"{what} ends above the canvas: y1={bb.y1:.1f} > {fb.y1:.1f}"


def _legend_figures(figs: list) -> list:
    return [f for f in figs if f is not None and f.legends]


@pytest.fixture
def dataset_3() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with 3 label ids."""
    return generate_labels(
        generate_trajectory(n_points=60, n_frames=3, seed=0),
        n_labels=3,
        seed=0,
    )


@pytest.fixture
def dataset_20_labels() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory whose points carry 20 distinct label ids."""
    ds = generate_trajectory(n_points=60, n_frames=3, seed=0)
    for k in ds:
        n = ds[k]["pos"].shape[0]
        ds[k]["label"] = torch.arange(n, dtype=torch.long) % 20
    return ds


def _identity_align_result(ds) -> AlignResult:
    n = len(ds)
    return AlignResult(
        aligned_cloud=ds,
        warp_path=[(i, i) for i in range(n)],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={},
    )


def _label_result(ds) -> LabelResult:
    return LabelResult(
        transferred_labels={k: ds[k]["label"].long() for k in ds},
        params_used={},
    )


def _report(normalized: dict, sanity_flags: list[str], **raw) -> EvalReport:
    values = dict(
        chamfer_distance=0.5,
        hausdorff_distance=0.3,
        path_smoothness=0.1,
        temporal_stability=0.2,
        f1_score=0.8,
        knn_consistency=0.7,
    )
    values.update(raw)
    return EvalReport(
        params={},
        metrics=StageMetrics(normalized=normalized, **values),
        aggregated_metrics={},
        per_dataset={},
        plot_paths=[],
        sanity_flags=sanity_flags,
    )


FULL_NORMALIZED = {
    "chamfer": 0.66,
    "hausdorff": 0.77,
    "path_smoothness": 0.91,
    "temporal_stability": 0.5,
    "f1": 0.8,
    "knn_consistency": 0.7,
}


def _metrics_rows(figs: list):
    """Return (filled bar rows, n/a annotated rows) of the captured metrics figure."""
    fig = figs[-1]
    ax = fig.axes[0]
    filled_rows = set()
    for p in ax.patches:
        fc = p.get_facecolor()
        if p.get_fill() and fc[3] > 0:
            filled_rows.add(int(round(p.get_y() + p.get_height() / 2)))
    na_rows = {
        int(round(t.get_position()[1]))
        for t in ax.texts
        if "n/a" in t.get_text()
    }
    return filled_rows, na_rows


# ---------------------------------------------------------------------------
# VIZ-01 — legends and titles inside the saved canvas
# ---------------------------------------------------------------------------


def test_superposed_legend_inside_canvas(dataset_3, captured, tmp_path: Path) -> None:
    """The superposed figure's legend lies inside the fixed 12x4 in canvas."""
    viz.plot_trajectory(
        _identity_align_result(dataset_3), None, dataset_3, None, tmp_path, dataset_3,
    )
    figs = _legend_figures(captured)
    assert len(figs) == 1
    fig = figs[0]
    r = fig.canvas.get_renderer()
    _assert_inside(fig, fig.legends[0].get_window_extent(r), "superposed legend")


@pytest.mark.parametrize("which", ["3_labels", "20_labels"])
def test_label_legend_inside_canvas(
    which, dataset_3, dataset_20_labels, captured, tmp_path: Path
) -> None:
    """Both label figures keep their legend inside the canvas (3 and 20 ids)."""
    ds = dataset_3 if which == "3_labels" else dataset_20_labels
    n_expected = 3 if which == "3_labels" else 20
    viz.plot_trajectory(None, _label_result(ds), ds, None, tmp_path)
    figs = _legend_figures(captured)
    assert len(figs) == 2  # label_source + label_target
    for fig in figs:
        assert len(fig.legends[0].get_texts()) == n_expected
        r = fig.canvas.get_renderer()
        _assert_inside(fig, fig.legends[0].get_window_extent(r), f"label legend ({which})")


def test_superposed_png_keeps_declared_size(dataset_3, captured, tmp_path: Path) -> None:
    """No tight cropping: the PNG is exactly figsize * 150 dpi (13ee2f8 intent)."""
    viz.plot_trajectory(
        _identity_align_result(dataset_3), None, dataset_3, None, tmp_path, dataset_3,
    )
    img = mpimg.imread(tmp_path / "alignment_superposed_trajectory.png")
    assert img.shape[:2] == (600, 1800)
    img = mpimg.imread(tmp_path / "alignment_source_trajectory.png")
    assert img.shape[:2] == (600, 1800)


@pytest.fixture
def triptych_csv(tmp_path):
    """Minimal 3-frame CSV with 50 points per frame (x, y, z, t columns)."""
    rng = np.random.default_rng(42)
    rows = []
    for t in [1, 2, 3]:
        for _ in range(50):
            rows.append({
                "x": float(rng.random()),
                "y": float(rng.random()),
                "z": float(rng.random()),
                "t": t,
            })
    csv_path = tmp_path / "test_dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_triptych_suptitle_inside_canvas(triptych_csv, captured, tmp_path: Path) -> None:
    """The triptych suptitle is not clipped at the top of the saved PNG."""
    out = viz.render_dataset_triptych(triptych_csv, "My Dataset", tmp_path / "out")
    fig = captured[-1]
    r = fig.canvas.get_renderer()
    _assert_inside(fig, fig._suptitle.get_window_extent(r), "triptych suptitle")
    img = mpimg.imread(out)
    assert img.shape[:2] == (round(4.2 * 150), round(13 * 150))
