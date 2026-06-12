"""Tests for eval.viz — plot_trajectory and plot_metrics (FRAME-08)."""

from pathlib import Path

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_labels, generate_trajectory

import torch

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from eval.types import AlignResult, EvalReport, LabelResult, StageMetrics
from eval.viz import plot_metrics, plot_trajectory, render_dataset_triptych


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset_3() -> dict[int, zRegPointCloud]:
    """3-frame synthetic trajectory with color labels (normal case)."""
    return generate_labels(
        generate_trajectory(n_points=20, n_frames=3, seed=0),
        n_classes=4,
        seed=0,
    )


@pytest.fixture
def synthetic_dataset_1() -> dict[int, zRegPointCloud]:
    """1-frame synthetic dataset — D-08 lower-bound test."""
    return generate_labels(
        generate_trajectory(n_points=20, n_frames=1, seed=0),
        n_classes=4,
        seed=0,
    )


@pytest.fixture
def synthetic_dataset_5() -> dict[int, zRegPointCloud]:
    """5-frame synthetic dataset — D-08 cap test (should produce 4 subplots)."""
    return generate_labels(
        generate_trajectory(n_points=20, n_frames=5, seed=0),
        n_classes=4,
        seed=0,
    )


@pytest.fixture
def fake_align_result(synthetic_dataset_3) -> AlignResult:
    """AlignResult constructed without DTW execution (3-frame dataset)."""
    return AlignResult(
        aligned_cloud=synthetic_dataset_3,
        warp_path=[(0, 0), (1, 1), (2, 2)],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={},
    )


@pytest.fixture
def fake_align_result_1frame(synthetic_dataset_1) -> AlignResult:
    """AlignResult constructed without DTW execution (1-frame dataset)."""
    return AlignResult(
        aligned_cloud=synthetic_dataset_1,
        warp_path=[(0, 0)],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={},
    )


@pytest.fixture
def fake_align_result_5frames(synthetic_dataset_5) -> AlignResult:
    """AlignResult constructed without DTW execution (5-frame dataset)."""
    return AlignResult(
        aligned_cloud=synthetic_dataset_5,
        warp_path=[(i, i) for i in range(5)],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={},
    )


@pytest.fixture
def fake_label_result(synthetic_dataset_3) -> LabelResult:
    """LabelResult constructed from synthetic_dataset_3 with 1-D long tensors.

    generate_labels produces color tensors of shape (N,) with dtype torch.int64,
    so we use them directly as 1-D long tensors.
    """
    return LabelResult(
        transferred_labels={
            k: synthetic_dataset_3[k]['color'].long()
            for k in synthetic_dataset_3
        },
        params_used={},
    )


@pytest.fixture
def fake_report() -> EvalReport:
    """EvalReport with all 6 normalized metric keys populated."""
    metrics = StageMetrics(
        chamfer_distance=0.5,
        hausdorff_distance=0.3,
        path_smoothness=0.1,
        temporal_stability=0.0,
        f1_score=0.8,
        knn_consistency=0.7,
        normalized={
            "chamfer": 0.66,
            "hausdorff": 0.77,
            "path_smoothness": 0.91,
            "temporal_stability": 1.0,
            "f1": 0.8,
            "knn_consistency": 0.7,
        },
    )
    return EvalReport(
        params={},
        metrics=metrics,
        aggregated_metrics={},
        per_dataset={},
        plot_paths=[],
        sanity_flags=[],
    )


# ---------------------------------------------------------------------------
# TestPlotMetrics — FRAME-08 G2
# ---------------------------------------------------------------------------


class TestPlotMetrics:
    """FRAME-08 G2: plot_metrics produces non-empty PDF without figure leaks."""

    def test_creates_pdf_file_at_path(self, fake_report, tmp_path) -> None:
        """PDF is created at the given path with size > 0."""
        out = tmp_path / "summary.pdf"
        plot_metrics(fake_report, out)
        assert out.exists()
        assert out.suffix == ".pdf"
        assert out.stat().st_size > 0

    def test_no_figure_leak(self, fake_report, tmp_path) -> None:
        """plt.close(fig) is called — no leaked figure handles after return."""
        before = len(plt.get_fignums())
        plot_metrics(fake_report, tmp_path / "summary_leak.pdf")
        after = len(plt.get_fignums())
        assert after == before

    def test_accepts_report_with_partial_normalized_dict(self, tmp_path) -> None:
        """Report with only one normalized key still produces 6 bars via .get(k, 0.0)."""
        partial_metrics = StageMetrics(
            chamfer_distance=0.5,
            hausdorff_distance=0.3,
            path_smoothness=0.1,
            temporal_stability=0.0,
            f1_score=0.8,
            knn_consistency=0.7,
            normalized={"chamfer": 0.5},  # only one key — remaining default to 0.0
        )
        partial_report = EvalReport(
            params={},
            metrics=partial_metrics,
            aggregated_metrics={},
            per_dataset={},
            plot_paths=[],
            sanity_flags=[],
        )
        out = tmp_path / "partial.pdf"
        plot_metrics(partial_report, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# TestPlotTrajectory — EXT-02
# ---------------------------------------------------------------------------


class TestPlotTrajectory:
    """EXT-02: plot_trajectory produces correct files for all 4 stage combinations."""

    def test_align_only_writes_alignment_files(self, fake_align_result, synthetic_dataset_3, tmp_path):
        result = plot_trajectory(fake_align_result, None, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 2
        assert (tmp_path / "alignment_trajectory.pdf").exists()
        assert (tmp_path / "alignment_trajectory.png").exists()
        assert not (tmp_path / "label_trajectory.pdf").exists()

    def test_label_only_writes_label_files(self, fake_label_result, synthetic_dataset_3, tmp_path):
        result = plot_trajectory(None, fake_label_result, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 2
        assert (tmp_path / "label_trajectory.pdf").exists()
        assert (tmp_path / "label_trajectory.png").exists()
        assert not (tmp_path / "alignment_trajectory.pdf").exists()

    def test_both_stages_writes_four_files(self, fake_align_result, fake_label_result, synthetic_dataset_3, tmp_path):
        result = plot_trajectory(fake_align_result, fake_label_result, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 4
        assert (tmp_path / "alignment_trajectory.pdf").exists()
        assert (tmp_path / "label_trajectory.pdf").exists()

    def test_neither_stage_returns_empty_list(self, synthetic_dataset_3, tmp_path):
        result = plot_trajectory(None, None, synthetic_dataset_3, None, tmp_path)
        assert result == []

    def test_label_names_used_when_provided(self, fake_align_result, fake_label_result, synthetic_dataset_3, tmp_path):
        label_names = {0: "T cell", 1: "B cell", 2: "NK cell", 3: "Monocyte"}
        result = plot_trajectory(fake_align_result, fake_label_result, synthetic_dataset_3, label_names, tmp_path)
        assert len(result) == 4
        assert (tmp_path / "label_trajectory.pdf").exists()
        assert (tmp_path / "label_trajectory.pdf").stat().st_size > 0

    def test_no_figure_leak(self, fake_align_result, fake_label_result, synthetic_dataset_3, tmp_path):
        before = len(plt.get_fignums())
        plot_trajectory(fake_align_result, fake_label_result, synthetic_dataset_3, None, tmp_path)
        after = len(plt.get_fignums())
        assert after == before

    # D-08 lower-bound: 1-frame dataset — frame_indices must deduplicate to [0]
    def test_1frame_dataset_returns_two_files_no_index_error(
        self, fake_align_result_1frame, synthetic_dataset_1, tmp_path
    ):
        """D-08: 1-frame dataset produces exactly 2 output files (PDF + PNG) without IndexError."""
        result = plot_trajectory(fake_align_result_1frame, None, synthetic_dataset_1, None, tmp_path)
        assert len(result) == 2
        assert (tmp_path / "alignment_trajectory.pdf").exists()
        assert (tmp_path / "alignment_trajectory.png").exists()

    # D-08 cap: 5-frame dataset — frame_indices selects first/middle/last (no duplicates)
    def test_5frame_dataset_returns_two_files_no_index_error(
        self, fake_align_result_5frames, synthetic_dataset_5, tmp_path
    ):
        """D-08: 5-frame dataset produces exactly 2 output files (PDF + PNG) without IndexError."""
        result = plot_trajectory(fake_align_result_5frames, None, synthetic_dataset_5, None, tmp_path)
        assert len(result) == 2
        assert (tmp_path / "alignment_trajectory.pdf").exists()
        assert (tmp_path / "alignment_trajectory.png").exists()

    def test_empty_dataset_returns_empty_list(self, tmp_path):
        """CR-01: empty dataset returns [] immediately without IndexError."""
        result = plot_trajectory(None, None, {}, None, tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# triptych_csv fixture
# ---------------------------------------------------------------------------


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
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "test_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# TestRenderDatasetTriptych — VIZ-01
# ---------------------------------------------------------------------------


class TestRenderDatasetTriptych:
    """VIZ-01c/d/g: render_dataset_triptych smoke tests."""

    def test_creates_png_file(self, triptych_csv, tmp_path) -> None:
        """VIZ-01c: PNG written at output_dir/name.png with size > 0."""
        out = render_dataset_triptych(triptych_csv, "test_dataset", tmp_path)
        assert (tmp_path / "test_dataset.png").exists()
        assert (tmp_path / "test_dataset.png").stat().st_size > 0

    def test_returns_path(self, triptych_csv, tmp_path) -> None:
        """VIZ-01d: Return value is a pathlib.Path pointing to the PNG."""
        result = render_dataset_triptych(triptych_csv, "test_dataset", tmp_path / "sub")
        assert isinstance(result, Path)
        assert result.suffix == ".png"
        assert result.exists()

    def test_no_figure_leak(self, triptych_csv, tmp_path) -> None:
        """VIZ-01g: plt.close(fig) called — no leaked figure handles."""
        before = len(plt.get_fignums())
        render_dataset_triptych(triptych_csv, "test_dataset", tmp_path)
        after = len(plt.get_fignums())
        assert after == before
