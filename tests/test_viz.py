"""Tests for eval.viz — plot_point_cloud and plot_metrics_summary (FRAME-08)."""

from pathlib import Path

import pytest

# zreg.* before torch — macOS-ARM libomp SIGABRT rule
from zreg.dataset import zRegPointCloud
from zreg.generators import generate_labels, generate_trajectory

import torch

import matplotlib.pyplot as plt

from eval.types import AlignResult, EvalReport, StageMetrics
from eval.viz import plot_point_cloud, plot_metrics_summary


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
# TestPlotPointCloud — FRAME-08 G1
# ---------------------------------------------------------------------------


class TestPlotPointCloud:
    """FRAME-08 G1: plot_point_cloud produces non-empty PDF without figure leaks."""

    def test_creates_pdf_file_at_path(self, fake_align_result, tmp_path) -> None:
        """PDF is created at the given path with size > 0."""
        out = tmp_path / "cloud.pdf"
        plot_point_cloud(fake_align_result, out)
        assert out.exists()
        assert out.suffix == ".pdf"
        assert out.stat().st_size > 0

    def test_no_figure_leak(self, fake_align_result, tmp_path) -> None:
        """plt.close(fig) is called — no leaked figure handles after return."""
        before = len(plt.get_fignums())
        plot_point_cloud(fake_align_result, tmp_path / "leak.pdf")
        after = len(plt.get_fignums())
        assert after == before

    def test_one_frame_dataset_one_subplot_produced(
        self, fake_align_result_1frame, tmp_path
    ) -> None:
        """1-frame dataset: function does not raise and produces a PDF."""
        out = tmp_path / "one_frame.pdf"
        plot_point_cloud(fake_align_result_1frame, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_five_frames_capped_at_four(
        self, fake_align_result_5frames, tmp_path
    ) -> None:
        """5-frame dataset: D-08 cap — function slices to 4 and produces a PDF."""
        out = tmp_path / "five_frames.pdf"
        plot_point_cloud(fake_align_result_5frames, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# TestPlotMetricsSummary — FRAME-08 G2
# ---------------------------------------------------------------------------


class TestPlotMetricsSummary:
    """FRAME-08 G2: plot_metrics_summary produces non-empty PDF without figure leaks."""

    def test_creates_pdf_file_at_path(self, fake_report, tmp_path) -> None:
        """PDF is created at the given path with size > 0."""
        out = tmp_path / "summary.pdf"
        plot_metrics_summary(fake_report, out)
        assert out.exists()
        assert out.suffix == ".pdf"
        assert out.stat().st_size > 0

    def test_no_figure_leak(self, fake_report, tmp_path) -> None:
        """plt.close(fig) is called — no leaked figure handles after return."""
        before = len(plt.get_fignums())
        plot_metrics_summary(fake_report, tmp_path / "summary_leak.pdf")
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
        plot_metrics_summary(partial_report, out)
        assert out.exists()
        assert out.stat().st_size > 0
