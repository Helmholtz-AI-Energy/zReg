"""Tests for eval.viz — plot_trajectory and plot_metrics (FRAME-08)."""

from pathlib import Path
from unittest.mock import patch

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
# TestPlotTrajectory — EXT-02 / VIZ-02
# ---------------------------------------------------------------------------


class TestPlotTrajectory:
    """EXT-02 / VIZ-02: plot_trajectory produces correct files for all stage combinations."""

    def test_align_only_writes_alignment_files(self, fake_align_result, synthetic_dataset_3, tmp_path):
        """Align only, no target: 3 alignment figures = 6 files (source, aligned, superposed)."""
        result = plot_trajectory(fake_align_result, None, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 6
        for stem in [
            "alignment_source_trajectory",
            "alignment_aligned_trajectory",
            "alignment_superposed_trajectory",
        ]:
            assert (tmp_path / f"{stem}.pdf").exists()
            assert (tmp_path / f"{stem}.png").exists()
        # Old stem must NOT be written
        assert not (tmp_path / "alignment_trajectory.pdf").exists()
        assert not (tmp_path / "label_trajectory.pdf").exists()

    def test_label_only_writes_label_files(self, fake_label_result, synthetic_dataset_3, tmp_path):
        result = plot_trajectory(None, fake_label_result, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 4
        for stem in ["label_source_trajectory", "label_target_trajectory"]:
            assert (tmp_path / f"{stem}.pdf").exists()
            assert (tmp_path / f"{stem}.png").exists()
        assert not (tmp_path / "label_trajectory.pdf").exists()
        assert not (tmp_path / "alignment_trajectory.pdf").exists()

    def test_both_stages_writes_ten_files(
        self, fake_align_result, fake_label_result, synthetic_dataset_3, tmp_path
    ):
        """Align + label, no target: 6 alignment + 4 label = 10 files."""
        result = plot_trajectory(fake_align_result, fake_label_result, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 10
        assert (tmp_path / "alignment_source_trajectory.pdf").exists()
        assert (tmp_path / "label_source_trajectory.pdf").exists()
        assert (tmp_path / "label_target_trajectory.pdf").exists()
        assert not (tmp_path / "label_trajectory.pdf").exists()

    def test_neither_stage_returns_empty_list(self, synthetic_dataset_3, tmp_path):
        result = plot_trajectory(None, None, synthetic_dataset_3, None, tmp_path)
        assert result == []

    def test_label_names_used_when_provided(
        self, fake_align_result, fake_label_result, synthetic_dataset_3, tmp_path
    ):
        """Label names update applies; total files = 10 (6 align + 4 label, no target)."""
        label_names = {0: "T cell", 1: "B cell", 2: "NK cell", 3: "Monocyte"}
        result = plot_trajectory(fake_align_result, fake_label_result, synthetic_dataset_3, label_names, tmp_path)
        assert len(result) == 10
        assert (tmp_path / "label_source_trajectory.pdf").exists()
        assert (tmp_path / "label_target_trajectory.pdf").exists()
        assert (tmp_path / "label_target_trajectory.pdf").stat().st_size > 0

    def test_no_figure_leak(self, fake_align_result, fake_label_result, synthetic_dataset_3, tmp_path):
        before = len(plt.get_fignums())
        plot_trajectory(fake_align_result, fake_label_result, synthetic_dataset_3, None, tmp_path)
        after = len(plt.get_fignums())
        assert after == before

    # D-08 lower-bound: 1-frame dataset — frame_indices must deduplicate to [0]
    def test_1frame_dataset_returns_six_files_no_index_error(
        self, fake_align_result_1frame, synthetic_dataset_1, tmp_path
    ):
        """D-08: 1-frame dataset produces 6 output files without IndexError."""
        result = plot_trajectory(fake_align_result_1frame, None, synthetic_dataset_1, None, tmp_path)
        assert len(result) == 6
        assert (tmp_path / "alignment_source_trajectory.pdf").exists()
        assert (tmp_path / "alignment_source_trajectory.png").exists()
        assert not (tmp_path / "alignment_trajectory.pdf").exists()

    # D-08 cap: 5-frame dataset — frame_indices selects first/middle/last (no duplicates)
    def test_5frame_dataset_returns_six_files_no_index_error(
        self, fake_align_result_5frames, synthetic_dataset_5, tmp_path
    ):
        """D-08: 5-frame dataset produces 6 output files without IndexError."""
        result = plot_trajectory(fake_align_result_5frames, None, synthetic_dataset_5, None, tmp_path)
        assert len(result) == 6
        assert (tmp_path / "alignment_source_trajectory.pdf").exists()
        assert not (tmp_path / "alignment_trajectory.pdf").exists()

    def test_empty_dataset_returns_empty_list(self, tmp_path):
        """CR-01: empty dataset returns [] immediately without IndexError."""
        result = plot_trajectory(None, None, {}, None, tmp_path)
        assert result == []

    # --- New tests for 4-figure alignment behavior (VIZ-02) ---

    def test_alignment_with_target_writes_8_files(self, fake_align_result, synthetic_dataset_3, tmp_path):
        """When target is provided, all 4 alignment figures (8 files) are written."""
        result = plot_trajectory(
            fake_align_result, None, synthetic_dataset_3, None, tmp_path,
            target=synthetic_dataset_3,
        )
        assert len(result) == 8
        for stem in [
            "alignment_source_trajectory",
            "alignment_target_trajectory",
            "alignment_aligned_trajectory",
            "alignment_superposed_trajectory",
        ]:
            assert (tmp_path / f"{stem}.pdf").exists()
            assert (tmp_path / f"{stem}.png").exists()

    def test_alignment_without_target_skips_target_figure(self, fake_align_result, synthetic_dataset_3, tmp_path):
        """When target is None, target figure is skipped — 3 alignment figures (6 files)."""
        result = plot_trajectory(fake_align_result, None, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 6
        assert (tmp_path / "alignment_source_trajectory.pdf").exists()
        assert (tmp_path / "alignment_aligned_trajectory.pdf").exists()
        assert (tmp_path / "alignment_superposed_trajectory.pdf").exists()
        assert not (tmp_path / "alignment_target_trajectory.pdf").exists()
        assert not (tmp_path / "alignment_trajectory.pdf").exists()  # old stem gone

    def test_no_figure_leak_4panel(self, fake_align_result, synthetic_dataset_3, tmp_path):
        """4-figure alignment mode produces no leaked matplotlib figure handles."""
        before = len(plt.get_fignums())
        plot_trajectory(
            fake_align_result, None, synthetic_dataset_3, None, tmp_path,
            target=synthetic_dataset_3,
        )
        after = len(plt.get_fignums())
        assert after == before

    def test_label_writes_two_figure_pairs(self, fake_label_result, synthetic_dataset_3, tmp_path):
        """VIZ-03: label branch writes exactly 4 files — source and target figure pairs."""
        result = plot_trajectory(None, fake_label_result, synthetic_dataset_3, None, tmp_path)
        assert len(result) == 4
        for stem in ["label_source_trajectory", "label_target_trajectory"]:
            assert (tmp_path / f"{stem}.pdf").exists()
            assert (tmp_path / f"{stem}.png").exists()
        assert not (tmp_path / "label_trajectory.pdf").exists()

    def test_label_source_uses_id_when_available(self, tmp_path):
        """Source figure uses pc['id'] when present (not color)."""
        ds = {
            0: zRegPointCloud(
                pos=torch.randn(20, 3),
                color=torch.zeros(20, dtype=torch.long),
                id=torch.ones(20, dtype=torch.long),
            ),
            1: zRegPointCloud(
                pos=torch.randn(20, 3),
                color=torch.zeros(20, dtype=torch.long),
                id=torch.ones(20, dtype=torch.long),
            ),
            2: zRegPointCloud(
                pos=torch.randn(20, 3),
                color=torch.zeros(20, dtype=torch.long),
                id=torch.ones(20, dtype=torch.long),
            ),
        }
        lr = LabelResult(
            transferred_labels={k: torch.zeros(20, dtype=torch.long) for k in ds},
            params_used={},
        )
        result = plot_trajectory(None, lr, ds, None, tmp_path)
        assert (tmp_path / "label_source_trajectory.pdf").exists()
        assert len(result) == 4

    def test_label_no_figure_leak(self, fake_label_result, synthetic_dataset_3, tmp_path):
        """No leaked figure handles after label-only run."""
        before = len(plt.get_fignums())
        plot_trajectory(None, fake_label_result, synthetic_dataset_3, None, tmp_path)
        after = len(plt.get_fignums())
        assert after == before


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


# ---------------------------------------------------------------------------
# Coverage gap tests for viz.py
# ---------------------------------------------------------------------------


class TestVizCoverageGaps:
    """Coverage gaps: subsampling >4000 pts, empty CSV, ValueError paths."""

    def test_plot_trajectory_large_dataset_subsamples(self, tmp_path):
        """_subsample helper — subsampling when source/aligned pos > 4000 points."""
        # Create large dataset with >4000 points per frame
        large_ds = {
            0: zRegPointCloud(pos=torch.randn(5000, 3), color=None, id=None),
            1: zRegPointCloud(pos=torch.randn(5000, 3), color=None, id=None),
            2: zRegPointCloud(pos=torch.randn(5000, 3), color=None, id=None),
        }
        align_result = AlignResult(
            aligned_cloud=large_ds,
            warp_path=[(0, 0), (1, 1), (2, 2)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        paths = plot_trajectory(align_result, None, large_ds, None, tmp_path)
        # No target: 3 figures = 6 files
        assert len(paths) == 6
        assert (tmp_path / "alignment_source_trajectory.pdf").exists()
        assert (tmp_path / "alignment_aligned_trajectory.pdf").exists()
        assert (tmp_path / "alignment_superposed_trajectory.pdf").exists()

    def test_plot_metrics_large_label_dataset_subsamples(self, tmp_path):
        """viz.py label branch — subsampling when label points > 4000."""
        large_ds = {
            0: zRegPointCloud(pos=torch.randn(5000, 3), color=torch.zeros(5000, dtype=torch.long), id=None),
            1: zRegPointCloud(pos=torch.randn(5000, 3), color=torch.zeros(5000, dtype=torch.long), id=None),
            2: zRegPointCloud(pos=torch.randn(5000, 3), color=torch.zeros(5000, dtype=torch.long), id=None),
        }
        label_result = LabelResult(
            transferred_labels={k: torch.zeros(5000, dtype=torch.long) for k in large_ds},
            params_used={},
        )
        paths = plot_trajectory(None, label_result, large_ds, None, tmp_path)
        assert len(paths) == 4
        assert (tmp_path / "label_source_trajectory.pdf").exists()
        assert (tmp_path / "label_target_trajectory.pdf").exists()
        assert not (tmp_path / "label_trajectory.pdf").exists()

    def test_render_dataset_triptych_empty_csv_raises(self, tmp_path):
        """viz.py — ValueError when CSV has no time frames (header only)."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("x,y,z,t\n")
        with pytest.raises(ValueError, match="no time frames"):
            render_dataset_triptych(empty_csv, "empty", tmp_path)

    def test_render_dataset_triptych_no_matching_rows_raises(self, tmp_path):
        """viz.py — ValueError when no chunks match wanted t values.

        Strategy: create a real CSV (t=999), then patch pandas.read_csv so that
        the second call (chunked read) returns a chunk with t=888 (no match).
        """
        import pandas as pd

        csv_path = tmp_path / "mismatch.csv"
        df = pd.DataFrame([{"x": 1.0, "y": 2.0, "z": 3.0, "t": 999}])
        df.to_csv(csv_path, index=False)

        fake_chunk = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0], "t": [888]})
        real_read_csv = pd.read_csv
        call_count = [0]

        def intercepted_read_csv(path, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: frame discovery (usecols=["t"]) — use real impl
                return real_read_csv(path, **kwargs)
            # Second call: chunked read — return non-matching chunk
            return iter([fake_chunk])

        with patch("pandas.read_csv", side_effect=intercepted_read_csv):
            with pytest.raises(ValueError, match="no rows match"):
                render_dataset_triptych(csv_path, "mismatch", tmp_path)

    def test_render_dataset_triptych_large_points_subsamples(self, tmp_path):
        """viz.py — subsampling when n_pts > 4000 in triptych."""
        import pandas as pd
        rows = []
        for t in [1, 2, 3]:
            for _ in range(5000):
                rows.append({"x": 0.5, "y": 0.5, "z": 0.5, "t": t})
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "large.csv"
        df.to_csv(csv_path, index=False)
        result = render_dataset_triptych(csv_path, "large", tmp_path)
        assert result.exists()
