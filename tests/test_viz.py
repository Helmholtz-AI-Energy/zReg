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

    generate_labels produces label tensors of shape (N,) with dtype torch.int64,
    so we use them directly as 1-D long tensors.
    """
    return LabelResult(
        transferred_labels={
            k: synthetic_dataset_3[k]['label'].long()
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
    """FRAME-08 G2: plot_metrics produces non-empty PDF+PNG without figure leaks."""

    def test_creates_pdf_and_png_files(self, fake_report, tmp_path) -> None:
        """PDF and PNG are both created at path with size > 0."""
        out = tmp_path / "summary.pdf"
        result = plot_metrics(fake_report, out)
        assert len(result) == 2
        assert (tmp_path / "summary.pdf").exists()
        assert (tmp_path / "summary.pdf").stat().st_size > 0
        assert (tmp_path / "summary.png").exists()
        assert (tmp_path / "summary.png").stat().st_size > 0

    def test_returns_path_list(self, fake_report, tmp_path) -> None:
        """Return value is a list of two absolute path strings."""
        out = tmp_path / "summary.pdf"
        result = plot_metrics(fake_report, out)
        assert isinstance(result, list)
        assert len(result) == 2
        assert any(r.endswith(".pdf") for r in result)
        assert any(r.endswith(".png") for r in result)

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
        result = plot_metrics(partial_report, out)
        assert len(result) == 2
        assert (tmp_path / "partial.pdf").exists()
        assert (tmp_path / "partial.png").exists()


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
        """Source figure renders grey (no label field) without error; transferred_labels provide target colors."""
        # label=None: _get_source_labels returns None → grey source figure (CLN-02: no id fallback).
        # transferred_labels=zeros means label 0 colors the target figure.
        # Both figures must render without error.
        ds = {
            0: zRegPointCloud(pos=torch.randn(20, 3), label=None, id=torch.ones(20, dtype=torch.long)),
            1: zRegPointCloud(pos=torch.randn(20, 3), label=None, id=torch.ones(20, dtype=torch.long)),
            2: zRegPointCloud(pos=torch.randn(20, 3), label=None, id=torch.ones(20, dtype=torch.long)),
        }
        lr = LabelResult(
            transferred_labels={k: torch.zeros(20, dtype=torch.long) for k in ds},
            params_used={},
        )
        result = plot_trajectory(None, lr, ds, None, tmp_path)
        assert len(result) == 4
        assert (tmp_path / "label_source_trajectory.pdf").exists()
        assert (tmp_path / "label_target_trajectory.pdf").exists()

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


class TestVizSubsamplingAndRenderCoverage:
    """Coverage gaps: subsampling >4000 pts, empty CSV, ValueError paths."""

    def test_plot_trajectory_large_dataset_subsamples(self, tmp_path):
        """_subsample helper — subsampling when source/aligned pos > 4000 points."""
        # Create large dataset with >4000 points per frame
        large_ds = {
            0: zRegPointCloud(pos=torch.randn(5000, 3), label=None, id=None),
            1: zRegPointCloud(pos=torch.randn(5000, 3), label=None, id=None),
            2: zRegPointCloud(pos=torch.randn(5000, 3), label=None, id=None),
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

    def test_plot_trajectory_large_label_dataset_subsamples(self, tmp_path):
        """viz.py label branch — subsampling when label points > 4000."""
        large_ds = {
            0: zRegPointCloud(pos=torch.randn(5000, 3), label=torch.zeros(5000, dtype=torch.long), id=None),
            1: zRegPointCloud(pos=torch.randn(5000, 3), label=torch.zeros(5000, dtype=torch.long), id=None),
            2: zRegPointCloud(pos=torch.randn(5000, 3), label=torch.zeros(5000, dtype=torch.long), id=None),
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


# ---------------------------------------------------------------------------
# TestGetSourceLabels — CLN-02 regression
# ---------------------------------------------------------------------------


class TestGetSourceLabels:
    """CLN-02 regression: _get_source_labels reads 'label' field only; no 'id' fallback."""

    def test_returns_label_field_when_set(self):
        """_get_source_labels returns label tensor when pc['label'] is set."""
        from eval.viz import _get_source_labels
        pc = zRegPointCloud(pos=torch.zeros(5, 3), label=torch.tensor([0, 1, 2, 1, 0]), id=None)
        result = _get_source_labels(pc)
        assert result is not None
        assert result.dtype == torch.long

    def test_returns_none_when_label_absent(self):
        """CLN-02 regression: _get_source_labels returns None when label=None (no id fallback)."""
        from eval.viz import _get_source_labels
        pc = zRegPointCloud(pos=torch.zeros(5, 3), label=None, id=torch.arange(5))
        result = _get_source_labels(pc)
        assert result is None  # id fallback removed; label=None → return None


# ---------------------------------------------------------------------------
# Coverage gaps: viz.py lines 90->89, 439->447, 486, 503, 531
# ---------------------------------------------------------------------------


class TestVizCoverageGaps:
    """Lines 90->89, 439->447, 486, 503, 531 — edge cases in plot_trajectory."""

    def _make_pc(self, n=5):
        return zRegPointCloud(
            pos=torch.randn(n, 3),
            label=torch.zeros(n, dtype=torch.long),
        )

    def test_duplicate_target_in_warp_path_skips_second(self, tmp_path):
        """viz.py:90->89 — duplicate target index in warp_path: second mapping skipped."""
        ds = {0: self._make_pc(), 1: self._make_pc()}
        aligned = {0: self._make_pc(), 1: self._make_pc()}
        # warp_path: target 0 appears twice (from source 0 and source 1)
        align_result = AlignResult(
            aligned_cloud=aligned,
            warp_path=[(0, 0), (1, 0), (1, 1)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        result = plot_trajectory(align_result, None, ds, None, tmp_path)
        assert len(result) > 0

    def test_target_with_no_overlapping_keys_skips_target_figure(self, tmp_path):
        """viz.py:439->447 — target provided but no frame keys overlap with align_frame_indices."""
        ds = {0: self._make_pc(), 1: self._make_pc(), 2: self._make_pc()}
        aligned = {0: self._make_pc(), 1: self._make_pc(), 2: self._make_pc()}
        # target has completely different keys → per_frame_target is empty → target figure skipped
        target = {10: self._make_pc(), 11: self._make_pc(), 12: self._make_pc()}
        align_result = AlignResult(
            aligned_cloud=aligned,
            warp_path=[(0, 0), (1, 1), (2, 2)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        result = plot_trajectory(align_result, None, ds, None, tmp_path, target=target)
        # No target figure written (3 figures × 2 formats = 6, not 8)
        from pathlib import Path
        assert not (Path(tmp_path) / "alignment_target_trajectory.pdf").exists()

    def test_label_frame_not_in_dataset_skipped_when_in_target(self, tmp_path):
        """viz.py:486+503 — label frame not in dataset but present in target: continue branches hit."""
        ds = {0: self._make_pc()}
        target_extra = {0: self._make_pc(), 99: self._make_pc()}
        # label_result has key 99 which is not in dataset
        label_result = LabelResult(
            transferred_labels={
                0: torch.zeros(5, dtype=torch.long),
                99: torch.zeros(5, dtype=torch.long),
            },
            params_used={},
        )
        # Should complete without error; fk=99 skipped in union_labels and source_pos_map
        result = plot_trajectory(None, label_result, ds, None, tmp_path, target=target_extra)
        assert isinstance(result, list)

    def test_label_frame_not_in_dataset_or_target_raises_key_error(self, tmp_path):
        """viz.py:531 — label frame not in dataset, target, or align_result raises KeyError."""
        ds = {0: self._make_pc()}
        label_result = LabelResult(
            transferred_labels={99: torch.zeros(5, dtype=torch.long)},
            params_used={},
        )
        with pytest.raises(KeyError):
            plot_trajectory(None, label_result, ds, None, tmp_path)

    def test_disjoint_source_target_keys_skips_source_label_figure(self, tmp_path):
        """Empty source_frame_indices guard — no blank source figure written.

        When transferred_labels keys are entirely absent from dataset (disjoint
        key sets, e.g. paired alignment where source frames are {0,1,2} and
        target frames are {10,11,12}), source_frame_indices is empty and the
        source label figure must be skipped rather than saved as a blank image.
        Only the target label figure (2 files) is written.
        """
        source = {k: self._make_pc() for k in range(3)}
        target = {k: self._make_pc() for k in range(10, 13)}
        label_result = LabelResult(
            transferred_labels={k: torch.zeros(5, dtype=torch.long) for k in range(10, 13)},
            params_used={},
        )
        result = plot_trajectory(None, label_result, source, None, tmp_path, target=target)
        # source label figure must NOT be written (disjoint keys → skipped)
        assert not (tmp_path / "label_source_trajectory.pdf").exists()
        assert not (tmp_path / "label_source_trajectory.png").exists()
        # target label figure IS written
        assert (tmp_path / "label_target_trajectory.pdf").exists()
        assert (tmp_path / "label_target_trajectory.png").exists()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Regression tests for source/target length mismatch (ee468eb + 5f98e11)
# ---------------------------------------------------------------------------


class TestVizMismatchedSourceTarget:
    """All prior tests use equal-length source/target with identity warp paths.
    These tests cover the distinct-length case that triggered two production bugs.
    """

    def _make_pc(self, n=20, n_labels=None):
        label = torch.randint(0, n_labels, (n,)) if n_labels is not None else None
        return zRegPointCloud(pos=torch.randn(n, 3), label=label)

    def test_source_shorter_than_aligned_cloud_no_key_error(self, tmp_path):
        """Regression for ee468eb: source has 5 frames (keys 0-4) but aligned_cloud
        has 8 frames (keys 0-7).  warp_path carries subsampled indices (not original
        frame keys), so align_frame_index 7 must map to a valid source key (0-4).
        Without the fix this raises KeyError: 7."""
        source = {k: self._make_pc() for k in range(5)}
        aligned = {k: self._make_pc() for k in range(8)}
        # Subsampled warp_path (step=2):
        #   source_sub {0,1,2} → original frames 0,2,4
        #   target_sub {0,1,2,3} → original frames 0,2,4,6
        warp_path = [(0, 0), (0, 1), (1, 2), (2, 3)]
        align_result = AlignResult(
            aligned_cloud=aligned,
            warp_path=warp_path,
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        result = plot_trajectory(align_result, None, source, None, tmp_path)
        assert len(result) == 6
        assert (tmp_path / "alignment_source_trajectory.pdf").exists()

    def test_transferred_label_unique_to_frame_outside_source_included_in_palette(self, tmp_path):
        """Regression for 5f98e11: transferred labels for a frame absent from the
        source dataset must still be added to the colour palette.  Frame 5 is not
        in source (keys 0-2) and carries class 9 which appears nowhere else.
        Without the fix this raises KeyError: 9."""
        source = {k: self._make_pc(n_labels=3) for k in range(3)}  # keys 0, 1, 2
        transferred = {
            0: torch.zeros(20, dtype=torch.long),          # class 0
            2: torch.zeros(20, dtype=torch.long),          # class 0
            5: torch.full((20,), 9, dtype=torch.long),     # class 9 — unique to frame 5
        }
        target = {5: self._make_pc()}  # needed so target_pos_map can resolve fk=5
        label_result = LabelResult(transferred_labels=transferred, params_used={})
        result = plot_trajectory(None, label_result, source, None, tmp_path, target=target)
        assert len(result) == 4
        assert (tmp_path / "label_target_trajectory.pdf").exists()
