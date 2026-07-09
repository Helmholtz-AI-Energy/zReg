"""Tests for zreg.metrics.alignment module (Plan 13-01, Tasks 2 & 3).

Covers:
- chamfer: known-distance assertions, symmetric, squared flag, shape guard, NaN guard
- hausdorff: known-distance, identical clouds, torch.quantile usage
- path_smoothness: zero-variance path, edge cases (empty, 1 point, 2 points)
- knn_consistency: all-same-label = 1.0, validation errors
- temporal_stability: empty/single list, identity transforms, known translation
"""

import pytest
import torch

from zreg.transforms import RigidTransformation, AffineTransformation


class TestChamfer:
    """Tests for the chamfer function."""

    def test_identical_clouds_zero(self):
        """chamfer(A, A) must be zero for any finite (N, 3) tensor."""
        from zreg.metrics.alignment import chamfer
        a = torch.randn(20, 3)
        result = chamfer(a, a)
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-5)

    def test_known_distance_unit(self):
        """Single-point clouds distance 1.0 apart returns 1.0."""
        from zreg.metrics.alignment import chamfer
        a = torch.tensor([[0.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0]])
        result = chamfer(a, b)
        assert abs(result.item() - 1.0) < 1e-5

    def test_squared_flag_larger_than_plain(self):
        """squared=True gives larger result when distance > 1."""
        from zreg.metrics.alignment import chamfer
        a = torch.tensor([[0.0, 0.0, 0.0]])
        b = torch.tensor([[2.0, 0.0, 0.0]])
        plain = chamfer(a, b, squared=False)
        sq = chamfer(a, b, squared=True)
        assert sq.item() > plain.item()

    def test_returns_scalar_tensor(self):
        """Must return a 0-dim Tensor."""
        from zreg.metrics.alignment import chamfer
        a = torch.randn(10, 3)
        result = chamfer(a, a)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_invalid_shape_raises_value_error(self):
        """Non-(N,3) source raises ValueError mentioning shape."""
        from zreg.metrics.alignment import chamfer
        x = torch.randn(10, 4)
        y = torch.randn(10, 3)
        with pytest.raises(ValueError):
            chamfer(x, y)

    def test_invalid_target_shape_raises_value_error(self):
        """Non-(M,3) target raises ValueError even when source is valid."""
        from zreg.metrics.alignment import chamfer
        x = torch.randn(5, 3)
        y = torch.randn(5, 2)
        with pytest.raises(ValueError):
            chamfer(x, y)

    def test_empty_source_raises_value_error(self):
        """alignment.py:65 — empty source (N=0) raises ValueError."""
        from zreg.metrics.alignment import chamfer
        x = torch.zeros(0, 3)
        y = torch.randn(5, 3)
        with pytest.raises(ValueError, match="non-empty"):
            chamfer(x, y)

    def test_empty_target_raises_value_error(self):
        """alignment.py:67 — empty target (M=0) raises ValueError."""
        from zreg.metrics.alignment import chamfer
        x = torch.randn(5, 3)
        y = torch.zeros(0, 3)
        with pytest.raises(ValueError, match="non-empty"):
            chamfer(x, y)

    def test_nan_raises_value_error(self):
        """NaN in input raises ValueError (via _validate_tensors)."""
        from zreg.metrics.alignment import chamfer
        a = torch.tensor([[float("nan"), 0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="NaN"):
            chamfer(a, b)

    def test_output_on_same_device(self):
        """Output is on the same device as input (CPU test)."""
        from zreg.metrics.alignment import chamfer
        a = torch.randn(10, 3)
        b = torch.randn(10, 3)
        result = chamfer(a, b)
        assert result.device == a.device


class TestHausdorff:
    """Tests for the hausdorff function."""

    def test_identical_clouds_zero(self):
        """hausdorff(A, A) must be near zero."""
        from zreg.metrics.alignment import hausdorff
        a = torch.randn(20, 3)
        result = hausdorff(a, a)
        assert abs(result.item()) < 1e-5

    def test_single_point_distance(self):
        """Single-point clouds distance 1.0 returns 1.0 (100th percentile)."""
        from zreg.metrics.alignment import hausdorff
        a = torch.tensor([[0.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0]])
        # percentile=95 on a 1-element set = that element = 1.0
        result = hausdorff(a, b, percentile=95.0)
        assert abs(result.item() - 1.0) < 1e-5

    def test_no_numpy_percentile(self):
        """alignment.py must not use np.percentile or import numpy."""
        import pathlib
        src = pathlib.Path(
            __file__
        ).parent.parent / "src" / "zreg" / "metrics" / "alignment.py"
        text = src.read_text()
        assert "np.percentile" not in text
        assert "import numpy" not in text

    def test_uses_torch_quantile(self):
        """alignment.py must use torch.quantile."""
        import pathlib
        src = pathlib.Path(
            __file__
        ).parent.parent / "src" / "zreg" / "metrics" / "alignment.py"
        text = src.read_text()
        assert "torch.quantile(" in text

    def test_invalid_shape_raises(self):
        """Non-(N,3) input raises ValueError."""
        from zreg.metrics.alignment import hausdorff
        x = torch.randn(5, 2)
        y = torch.randn(5, 3)
        with pytest.raises(ValueError):
            hausdorff(x, y)

    def test_invalid_target_shape_raises(self):
        """Non-(M,3) target raises ValueError even when source is valid."""
        from zreg.metrics.alignment import hausdorff
        x = torch.randn(5, 3)
        y = torch.randn(5, 4)
        with pytest.raises(ValueError):
            hausdorff(x, y)

    def test_empty_source_raises(self):
        """alignment.py:127 — empty source (N=0) raises ValueError."""
        from zreg.metrics.alignment import hausdorff
        x = torch.zeros(0, 3)
        y = torch.randn(5, 3)
        with pytest.raises(ValueError, match="non-empty"):
            hausdorff(x, y)

    def test_empty_target_raises(self):
        """alignment.py:129 — empty target (M=0) raises ValueError."""
        from zreg.metrics.alignment import hausdorff
        x = torch.randn(5, 3)
        y = torch.zeros(0, 3)
        with pytest.raises(ValueError, match="non-empty"):
            hausdorff(x, y)

    def test_percentile_out_of_range_raises(self):
        """alignment.py:131 — percentile outside [0, 100] raises ValueError."""
        from zreg.metrics.alignment import hausdorff
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        with pytest.raises(ValueError, match="percentile"):
            hausdorff(x, y, percentile=101.0)


class TestPathSmoothness:
    """Tests for the path_smoothness function."""

    def test_linear_path_zero_variance(self):
        """Perfectly linear path has zero smoothness (uniform slopes)."""
        from zreg.metrics.alignment import path_smoothness
        path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        result = path_smoothness(path)
        assert result == 0.0

    def test_returns_float(self):
        """Must return a plain Python float."""
        from zreg.metrics.alignment import path_smoothness
        result = path_smoothness([(0, 0), (1, 1), (2, 2)])
        assert isinstance(result, float)

    def test_empty_path_returns_zero(self):
        """Empty path returns 0.0."""
        from zreg.metrics.alignment import path_smoothness
        assert path_smoothness([]) == 0.0

    def test_single_point_returns_zero(self):
        """Single-point path returns 0.0."""
        from zreg.metrics.alignment import path_smoothness
        assert path_smoothness([(0, 0)]) == 0.0

    def test_two_points_returns_zero(self):
        """Two-point path (one slope) returns 0.0 — need >=2 slopes for variance."""
        from zreg.metrics.alignment import path_smoothness
        assert path_smoothness([(0, 0), (1, 1)]) == 0.0

    def test_varying_slopes_nonzero(self):
        """Varying slopes produce non-zero smoothness.

        Need >= 4 points to get >= 2 slope-deltas whose variance can be > 0.
        With 3 points there are 2 slopes but only 1 delta — variance of a
        single value is always 0.0.
        """
        from zreg.metrics.alignment import path_smoothness
        # 4 points → slopes: 1.0, 2.0, 0.5 → deltas: [1.0, -1.5] → variance > 0
        path = [(0, 0), (1, 1), (2, 3), (3, 3)]
        result = path_smoothness(path)
        assert result > 0.0


class TestKnnConsistency:
    """Tests for the knn_consistency function."""

    def test_all_same_labels_returns_one(self):
        """All-same-label point cloud returns 1.0."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(50, 3)
        lbl = torch.zeros(50, dtype=torch.long)
        assert abs(knn_consistency(pts, lbl, k=5) - 1.0) < 1e-9

    def test_returns_float(self):
        """Must return a Python float."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(20, 3)
        lbl = torch.zeros(20, dtype=torch.long)
        result = knn_consistency(pts, lbl, k=3)
        assert isinstance(result, float)

    def test_in_range_zero_to_one(self):
        """Score must be in [0, 1]."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(30, 3)
        lbl = torch.randint(0, 3, (30,))
        result = knn_consistency(pts, lbl, k=5)
        assert 0.0 <= result <= 1.0

    def test_shape_mismatch_raises(self):
        """points.shape[0] != labels.shape[0] raises ValueError."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(10, 3)
        lbl = torch.zeros(8, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=3)

    def test_k_too_large_raises(self):
        """k >= N raises ValueError."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(5, 3)
        lbl = torch.zeros(5, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=5)  # k must be < N=5

    def test_invalid_points_shape_raises(self):
        """points not (N, 3) raises ValueError."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(10, 2)
        lbl = torch.zeros(10, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=3)

    def test_invalid_labels_shape_raises(self):
        """2D labels tensor raises ValueError."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(10, 3)
        lbl = torch.zeros(10, 10, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=3)

    def test_k_zero_raises(self):
        """k < 1 raises ValueError."""
        from zreg.metrics.label_transfer import knn_consistency
        pts = torch.randn(10, 3)
        lbl = torch.zeros(10, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=0)

    def test_uses_detach_cpu_numpy(self):
        """label_transfer.py must contain .detach().cpu().numpy() for CPU entry."""
        import pathlib
        src = pathlib.Path(
            __file__
        ).parent.parent / "src" / "zreg" / "metrics" / "label_transfer.py"
        text = src.read_text()
        assert ".detach().cpu().numpy()" in text


class TestTemporalStability:
    """Tests for the temporal_stability function."""

    def test_empty_list_returns_zero(self):
        """Empty transform list returns tensor(0.0)."""
        from zreg.metrics.label_transfer import temporal_stability
        result = temporal_stability([])
        assert isinstance(result, torch.Tensor)
        assert result.item() == 0.0

    def test_single_transform_returns_zero(self):
        """Single transform returns tensor(0.0)."""
        from zreg.metrics.label_transfer import temporal_stability
        tf = RigidTransformation()
        result = temporal_stability([tf])
        assert result.item() == 0.0

    def test_identical_transforms_zero(self):
        """Consecutive identical transforms produce zero Frobenius difference."""
        from zreg.metrics.label_transfer import temporal_stability
        tf = RigidTransformation()
        result = temporal_stability([tf, tf])
        assert abs(result.item()) < 1e-5

    def test_known_translation_norm(self):
        """Two rigid transforms differing only by translation (1,0,0) => norm 1.0."""
        from zreg.metrics.label_transfer import temporal_stability
        tf1 = RigidTransformation(t=torch.zeros(3))
        tf2 = RigidTransformation(t=torch.tensor([1.0, 0.0, 0.0]))
        result = temporal_stability([tf1, tf2])
        assert abs(result.item() - 1.0) < 1e-5

    def test_unsupported_type_raises_type_error(self):
        """Non-transform types raise TypeError mentioning 'Unsupported'."""
        from zreg.metrics.label_transfer import temporal_stability
        with pytest.raises(TypeError, match="[Uu]nsupported"):
            temporal_stability([1, 2])

    def test_no_to_matrix_calls(self):
        """alignment.py must not call .to_matrix() on any transform."""
        import pathlib
        src = pathlib.Path(
            __file__
        ).parent.parent / "src" / "zreg" / "metrics" / "alignment.py"
        text = src.read_text()
        assert ".to_matrix(" not in text

    def test_uses_frobenius_norm(self):
        """label_transfer.py must compute Frobenius norm via torch.norm(..., p='fro')."""
        import pathlib
        src = pathlib.Path(
            __file__
        ).parent.parent / "src" / "zreg" / "metrics" / "label_transfer.py"
        text = src.read_text()
        assert 'p="fro"' in text or "p='fro'" in text

    def test_affine_transform_accepted(self):
        """AffineTransformation is accepted without error."""
        from zreg.metrics.label_transfer import temporal_stability
        tf = AffineTransformation()
        result = temporal_stability([tf, tf])
        assert abs(result.item()) < 1e-5
