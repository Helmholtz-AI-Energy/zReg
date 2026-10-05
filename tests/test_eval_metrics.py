"""Tests for zreg.evaluation metrics (alignment and label_transfer)."""

import inspect

import pytest
import torch

from zreg.evaluation.alignment import chamfer, hausdorff, path_smoothness
from zreg.evaluation.label_transfer import compute_f1, knn_consistency, temporal_stability
from zreg.core.transforms import RigidTransformation, AffineTransformation


# ---------------------------------------------------------------------------
# TestChamfer
# ---------------------------------------------------------------------------


class TestChamfer:
    """Tests for the chamfer distance function."""

    def test_identical_clouds_zero(self):
        """Chamfer distance of a cloud with itself is zero."""
        x = torch.randn(20, 3)
        assert torch.allclose(chamfer(x, x), torch.zeros(()), atol=1e-5)

    def test_known_distance_single_pair(self):
        """Chamfer distance for a single pair 1 unit apart equals 1.0."""
        a = torch.tensor([[0., 0., 0.]])
        b = torch.tensor([[1., 0., 0.]])
        assert torch.allclose(chamfer(a, b), torch.tensor(1.0), atol=1e-5)

    def test_squared_flag_increases_for_d_gt_1(self):
        """squared=True increases result when distance > 1."""
        a = torch.tensor([[0., 0., 0.]])
        b = torch.tensor([[2., 0., 0.]])
        assert chamfer(a, b, squared=True) > chamfer(a, b, squared=False)

    def test_squared_flag_decreases_for_d_lt_1(self):
        """squared=True decreases result when distance < 1."""
        a = torch.tensor([[0., 0., 0.]])
        b = torch.tensor([[0.5, 0., 0.]])
        assert chamfer(a, b, squared=True) < chamfer(a, b, squared=False)

    def test_invalid_shape_raises_value_error(self):
        """Non-(N,3) source raises ValueError mentioning 'shape'."""
        x = torch.randn(10, 4)
        y = torch.randn(10, 3)
        with pytest.raises(ValueError, match="shape"):
            chamfer(x, y)

    def test_nan_input_raises_value_error(self):
        """Tensor with NaN raises ValueError mentioning 'NaN'."""
        x = torch.tensor([[float("nan"), 0., 0.]])
        y = torch.tensor([[0., 0., 0.]])
        with pytest.raises(ValueError, match="NaN"):
            chamfer(x, y)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_mismatch_raises_value_error(self):  # pragma: no cover
        """CPU source vs CUDA target raises ValueError mentioning 'device'."""
        a = torch.randn(5, 3)
        b = torch.randn(5, 3).cuda()
        with pytest.raises(ValueError, match="device"):
            chamfer(a, b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_chamfer_on_cuda(self):  # pragma: no cover
        """Result device matches CUDA inputs."""
        a = torch.randn(10, 3).cuda()
        b = torch.randn(10, 3).cuda()
        assert chamfer(a, b).device.type == "cuda"


# ---------------------------------------------------------------------------
# TestHausdorff
# ---------------------------------------------------------------------------


class TestHausdorff:
    """Tests for the hausdorff distance function."""

    def test_identical_clouds_zero(self):
        """Hausdorff distance of a cloud with itself is zero."""
        x = torch.randn(20, 3)
        assert torch.allclose(hausdorff(x, x), torch.zeros(()), atol=1e-5)

    def test_default_percentile_is_95(self):
        """Default call equals explicit percentile=95.0."""
        a = torch.randn(30, 3)
        b = torch.randn(30, 3)
        assert torch.allclose(hausdorff(a, b), hausdorff(a, b, percentile=95.0), atol=1e-6)

    def test_percentile_100_returns_max(self):
        """percentile=100 returns the worst-case (max) nearest-neighbour distance."""
        a = torch.tensor([[0., 0., 0.], [10., 0., 0.]])
        b = torch.tensor([[0., 0., 0.]])
        # per-row mins: [0., 10.], per-col min: [0.]; max after percentile=100 is 10.0
        assert torch.allclose(
            hausdorff(a, b, percentile=100.0), torch.tensor(10.0), atol=1e-4
        )

    def test_no_numpy_used(self):
        """Result is a torch.Tensor on CPU (not a NumPy scalar)."""
        a = torch.randn(15, 3)
        b = torch.randn(15, 3)
        result = hausdorff(a, b)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_invalid_shape_raises_value_error(self):
        """Non-(N,3) input raises ValueError."""
        x = torch.randn(10, 2)
        y = torch.randn(10, 3)
        with pytest.raises(ValueError, match="shape"):
            hausdorff(x, y)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_hausdorff_on_cuda(self):  # pragma: no cover
        """Result device matches CUDA inputs."""
        a = torch.randn(10, 3).cuda()
        b = torch.randn(10, 3).cuda()
        assert hausdorff(a, b).device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_hausdorff_device_mismatch_raises(self):  # pragma: no cover
        """CPU source vs CUDA target raises ValueError mentioning 'device'."""
        a = torch.randn(5, 3)
        b = torch.randn(5, 3).cuda()
        with pytest.raises(ValueError, match="device"):
            hausdorff(a, b)


# ---------------------------------------------------------------------------
# TestPathSmoothness
# ---------------------------------------------------------------------------


class TestPathSmoothness:
    """Tests for the path_smoothness function."""

    def test_empty_path_returns_zero(self):
        """Empty path returns 0.0."""
        assert path_smoothness([]) == 0.0

    def test_single_point_returns_zero(self):
        """Single-point path returns 0.0."""
        assert path_smoothness([(0, 0)]) == 0.0

    def test_two_points_returns_zero(self):
        """Two-point path returns 0.0 (need at least 3 points for a slope change)."""
        assert path_smoothness([(0, 0), (1, 1)]) == 0.0

    def test_diagonal_constant_slope_returns_zero(self):
        """Constant-slope diagonal path has zero variance of slope changes."""
        path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        assert abs(path_smoothness(path)) < 1e-9

    def test_bent_path_returns_positive(self):
        """Path with varying slopes yields positive smoothness value."""
        # slopes: ~1, ~2, ~4 — clearly not constant
        path = [(0, 0), (1, 1), (2, 3), (3, 7)]
        assert path_smoothness(path) > 0.0

    def test_returns_python_float(self):
        """Return type is a Python float."""
        result = path_smoothness([(0, 0), (1, 1), (2, 2)])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TestKnnConsistency
# ---------------------------------------------------------------------------


class TestKnnConsistency:
    """Tests for the knn_consistency function."""

    def test_all_same_label_returns_one(self):
        """When all points share a label, consistency is 1.0."""
        pts = torch.randn(50, 3)
        lbl = torch.zeros(50, dtype=torch.long)
        assert abs(knn_consistency(pts, lbl, k=5) - 1.0) < 1e-9

    def test_alternating_labels_low_score(self):
        """Alternating labels on a regular grid yield consistency < 0.5."""
        # Regularly spaced on a 1-D line; each point's k nearest neighbours
        # are its immediate neighbours which alternate labels 0/1.
        pts = torch.linspace(0, 49, 50).unsqueeze(1).expand(-1, 3).float().contiguous()
        lbl = (torch.arange(50) % 2).long()
        assert knn_consistency(pts, lbl, k=5) < 0.5

    def test_k_equals_n_minus_one_works(self):
        """k = N-1 (maximum allowed) succeeds without error."""
        pts = torch.randn(10, 3)
        lbl = torch.zeros(10, dtype=torch.long)
        result = knn_consistency(pts, lbl, k=9)
        assert 0.0 <= result <= 1.0

    def test_k_too_large_raises(self):
        """k = N (equal to cloud size) raises ValueError."""
        pts = torch.randn(10, 3)
        lbl = torch.zeros(10, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=10)

    def test_k_zero_or_negative_raises(self):
        """k=0 raises ValueError."""
        pts = torch.randn(10, 3)
        lbl = torch.zeros(10, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=0)

    def test_mismatched_lengths_raises(self):
        """Mismatched points/labels lengths raise ValueError."""
        pts = torch.randn(10, 3)
        lbl = torch.zeros(5, dtype=torch.long)
        with pytest.raises(ValueError):
            knn_consistency(pts, lbl, k=3)

    def test_returns_python_float(self):
        """Return type is a Python float."""
        pts = torch.randn(20, 3)
        lbl = torch.zeros(20, dtype=torch.long)
        assert isinstance(knn_consistency(pts, lbl, k=5), float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_knn_on_cuda(self):  # pragma: no cover
        """CUDA input tensors produce a valid float result (CPU KDTree entry point)."""
        pts = torch.randn(20, 3).cuda()
        lbl = torch.zeros(20, dtype=torch.long).cuda()
        result = knn_consistency(pts, lbl, k=5)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestTemporalStability
# ---------------------------------------------------------------------------


class TestTemporalStability:
    """Tests for the temporal_stability function."""

    def test_empty_list_returns_zero(self):
        """Empty transform list returns tensor(0.0)."""
        assert temporal_stability([]).item() == 0.0

    def test_single_element_returns_zero(self):
        """Single-element list returns tensor(0.0)."""
        tf = RigidTransformation()
        assert temporal_stability([tf]).item() == 0.0

    def test_two_identity_rigids_returns_zero(self):
        """Two identical identity transforms yield zero stability score."""
        tf = RigidTransformation()
        assert abs(temporal_stability([tf, tf]).item()) < 1e-9

    def test_unit_translation_returns_one(self):
        """Translation of magnitude 1 along x yields Frobenius norm of 1.0."""
        tf1 = RigidTransformation(t=torch.zeros(3))
        tf2 = RigidTransformation(t=torch.tensor([1., 0., 0.]))
        result = temporal_stability([tf1, tf2])
        assert abs(result.item() - 1.0) < 1e-5

    def test_mixed_rigid_and_affine(self):
        """Mixed Rigid + Affine list runs without error and returns finite value."""
        rigid = RigidTransformation()
        affine = AffineTransformation()
        result = temporal_stability([rigid, affine])
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result)

    def test_unsupported_type_raises_type_error(self):
        """Non-transform inputs raise TypeError mentioning 'Unsupported'."""
        with pytest.raises(TypeError, match="Unsupported"):
            temporal_stability([1, 2])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_temporal_on_cuda(self):  # pragma: no cover
        """Transforms on CUDA device produce a CUDA result tensor."""
        tf1 = RigidTransformation(device=torch.device("cuda"))
        tf2 = RigidTransformation(t=torch.tensor([1., 0., 0.], device="cuda"),
                                  device=torch.device("cuda"))
        result = temporal_stability([tf1, tf2])
        assert result.device.type == "cuda"


# ---------------------------------------------------------------------------
# TestComputeF1
# ---------------------------------------------------------------------------


class TestComputeF1:
    """Tests for the compute_f1 function."""

    def test_perfect_prediction_returns_one(self):
        """Perfect predictions yield F1 = 1.0."""
        y = torch.tensor([0, 1, 2, 0, 1])
        assert compute_f1(y, y) == 1.0

    def test_total_mismatch_returns_zero(self):
        """Completely mismatched binary predictions yield F1 = 0.0."""
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([1, 0, 1, 0])
        assert compute_f1(y_true, y_pred) == 0.0

    def test_default_average_is_weighted(self):
        """Default value of 'average' parameter is 'weighted' (EVAL-02 contract)."""
        assert inspect.signature(compute_f1).parameters["average"].default == "weighted"

    def test_default_zero_division_is_zero(self):
        """Default value of 'zero_division' parameter is 0."""
        assert inspect.signature(compute_f1).parameters["zero_division"].default == 0

    def test_sentinel_masking_skips_minus_one(self):
        """Entries with y_true == -1 are excluded; unmasked subset is perfect."""
        y_true = torch.tensor([-1, -1, 0, 1])
        y_pred = torch.tensor([0, 0, 0, 1])
        assert compute_f1(y_true, y_pred) == 1.0

    def test_all_masked_returns_zero(self):
        """All-sentinel y_true returns 0.0."""
        y_true = torch.tensor([-1, -1, -1])
        y_pred = torch.tensor([0, 0, 0])
        assert compute_f1(y_true, y_pred) == 0.0

    def test_mismatched_lengths_raises(self):
        """Different-length inputs raise ValueError mentioning 'length'."""
        with pytest.raises(ValueError, match="length"):
            compute_f1(torch.tensor([0, 1]), torch.tensor([0, 1, 2]))

    def test_non_1d_input_raises(self):
        """2-D y_true raises ValueError mentioning '1-D'."""
        y_true = torch.tensor([[0, 1], [0, 1]])
        y_pred = torch.tensor([0, 1, 0, 1])
        with pytest.raises(ValueError, match="1-D"):
            compute_f1(y_true, y_pred)

    def test_weighted_vs_macro_differ_on_imbalanced(self):
        """Weighted and macro averages differ on imbalanced class distributions."""
        # Heavy class imbalance: class 0 has 4 samples, class 1 has 1 sample
        y_t = torch.tensor([0, 0, 0, 0, 1])
        y_p = torch.tensor([0, 0, 0, 1, 0])  # one mistake per class
        weighted = compute_f1(y_t, y_p, average="weighted")
        macro = compute_f1(y_t, y_p, average="macro")
        assert weighted != macro

    def test_returns_python_float(self):
        """Return type is a Python float."""
        y = torch.tensor([0, 1])
        assert type(compute_f1(y, y)) is float

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compute_f1_on_cuda(self):  # pragma: no cover
        """CUDA input tensors work correctly (CPU sklearn entry point)."""
        y = torch.tensor([0, 1, 2, 0, 1]).cuda()
        assert compute_f1(y, y) == 1.0
