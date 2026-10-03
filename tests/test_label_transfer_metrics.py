"""Tests for zreg.metrics.label_transfer module (compute_f1)."""

import inspect

import pytest
import torch

from zreg.evaluation.label_transfer import compute_f1, knn_consistency


class TestComputeF1Signature:
    """Tests for compute_f1 function signature defaults."""

    def test_average_default_is_weighted(self):
        """Default value of `average` parameter must be 'weighted' (not 'macro')."""
        sig = inspect.signature(compute_f1)
        assert sig.parameters["average"].default == "weighted"

    def test_zero_division_default_is_zero(self):
        """Default value of `zero_division` parameter must be 0."""
        sig = inspect.signature(compute_f1)
        assert sig.parameters["zero_division"].default == 0


class TestComputeF1PerfectPrediction:
    """Tests for compute_f1 with perfect predictions."""

    def test_perfect_multiclass_prediction_returns_one(self):
        """Perfect prediction on multiclass labels returns 1.0."""
        y = torch.tensor([0, 1, 2, 0, 1])
        result = compute_f1(y, y)
        assert result == 1.0

    def test_perfect_binary_prediction_returns_one(self):
        """Perfect binary prediction returns 1.0."""
        y = torch.tensor([0, 1, 0, 1, 0])
        result = compute_f1(y, y)
        assert result == 1.0


class TestComputeF1WrongPrediction:
    """Tests for compute_f1 with incorrect predictions."""

    def test_all_wrong_binary_returns_zero(self):
        """All predictions wrong on balanced binary labels returns 0.0."""
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([1, 0, 1, 0])
        result = compute_f1(y_true, y_pred)
        assert result == 0.0


class TestComputeF1SentinelMasking:
    """Tests for compute_f1 sentinel masking (y_true == -1 entries dropped)."""

    def test_masked_entries_ignored_perfect_on_unmasked(self):
        """Entries with y_true == -1 are ignored; perfect on unmasked subset returns 1.0."""
        y_true = torch.tensor([-1, -1, 0, 1])
        y_pred = torch.tensor([0, 0, 0, 1])
        result = compute_f1(y_true, y_pred)
        assert result == 1.0

    def test_all_masked_returns_zero(self):
        """All entries masked out (y_true all -1) returns 0.0, no exception."""
        y_true = torch.tensor([-1, -1, -1])
        y_pred = torch.tensor([0, 0, 0])
        result = compute_f1(y_true, y_pred)
        assert result == 0.0

    def test_single_valid_entry_does_not_crash(self):
        """Single unmasked entry does not crash."""
        y_true = torch.tensor([-1, -1, 0])
        y_pred = torch.tensor([0, 0, 0])
        result = compute_f1(y_true, y_pred)
        assert isinstance(result, float)


class TestComputeF1ReturnType:
    """Tests for compute_f1 return type."""

    def test_returns_python_float(self):
        """Return type must be Python float, not numpy.float64 or torch.Tensor."""
        y = torch.tensor([0, 1, 2])
        result = compute_f1(y, y)
        assert type(result) is float

    def test_perfect_prediction_type_is_float(self):
        """Return type is float even for perfect prediction."""
        y = torch.tensor([0, 1, 2, 0, 1])
        result = compute_f1(y, y)
        assert type(result) is float


class TestComputeF1AverageParameter:
    """Tests for compute_f1 `average` parameter."""

    def test_macro_average_accepted(self):
        """average='macro' is accepted without error."""
        y_true = torch.tensor([0, 0, 1, 1, 1])
        y_pred = torch.tensor([0, 1, 0, 1, 1])
        result = compute_f1(y_true, y_pred, average="macro")
        assert isinstance(result, float)

    def test_weighted_and_macro_differ_on_imbalanced_data(self):
        """average='weighted' and average='macro' differ on imbalanced class distributions."""
        # Class 0: 1 sample; class 1: 4 samples — imbalanced
        y_true = torch.tensor([0, 1, 1, 1, 1])
        y_pred = torch.tensor([1, 0, 1, 1, 1])
        result_weighted = compute_f1(y_true, y_pred, average="weighted")
        result_macro = compute_f1(y_true, y_pred, average="macro")
        assert result_weighted != result_macro

    def test_invalid_average_raises_value_error(self):
        """Passing an invalid average value raises ValueError from sklearn."""
        y = torch.tensor([0, 1, 2])
        with pytest.raises(ValueError):
            compute_f1(y, y, average="banana")


class TestComputeF1ShapeValidation:
    """Tests for compute_f1 shape guards."""

    def test_2d_y_true_raises_value_error(self):
        """2-D y_true raises ValueError (must be 1-D)."""
        y_true = torch.tensor([[0, 1], [1, 0]])
        y_pred = torch.tensor([0, 1, 0, 1])
        with pytest.raises(ValueError):
            compute_f1(y_true, y_pred)

    def test_2d_y_pred_raises_value_error(self):
        """2-D y_pred raises ValueError (must be 1-D)."""
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([[0, 1], [1, 0]])
        with pytest.raises(ValueError):
            compute_f1(y_true, y_pred)

    def test_length_mismatch_raises_value_error(self):
        """y_true and y_pred with different lengths raises ValueError."""
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1])
        with pytest.raises(ValueError):
            compute_f1(y_true, y_pred)


class TestComputeF1PackageImport:
    """Tests for compute_f1 package-level re-export."""

    def test_importable_from_package(self):
        """compute_f1 is importable from zreg.metrics package."""
        from zreg.evaluation import compute_f1 as cf  # noqa: F401
        assert cf is compute_f1


class TestToMatrixUnsupportedType:
    """label_transfer.py:127 — _to_matrix raises TypeError for unknown transform type."""

    def test_unsupported_type_raises_type_error(self):
        """_to_matrix(obj) where obj is not Rigid/Affine raises TypeError."""
        from zreg.evaluation.label_transfer import _to_matrix
        with pytest.raises(TypeError, match="Unsupported"):
            _to_matrix("not_a_transform")


class TestKnnConsistencySelfExclusion:
    """knn_consistency excludes the query point by index (LT-04 U1-8)."""

    def test_coincident_pairs_with_different_labels(self):
        """Coincident points with different labels are each other's neighbour."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 5.1],
                [9.0, 9.0, 9.0],
                [9.0, 9.0, 9.1],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 1, 2, 2, 3, 3])
        # Points 0/1 see each other (different labels -> 0); the other four
        # points see their partner with the same label -> 1. Mean = 4/6.
        assert knn_consistency(points, labels, k=1) == pytest.approx(4 / 6, abs=1e-12)

    def test_more_than_k_plus_one_coincident_points_use_exactly_k(self):
        """When self is not among the k+1 returned neighbours, k neighbours are still used."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [9.0, 9.0, 9.0],
                [9.0, 9.0, 9.1],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 1, 2, 3, 4, 4])
        # Each origin point: 2 other origin points, all labels distinct -> 0.
        # Each far point: its partner (match) and one origin point (no match)
        # -> 1/2. Mean = (0 * 4 + 0.5 + 0.5) / 6.
        expected = (0.0 * 4 + 0.5 + 0.5) / 6
        assert knn_consistency(points, labels, k=2) == pytest.approx(expected, abs=1e-12)

    def test_well_separated_points_unchanged(self):
        """Without ties the result equals the definition."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 12.0, 0.0]]
        )
        labels = torch.tensor([0, 0, 1, 2])
        # 0<->1 match, 2<->3 do not match -> 2/4.
        result = knn_consistency(points, labels, k=1)
        assert isinstance(result, float)
        assert result == pytest.approx(0.5, abs=1e-12)


class TestTemporalStabilityMixedInputs:
    """temporal_stability raises TypeError for mixed device/dtype lists (LT-04 U1-9)."""

    def test_mixed_device_raises_type_error(self):
        from zreg.core.transforms import RigidTransformation
        from zreg.evaluation.label_transfer import temporal_stability

        transforms = [
            RigidTransformation(),
            RigidTransformation(device=torch.device("meta")),
        ]
        with pytest.raises(TypeError, match="device"):
            temporal_stability(transforms)

    def test_mixed_dtype_raises_type_error(self):
        from zreg.core.transforms import RigidTransformation
        from zreg.evaluation.label_transfer import temporal_stability

        transforms = [
            RigidTransformation(dtype=torch.float32),
            RigidTransformation(dtype=torch.float64),
        ]
        with pytest.raises(TypeError, match="dtype"):
            temporal_stability(transforms)

    def test_homogeneous_rigid_and_affine_unchanged(self):
        from zreg.core.transforms import AffineTransformation, RigidTransformation
        from zreg.evaluation.label_transfer import temporal_stability

        rigid = RigidTransformation(dtype=torch.float32)
        affine = AffineTransformation(
            b=2.0 * torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
        )
        result = temporal_stability([rigid, affine])
        # Difference of the 4x4 matrices is diag(1, 1, 1, 0) -> Frobenius sqrt(3).
        assert result.dtype == torch.float32
        assert result.item() == pytest.approx(3.0 ** 0.5, abs=1e-6)

    def test_single_and_empty_lists_unchanged(self):
        from zreg.core.transforms import RigidTransformation
        from zreg.evaluation.label_transfer import temporal_stability

        assert temporal_stability([]).item() == 0.0
        single = temporal_stability([RigidTransformation(dtype=torch.float64)])
        assert single.item() == 0.0
        assert single.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_and_cuda_mix_raises_type_error(self):
        from zreg.core.transforms import RigidTransformation
        from zreg.evaluation.label_transfer import temporal_stability

        transforms = [
            RigidTransformation(),
            RigidTransformation(device=torch.device("cuda")),
        ]
        with pytest.raises(TypeError, match="device"):
            temporal_stability(transforms)
