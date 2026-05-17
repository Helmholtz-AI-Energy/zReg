"""Tests for zreg.metrics.label_transfer module (compute_f1)."""

import inspect

import pytest
import torch

from zreg.metrics.label_transfer import compute_f1


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
        from zreg.metrics import compute_f1 as cf  # noqa: F401
        assert cf is compute_f1
