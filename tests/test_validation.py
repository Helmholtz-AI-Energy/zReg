"""Tests for zreg.validation module."""

import pytest
import torch
from unittest.mock import PropertyMock, patch

from zreg.validation import _validate_tensors


class TestValidateTensors:
    """Unit tests for the _validate_tensors function."""

    def test_nan_raises(self):
        """Test that a tensor containing NaN raises ValueError naming the parameter."""
        t = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError) as exc_info:
            _validate_tensors(t, names=["source"])
        msg = str(exc_info.value)
        assert "source" in msg
        assert "NaN" in msg

    def test_inf_raises(self):
        """Test that a tensor containing inf raises ValueError naming the parameter."""
        t = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError) as exc_info:
            _validate_tensors(t, names=["source"])
        msg = str(exc_info.value)
        assert "source" in msg
        assert "inf" in msg

    def test_finite_ok(self):
        """Test that a valid finite tensor does not raise."""
        t = torch.randn(10, 3)
        _validate_tensors(t, names=["source"])  # should not raise

    def test_device_mismatch_raises(self):
        """Test that tensors on different devices raise ValueError containing both device names."""
        from unittest.mock import MagicMock

        cpu_tensor = torch.randn(5, 3)

        # Create a mock tensor that reports being on cuda:0
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.device = torch.device("cuda:0")
        # Make isfinite work on it (returns all-true tensor)
        torch.isfinite = torch.isfinite  # ensure original still works
        mock_tensor.__class__ = torch.Tensor  # allow isinstance checks

        # Use a real CPU tensor and a mock tensor with cuda:0 device
        with pytest.raises(ValueError) as exc_info:
            _validate_tensors(cpu_tensor, mock_tensor, names=["source", "target"], check_finite=False)
        msg = str(exc_info.value)
        assert "cpu" in msg
        assert "cuda:0" in msg

    def test_same_device_ok(self):
        """Test that tensors on the same device do not raise."""
        t1 = torch.randn(5, 3)
        t2 = torch.randn(5, 3)
        _validate_tensors(t1, t2, names=["source", "target"])  # should not raise

    def test_names_length_mismatch(self):
        """Test that mismatched number of tensors and names raises ValueError."""
        t1 = torch.randn(5, 3)
        t2 = torch.randn(5, 3)
        with pytest.raises(ValueError) as exc_info:
            _validate_tensors(t1, t2, names=["only_one"])
        msg = str(exc_info.value)
        # Should mention something about the mismatch
        assert "2" in msg or "1" in msg

    def test_check_finite_false_skips(self):
        """Test that check_finite=False skips NaN validation."""
        nan_tensor = torch.tensor([float("nan"), 2.0, 3.0])
        _validate_tensors(nan_tensor, names=["x"], check_finite=False)  # should not raise

    def test_check_device_false_skips(self):
        """Test that check_device=False skips device mismatch validation."""
        from unittest.mock import MagicMock

        cpu_tensor = torch.randn(5, 3)
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.device = torch.device("cuda:0")
        _validate_tensors(cpu_tensor, mock_tensor, names=["source", "target"], check_finite=False, check_device=False)  # should not raise

    def test_single_tensor_no_device_check(self):
        """Test that a single tensor never triggers device check even with check_device=True."""
        t = torch.randn(5, 3)
        _validate_tensors(t, names=["source"], check_device=True)  # should not raise
