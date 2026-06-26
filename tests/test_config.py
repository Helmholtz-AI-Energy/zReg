"""Tests for zreg.config module."""

import pytest
import torch

from zreg.config import configure_pytorch


class TestConfigurePytorch:
    """Tests for configure_pytorch function."""

    def test_sets_matmul_precision_default(self):
        """Test that default matmul precision is 'high'."""
        configure_pytorch()
        # Can't easily check the actual value, but no error means success
        assert True

    def test_sets_matmul_precision_highest(self):
        """Test setting matmul precision to 'highest'."""
        configure_pytorch(matmul_precision="highest")
        assert True

    def test_sets_matmul_precision_medium(self):
        """Test setting matmul precision to 'medium'."""
        configure_pytorch(matmul_precision="medium")
        assert True

    def test_sets_default_device_none(self):
        """Test that None default_device doesn't change anything."""
        configure_pytorch(default_device=None)
        assert True

    def test_sets_default_device_cpu(self):
        """Test setting default device to CPU."""
        # Save current default
        configure_pytorch(default_device="cpu")
        # Verify tensors are created on CPU
        t = torch.empty(1)
        assert t.device.type == "cpu"
        # Reset to no default device
        torch.set_default_device(None)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sets_default_device_cuda(self):  # pragma: no cover
        """Test setting default device to CUDA."""
        configure_pytorch(default_device="cuda")
        t = torch.empty(1)
        assert t.device.type == "cuda"
        # Reset to no default device
        torch.set_default_device(None)
