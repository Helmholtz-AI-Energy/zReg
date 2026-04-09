"""Tests for zreg.validation module and logging configuration."""

import os
import subprocess
import sys
import logging
import pytest
import torch
from unittest.mock import MagicMock

from zreg.validation import _validate_tensors
from zreg.distances.general import minkowski_distance


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
        cpu_tensor = torch.randn(5, 3)

        # Create a mock tensor that reports being on cuda:0
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.device = torch.device("cuda:0")
        mock_tensor.__class__ = torch.Tensor

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
        assert "2" in msg or "1" in msg

    def test_check_finite_false_skips(self):
        """Test that check_finite=False skips NaN validation."""
        nan_tensor = torch.tensor([float("nan"), 2.0, 3.0])
        _validate_tensors(nan_tensor, names=["x"], check_finite=False)  # should not raise

    def test_check_device_false_skips(self):
        """Test that check_device=False skips device mismatch validation."""
        cpu_tensor = torch.randn(5, 3)
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.device = torch.device("cuda:0")
        _validate_tensors(cpu_tensor, mock_tensor, names=["source", "target"], check_finite=False, check_device=False)  # should not raise

    def test_single_tensor_no_device_check(self):
        """Test that a single tensor never triggers device check even with check_device=True."""
        t = torch.randn(5, 3)
        _validate_tensors(t, names=["source"], check_device=True)  # should not raise


class TestEntryPointValidation:
    """Integration tests confirming NaN/inf rejection at actual entry points."""

    def test_cpd_set_source_nan_raises(self):
        """Test that CPD.set_source rejects a NaN tensor with 'source' in message."""
        from zreg.cpd import RigidCPD

        valid_source = torch.randn(10, 3)
        cpd = RigidCPD(source=valid_source)
        nan_tensor = torch.full((10, 3), float("nan"))
        with pytest.raises(ValueError) as exc_info:
            cpd.set_source(nan_tensor)
        assert "source" in str(exc_info.value)

    def test_cpd_registration_nan_target_raises(self):
        """Test that CPD.registration rejects a NaN target tensor with 'target' in message."""
        from zreg.cpd import RigidCPD

        source = torch.randn(10, 3)
        cpd = RigidCPD(source=source)
        nan_target = torch.full((10, 3), float("nan"))
        with pytest.raises(ValueError) as exc_info:
            cpd.registration(nan_target)
        assert "target" in str(exc_info.value)

    def test_base_wd_forward_nan_raises(self):
        """Test that BaseWD.forward rejects a NaN x tensor with 'x' in message."""
        from zreg.distances.sw_varients import SlicedWassersteinDistance

        swd = SlicedWassersteinDistance(num_projs=50, nobatchdim=True, device="cpu")
        nan_tensor = torch.full((10, 3), float("nan"))
        valid_tensor = torch.randn(10, 3)
        with pytest.raises(ValueError) as exc_info:
            swd.forward(nan_tensor, valid_tensor)
        assert "x" in str(exc_info.value)

    def test_minkowski_nan_raises(self):
        """Test that minkowski_distance rejects a NaN x tensor with 'x' in message."""
        nan_tensor = torch.full((10, 3), float("nan"))
        valid_tensor = torch.randn(10, 3)
        with pytest.raises(ValueError) as exc_info:
            minkowski_distance(nan_tensor, valid_tensor)
        assert "x" in str(exc_info.value)


class TestLogging:
    """Tests for logging configuration (QUALITY-03)."""

    def test_set_log_level_string(self):
        import zreg
        zreg.set_log_level("WARNING")
        assert logging.getLogger("zreg").level == logging.WARNING
        zreg.set_log_level("INFO")

    def test_set_log_level_int(self):
        import zreg
        zreg.set_log_level(logging.WARNING)
        assert logging.getLogger("zreg").level == logging.WARNING
        zreg.set_log_level(logging.INFO)

    def test_set_log_level_case_insensitive(self):
        import zreg
        zreg.set_log_level("warning")
        assert logging.getLogger("zreg").level == logging.WARNING
        zreg.set_log_level("INFO")

    def test_env_var_suppresses_info(self):
        """ZREG_LOG_LEVEL=WARNING must suppress INFO output."""
        script = (
            "import logging; import zreg; "
            "logging.getLogger('zreg').info('should_not_appear'); "
            "print('DONE')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            env={**os.environ, "ZREG_LOG_LEVEL": "WARNING"},
        )
        assert "should_not_appear" not in result.stdout
        assert "should_not_appear" not in result.stderr
        assert "DONE" in result.stdout

    def test_default_level_is_info(self):
        """Default log level should be INFO when env var not set."""
        script = (
            "import logging; import zreg; "
            "print(logging.getLogger('zreg').level)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if k != "ZREG_LOG_LEVEL"},
        )
        assert result.stdout.strip() == str(logging.INFO)
