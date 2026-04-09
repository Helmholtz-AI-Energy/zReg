import os
import subprocess
import sys
import logging

import pytest


class TestLogging:
    """Tests for logging configuration (QUALITY-03)."""

    def test_set_log_level_string(self):
        import zreg
        zreg.set_log_level("WARNING")
        assert logging.getLogger("zreg").level == logging.WARNING
        # Reset to INFO for other tests
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
