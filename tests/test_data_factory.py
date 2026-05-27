"""Tests for eval.config.EvalConfig and eval.data_factory.DataFactory.

Seven test classes cover the two modules in phase 17:

- TestEvalConfigFromYAML       — FRAME-01 gate: from_yaml + EvalConfigError wrapping
                                 (this plan, 17-01; 6 tests fully implemented)
- TestDataFactoryConstruction  — D-08 lazy init (filled in by Plan 17-02)
- TestLoadReal                 — D-09 caching + dispatch by data_format (Plan 17-02)
- TestGenerateSynthetic        — caching + seed reproducibility (Plan 17-02)
- TestAugment                  — augmentation_params dispatch, 4 cases (Plan 17-02)
- TestPrepareSplit             — FRAME-02 split-shape gate (Plan 17-02)
- TestGetGroundTruth           — D-10/D-11 id extraction (Plan 17-02)

Tests use tmp_path for YAML files and unittest.mock.patch for loader dispatch
so neither data/raw/ nor experiments/runs/ is touched during test runs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pydantic
import pytest
import yaml

from eval.config import EvalConfig, EvalConfigError

# Plan 17-02 will add: import DataFactory from eval.data_factory

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "data/raw/example.mat"
DEFAULT_FORMAT = "tracklets"
DEFAULT_VAL_SPLIT = 0.2


# ---------------------------------------------------------------------------
# TestEvalConfigFromYAML — FRAME-01 gate (Plan 17-01)
# ---------------------------------------------------------------------------


class TestEvalConfigFromYAML:
    """FRAME-01 gate: EvalConfig.from_yaml and EvalConfigError wrapping."""

    def test_loads_minimal_yaml(self, tmp_path):
        """from_yaml with only data_path returns defaults for all other fields."""
        p = tmp_path / "cfg.yaml"
        p.write_text("data_path: data/raw/example.mat\n")
        cfg = EvalConfig.from_yaml(p)
        assert cfg.data_path == "data/raw/example.mat"
        assert cfg.val_split == DEFAULT_VAL_SPLIT
        assert cfg.data_format == "tracklets"
        assert cfg.n_synthetic == 100
        assert cfg.augmentation_params == {}
        assert cfg.run_alignment is True

    def test_missing_data_path_raises_EvalConfigError(self, tmp_path):
        """YAML without data_path raises EvalConfigError (not pydantic.ValidationError)."""
        p = tmp_path / "no_path.yaml"
        p.write_text("# empty config\n")
        with pytest.raises(EvalConfigError, match="data_path") as excinfo:
            EvalConfig.from_yaml(p)
        assert not isinstance(excinfo.value, pydantic.ValidationError)

    def test_unknown_field_raises_EvalConfigError(self, tmp_path):
        """YAML with an unknown key raises EvalConfigError mentioning the key."""
        p = tmp_path / "bad.yaml"
        p.write_text("data_path: x\nbogus_field: 7\n")
        with pytest.raises(EvalConfigError, match="bogus_field") as excinfo:
            EvalConfig.from_yaml(p)
        assert not isinstance(excinfo.value, pydantic.ValidationError)

    def test_bad_type_wrapped(self, tmp_path):
        """Type-coercion failure is wrapped into EvalConfigError (Pitfall 2)."""
        p = tmp_path / "bad_type.yaml"
        p.write_text("data_path: x\nval_split: not_a_float\n")
        with pytest.raises(EvalConfigError, match="val_split") as excinfo:
            EvalConfig.from_yaml(p)
        assert not isinstance(excinfo.value, pydantic.ValidationError)

    def test_empty_yaml_handled(self, tmp_path):
        """Empty YAML file (yaml.safe_load returns None) raises EvalConfigError, not TypeError."""
        p = tmp_path / "empty.yaml"
        p.write_text("")
        # yaml.safe_load("") returns None; without `or {}` guard this raises TypeError
        with pytest.raises(EvalConfigError, match="data_path"):
            EvalConfig.from_yaml(p)

    def test_missing_file(self, tmp_path):
        """Non-existent path raises EvalConfigError with 'not found' in the message."""
        p = tmp_path / "nonexistent.yaml"
        with pytest.raises(EvalConfigError, match="not found"):
            EvalConfig.from_yaml(p)


# ---------------------------------------------------------------------------
# Stub classes — Plan 17-02 will populate these
# ---------------------------------------------------------------------------


class TestDataFactoryConstruction:
    """D-08 lazy init — populated in Plan 17-02."""

    pass


class TestLoadReal:
    """D-09 caching + dispatch by data_format — populated in Plan 17-02."""

    pass


class TestGenerateSynthetic:
    """Caching + seed reproducibility — populated in Plan 17-02."""

    pass


class TestAugment:
    """augmentation_params dispatch, 4 cases — populated in Plan 17-02."""

    pass


class TestPrepareSplit:
    """FRAME-02 split-shape gate — populated in Plan 17-02."""

    pass


class TestGetGroundTruth:
    """D-10/D-11 id extraction — populated in Plan 17-02."""

    pass
