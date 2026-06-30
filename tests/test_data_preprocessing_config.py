"""Tests for DataPreprocessingConfig Pydantic model (Phase 43).

Covers the Pydantic contracts for DataPreprocessingConfig and its integration
with EvalConfig.data_preprocessing field.
"""

import pydantic
import pytest
import yaml

import eval.config as config_mod
from eval.config import DataPreprocessingConfig, EvalConfig


def test_default_method_is_standardize():
    """DataPreprocessingConfig() with no args defaults to standardize and 3.0 threshold."""
    c = DataPreprocessingConfig()
    assert c.method == "standardize"
    assert c.robust_outlier_threshold == 3.0


def test_normalize_method_valid():
    """DataPreprocessingConfig(method='normalize') is valid."""
    c = DataPreprocessingConfig(method="normalize")
    assert c.method == "normalize"


def test_robust_method_valid():
    """DataPreprocessingConfig(method='robust') is valid with default threshold."""
    c = DataPreprocessingConfig(method="robust")
    assert c.method == "robust"
    assert c.robust_outlier_threshold == 3.0


def test_custom_threshold_valid():
    """DataPreprocessingConfig(method='robust', robust_outlier_threshold=1.5) is valid."""
    c = DataPreprocessingConfig(method="robust", robust_outlier_threshold=1.5)
    assert c.robust_outlier_threshold == 1.5


def test_invalid_method_rejected():
    """An unknown method value is rejected at parse time."""
    with pytest.raises(pydantic.ValidationError):
        DataPreprocessingConfig(method="whiten")


def test_extra_key_forbidden():
    """Unknown keys are rejected (extra='forbid')."""
    with pytest.raises(pydantic.ValidationError):
        DataPreprocessingConfig(method="standardize", unknown="x")


def test_in_all():
    """DataPreprocessingConfig is part of the module public API."""
    assert "DataPreprocessingConfig" in config_mod.__all__


def test_evalconfig_default_preprocessing_on():
    """EvalConfig.data_preprocessing defaults to standardize (D-01)."""
    cfg = EvalConfig(data_path="x")
    assert cfg.data_preprocessing is not None
    assert cfg.data_preprocessing.method == "standardize"


def test_evalconfig_opt_out_with_null():
    """EvalConfig(data_preprocessing=None) disables preprocessing entirely (D-02)."""
    cfg = EvalConfig(data_path="x", data_preprocessing=None)
    assert cfg.data_preprocessing is None


def test_evalconfig_yaml_without_field_gets_default(tmp_path):
    """A YAML without data_preprocessing field inherits standardize default (D-01)."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml.dump({"data_path": "x.mat"}))
    cfg = EvalConfig.from_yaml(str(yaml_file))
    assert cfg.data_preprocessing is not None
    assert cfg.data_preprocessing.method == "standardize"
