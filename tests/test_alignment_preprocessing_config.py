"""Tests for AlignmentPreprocessingConfig and AlignResult.velocity_landmarks (Phase 41).

Covers the Pydantic contracts that AlignmentStage wiring (plan 41-02) consumes:
the ``AlignmentPreprocessingConfig`` model, the backward-compatible
``EvalConfig.alignment_preprocessing`` field, and the
``AlignResult.velocity_landmarks`` field.
"""

import pydantic
import pytest

from eval.config import AlignmentPreprocessingConfig, EvalConfig
from eval.types import AlignResult


def test_config_defaults():
    """principal_axes config applies the documented defaults."""
    c = AlignmentPreprocessingConfig(method="principal_axes")
    assert c.velocity_threshold == 0.5
    assert c.velocity_metric == "mean"


def test_config_velocity_landmarks_explicit():
    """velocity_landmarks config accepts explicit threshold and metric."""
    c = AlignmentPreprocessingConfig(
        method="velocity_landmarks", velocity_threshold=0.3, velocity_metric="max"
    )
    assert c.method == "velocity_landmarks"
    assert c.velocity_threshold == 0.3
    assert c.velocity_metric == "max"


def test_config_invalid_method_rejected():
    """An unknown method value is rejected at parse time."""
    with pytest.raises(pydantic.ValidationError):
        AlignmentPreprocessingConfig(method="invalid")


def test_config_extra_key_forbidden():
    """Unknown keys are rejected (extra='forbid')."""
    with pytest.raises(pydantic.ValidationError):
        AlignmentPreprocessingConfig(method="principal_axes", extra_key="x")


def test_config_invalid_metric_rejected():
    """An unknown velocity_metric value is rejected."""
    with pytest.raises(pydantic.ValidationError):
        AlignmentPreprocessingConfig(method="principal_axes", velocity_metric="median")


def test_config_in_all():
    """AlignmentPreprocessingConfig is part of the module public API."""
    import eval.config as config_mod

    assert "AlignmentPreprocessingConfig" in config_mod.__all__


def test_evalconfig_backward_compatible():
    """EvalConfig without alignment_preprocessing defaults to None."""
    cfg = EvalConfig(data_path="x.mat")
    assert cfg.alignment_preprocessing is None


def test_evalconfig_coerces_nested_dict():
    """A nested dict from YAML coerces into AlignmentPreprocessingConfig."""
    cfg = EvalConfig(
        data_path="x.mat",
        alignment_preprocessing={"method": "principal_axes"},
    )
    assert isinstance(cfg.alignment_preprocessing, AlignmentPreprocessingConfig)
    assert cfg.alignment_preprocessing.method == "principal_axes"


def test_alignresult_velocity_landmarks_default_empty():
    """AlignResult constructed without velocity_landmarks defaults to []."""
    r = AlignResult(
        aligned_cloud={},
        warp_path=[],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={},
    )
    assert r.velocity_landmarks == []


def test_alignresult_velocity_landmarks_is_frozen():
    """AlignResult is frozen — reassigning velocity_landmarks raises."""
    r = AlignResult(
        aligned_cloud={},
        warp_path=[],
        dtw_distance=0.0,
        n_changepoints=0,
        params_used={},
    )
    with pytest.raises(pydantic.ValidationError):
        r.velocity_landmarks = [1]
