"""Tests for LabelGenerationConfig and EvalConfig.label_generation (D-13, Phase 56).

Covers the pydantic contract that DataFactory.generate_training_triple/
generate_training_set and HyperparamOptimizer's sanity tier consume: the
``LabelGenerationConfig`` model, the backward-compatible
``EvalConfig.label_generation`` field, and the end-to-end config-vs-explicit-
argument precedence resolved by ``DataFactory.generate_training_triple``
(Plan 56-03's sentinel fix).
"""

import pydantic
import pytest
import torch

from eval.config import EvalConfig, LabelGenerationConfig
from eval.data_factory import DataFactory
from zreg.data_generation import LabelComponentSpec, LabelSpec


def test_config_defaults():
    """LabelGenerationConfig(n_labels=4) applies the documented defaults."""
    c = LabelGenerationConfig(n_labels=4)
    assert c.mode == "deterministic"
    assert c.seed == 42


def test_config_label_specs_explicit():
    """LabelGenerationConfig accepts an explicit LabelSpec/LabelComponentSpec construction."""
    specs = [
        LabelSpec(
            label_id=0,
            components=[LabelComponentSpec(shape="blob", center=[0.0, 0.0, 0.0], sigma=1.0)],
        ),
        LabelSpec(
            label_id=1,
            components=[LabelComponentSpec(shape="voronoi", center=[10.0, 0.0, 0.0])],
        ),
    ]
    c = LabelGenerationConfig(label_specs=specs)
    assert c.label_specs == specs
    assert c.n_labels is None


def test_config_neither_n_labels_nor_label_specs_rejected():
    """LabelGenerationConfig() with neither n_labels nor label_specs raises ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        LabelGenerationConfig()


def test_config_both_n_labels_and_label_specs_rejected():
    """LabelGenerationConfig(n_labels=4, label_specs=[...]) with both set raises ValidationError."""
    specs = [
        LabelSpec(
            label_id=0,
            components=[LabelComponentSpec(shape="voronoi", center=[0.0, 0.0, 0.0])],
        )
    ]
    with pytest.raises(pydantic.ValidationError):
        LabelGenerationConfig(n_labels=4, label_specs=specs)


def test_config_extra_key_forbidden():
    """An unknown key is rejected (extra='forbid')."""
    with pytest.raises(pydantic.ValidationError):
        LabelGenerationConfig(n_labels=4, extra_key="x")


def test_config_in_all():
    """LabelGenerationConfig is part of the module public API."""
    import eval.config as config_mod

    assert "LabelGenerationConfig" in config_mod.__all__


def test_evalconfig_backward_compatible():
    """EvalConfig without label_generation defaults to None."""
    cfg = EvalConfig(data_path="x.mat")
    assert cfg.label_generation is None


def test_evalconfig_coerces_nested_dict():
    """A nested dict from YAML coerces into LabelGenerationConfig."""
    cfg = EvalConfig(data_path="x.mat", label_generation={"n_labels": 4})
    assert isinstance(cfg.label_generation, LabelGenerationConfig)
    assert cfg.label_generation.n_labels == 4


def test_end_to_end_precedence_config_applies_when_n_labels_omitted():
    """EvalConfig.label_generation applies when the caller does not pass an explicit n_labels.

    Note: seed=2 (not 0) is used because the random-Voronoi n_labels path can
    legitimately leave a label with zero assigned points for an arbitrary
    seed (some Voronoi cells may not be nearest for any sampled point) —
    seed=2 is the smallest seed verified to populate all n_labels/n_labels
    override cells for this ball-shape/point-count combination, so the
    "exactly N unique values" assertion below reliably proves the precedence
    invariant rather than an artifact of empty-cell luck.
    """
    cfg = EvalConfig(data_path="x.mat", label_generation={"n_labels": 3})
    factory = DataFactory(cfg)
    triple = factory.generate_training_triple(seed=2)
    unique_labels = torch.unique(triple.source_cloud["label"])
    assert unique_labels.numel() == 3


def test_end_to_end_precedence_explicit_n_labels_overrides_config():
    """An explicit caller n_labels argument overrides EvalConfig.label_generation.

    See test_end_to_end_precedence_config_applies_when_n_labels_omitted for
    why seed=2 is used instead of seed=0.
    """
    cfg = EvalConfig(data_path="x.mat", label_generation={"n_labels": 3})
    factory = DataFactory(cfg)
    triple = factory.generate_training_triple(seed=2, n_labels=5)
    unique_labels = torch.unique(triple.source_cloud["label"])
    assert unique_labels.numel() == 5


# ---------------------------------------------------------------------------
# Phase 62 DATA-03 (U4-7): degenerate label_generation values are rejected at
# config build, not at generate_labels() call time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_labels", [0, -1])
def test_config_n_labels_below_one_rejected(n_labels):
    """n_labels < 1 raises ValidationError naming the field."""
    with pytest.raises(pydantic.ValidationError, match="n_labels"):
        LabelGenerationConfig(n_labels=n_labels)


def test_config_n_labels_one_accepted():
    """n_labels=1 is the smallest accepted value."""
    assert LabelGenerationConfig(n_labels=1).n_labels == 1


def test_config_empty_label_specs_rejected():
    """label_specs=[] raises ValidationError naming the field."""
    with pytest.raises(pydantic.ValidationError, match="label_specs"):
        LabelGenerationConfig(label_specs=[])


def test_config_unknown_mode_rejected():
    """A misspelled mode keeps raising (pins the existing Literal)."""
    with pytest.raises(pydantic.ValidationError, match="mode"):
        LabelGenerationConfig(n_labels=2, mode="determinstic")


@pytest.mark.parametrize(
    "block", ["label_generation:\n  n_labels: 0\n", "label_generation:\n  label_specs: []\n"]
)
def test_yaml_degenerate_label_generation_raises_eval_config_error(tmp_path, block):
    """The same degenerate values through EvalConfig.from_yaml raise EvalConfigError."""
    from eval.config import EvalConfigError

    p = tmp_path / "cfg.yaml"
    p.write_text("data_path: data/raw/example.mat\n" + block)
    with pytest.raises(EvalConfigError):
        EvalConfig.from_yaml(p)
