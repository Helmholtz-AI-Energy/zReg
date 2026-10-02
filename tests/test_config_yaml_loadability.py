"""Regression tests: every experiment YAML in the repo loads through EvalConfig.

Phase 59 RUN-01: the HoreKa suite could not start because five
``configs_horeka/*/ew06_vs_shah.yaml`` files carried a ``label_source`` key that
``EvalConfig`` (``extra="forbid"``) no longer accepted, and
``hpo_paired/ew06_vs_shah.yaml`` silently carried a duplicate key.  These tests
glob every YAML under ``configs/`` and ``baseline_experiments/configs*/`` and
load each one through the canonical loader, so a config that drifts out of sync
with the schema is caught by CI instead of at cluster submission time.
"""

from pathlib import Path

import pytest
import yaml

from eval.config import EvalConfig, EvalConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent

_GLOBS = ("configs/**/*.yaml", "baseline_experiments/configs*/**/*.yaml")

YAMLS = sorted({p for pattern in _GLOBS for p in REPO_ROOT.glob(pattern)})

# YAML files under the globbed trees that are intentionally NOT EvalConfigs.
# Research verified every one of the 84 files is an EvalConfig, so this set is
# empty; any future non-EvalConfig YAML must be enumerated here explicitly.
NON_EVALCONFIG: frozenset[str] = frozenset()

_IDS = [str(p.relative_to(REPO_ROOT)) for p in YAMLS]


def test_glob_found_expected_count():
    """Guard against an empty or truncated glob silently passing the suite."""
    assert len(YAMLS) >= 84, f"expected >= 84 YAML configs, found {len(YAMLS)}"


@pytest.mark.parametrize("path", YAMLS, ids=_IDS)
def test_every_yaml_loads(path):
    """Every experiment YAML validates through EvalConfig.from_yaml."""
    if str(path.relative_to(REPO_ROOT)) in NON_EVALCONFIG:
        pytest.skip("enumerated non-EvalConfig YAML")
    cfg = EvalConfig.from_yaml(path)
    assert isinstance(cfg, EvalConfig)


@pytest.mark.parametrize("path", YAMLS, ids=_IDS)
def test_no_duplicate_keys(path):
    """No experiment YAML contains a duplicate mapping key."""
    from eval.config import _UniqueKeyLoader  # single source of truth for the strict loader

    with open(path) as f:
        yaml.load(f, Loader=_UniqueKeyLoader)


def test_from_yaml_rejects_duplicate_key(tmp_path):
    """A repeated top-level key is rejected instead of silently keeping the last value."""
    p = tmp_path / "dup.yaml"
    p.write_text("data_path: x.tracklets\ntier: dev\ntier: dev\n")
    with pytest.raises(EvalConfigError, match="duplicate"):
        EvalConfig.from_yaml(p)


def test_unique_key_loader_allows_merge_key_override():
    """An explicit key overriding a ``<<:`` merged key is not a duplicate."""
    from eval.config import _UniqueKeyLoader

    doc = "base: &b {k: 1}\nchild:\n  <<: *b\n  k: 2\n"
    assert yaml.load(doc, Loader=_UniqueKeyLoader) == {"base": {"k": 1}, "child": {"k": 2}}


def test_unique_key_loader_rejects_nested_duplicate_and_unhashable_key():
    """Nested duplicates and unhashable keys both surface as yaml.YAMLError."""
    from eval.config import _UniqueKeyLoader

    with pytest.raises(yaml.YAMLError, match="duplicate key 'k'"):
        yaml.load("a:\n  k: 1\n  k: 2\n", Loader=_UniqueKeyLoader)
    with pytest.raises(yaml.YAMLError, match="unhashable"):
        yaml.load("? [1, 2]\n: v\n", Loader=_UniqueKeyLoader)


# Census of Kobitski (.tracklets) -> Shah (.csv) paired label-transfer configs
# (Phase 59 NUM-04).  Pinned so a new config in the wrong direction, or a broken
# glob, is noticed.
KOBITSKI_TO_SHAH_LABEL_CONFIG_COUNT = 27


def test_kobitski_to_shah_label_configs_use_target_direction():
    """Every Kobitski->Shah paired label-transfer config takes labels from Shah.

    Shah (target) carries the 3-class germ-layer labels; Kobitski (source) is
    unlabeled.  The default ``label_source='source'`` would transfer Kobitski's
    arbitrary colour indices onto Shah, so every such config must set
    ``label_source: "target"``.  No allow-list: every matching config counts.
    """
    matched = []
    wrong = []
    for path in YAMLS:
        cfg = EvalConfig.from_yaml(path)
        if not (
            cfg.pipeline_mode == "paired"
            and cfg.run_label_transfer
            and cfg.data_path.endswith(".tracklets")
            and (cfg.target_data_path or "").endswith(".csv")
        ):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        matched.append(rel)
        if cfg.label_source != "target":
            wrong.append(rel)
    assert not wrong, (
        'Kobitski->Shah label-transfer configs must set label_source: "target"; '
        f"wrong direction in {len(wrong)} file(s):\n" + "\n".join(wrong)
    )
    assert len(matched) == KOBITSKI_TO_SHAH_LABEL_CONFIG_COUNT, (
        f"expected {KOBITSKI_TO_SHAH_LABEL_CONFIG_COUNT} Kobitski->Shah label-transfer configs, "
        f"found {len(matched)}:\n" + "\n".join(matched)
    )


def test_cpd_weighted_real_config_uses_target_direction():
    """Explicit pin for the cpd_weighted real-data config flagged in review (HIGH 1)."""
    cfg = EvalConfig.from_yaml(REPO_ROOT / "configs/experiments/stage2_label_transfer/cpd_weighted/real.yaml")
    assert cfg.label_transfer_method == "cpd_weighted"
    assert cfg.label_source == "target"
