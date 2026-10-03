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

# zreg.dataset MUST be imported before torch on macOS ARM to avoid libomp SIGABRT
from zreg.core.dataset import zRegPointCloud

import torch

from eval.data_factory import DataFactory

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

    def test_pipeline_mode_literal_accepts_paired_and_synthetic(self, tmp_path):
        """pipeline_mode='paired' and 'synthetic' both round-trip without error."""
        p = tmp_path / "paired.yaml"
        p.write_text("data_path: x.mat\npipeline_mode: paired\n")
        cfg = EvalConfig.from_yaml(p)
        assert cfg.pipeline_mode == "paired"

        p2 = tmp_path / "synthetic.yaml"
        p2.write_text("data_path: x.mat\npipeline_mode: synthetic\n")
        cfg2 = EvalConfig.from_yaml(p2)
        assert cfg2.pipeline_mode == "synthetic"

    def test_pipeline_mode_rejects_invalid_string(self, tmp_path):
        """pipeline_mode: Paired (capital P) is rejected with EvalConfigError (Pitfall 2)."""
        p = tmp_path / "bad_mode.yaml"
        p.write_text("data_path: x.mat\npipeline_mode: Paired\n")
        with pytest.raises(EvalConfigError, match="pipeline_mode"):
            EvalConfig.from_yaml(p)

    def test_ground_truth_field_defaults_to_label(self):
        """Phase 56 D-01: ground_truth_field defaults to 'label' on direct construction."""
        cfg = EvalConfig(data_path="x")
        assert cfg.ground_truth_field == "label"

    def test_ground_truth_field_accepts_id_override(self):
        """Phase 56 D-02: ground_truth_field='id' is a valid explicit override."""
        cfg = EvalConfig(data_path="x", ground_truth_field="id")
        assert cfg.ground_truth_field == "id"

    def test_ground_truth_field_rejects_invalid_string(self):
        """Phase 56 D-02: ground_truth_field is a closed Literal — no free-form values."""
        with pytest.raises(pydantic.ValidationError):
            EvalConfig(data_path="x", ground_truth_field="bogus")

    def test_target_data_path_optional_and_round_trips(self, tmp_path):
        """target_data_path round-trips from YAML; default is None."""
        p = tmp_path / "with_target.yaml"
        p.write_text("data_path: x.mat\ntarget_data_path: y.mat\n")
        cfg = EvalConfig.from_yaml(p)
        assert cfg.target_data_path == "y.mat"

        # Default (field absent): target_data_path is None
        p2 = tmp_path / "no_target.yaml"
        p2.write_text("data_path: x.mat\n")
        cfg2 = EvalConfig.from_yaml(p2)
        assert cfg2.target_data_path is None


# ---------------------------------------------------------------------------
# TestEvalConfigTransformSpec — MODE-02 field declaration (Plan 31-01)
# ---------------------------------------------------------------------------


class TestEvalConfigTransformSpec:
    """MODE-02: EvalConfig.transform_spec field declaration and YAML round-trip."""

    def test_default_is_none(self):
        """transform_spec defaults to None when not supplied."""
        cfg = EvalConfig(data_path="x")
        assert cfg.transform_spec is None

    def test_accepts_rigid_dict(self):
        """transform_spec accepts a rigid transform dict and round-trips equal to the input."""
        spec = {"type": "rigid", "rotation_deg": 30.0, "rotation_axis": [0, 0, 1]}
        cfg = EvalConfig(data_path="x", transform_spec=spec)
        assert cfg.transform_spec == spec

    def test_accepts_noise_dict(self):
        """transform_spec accepts a noise transform dict without raising."""
        spec = {"type": "noise", "sigma": 0.1}
        cfg = EvalConfig(data_path="x", transform_spec=spec)
        assert cfg.transform_spec == spec

    def test_yaml_round_trip(self, tmp_path):
        """EvalConfig.from_yaml loads a YAML with transform_spec block; field equals the parsed dict."""
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "data_path: x\n"
            "transform_spec:\n"
            "  type: rigid\n"
            "  rotation_deg: 30.0\n"
            "  rotation_axis: [0, 0, 1]\n"
        )
        cfg = EvalConfig.from_yaml(p)
        assert cfg.transform_spec == {
            "type": "rigid",
            "rotation_deg": 30.0,
            "rotation_axis": [0, 0, 1],
        }

    def test_rejects_unknown_sibling_field(self, tmp_path):
        """YAML with transform_spec_typo: {} raises EvalConfigError (extra='forbid' still applies)."""
        p = tmp_path / "cfg.yaml"
        p.write_text("data_path: x\ntransform_spec_typo: {}\n")
        with pytest.raises(EvalConfigError):
            EvalConfig.from_yaml(p)


# ---------------------------------------------------------------------------
# TestDataFactoryConstruction — D-08 lazy init (Plan 17-02)
# ---------------------------------------------------------------------------


class TestDataFactoryConstruction:
    """D-08 lazy init — DataFactory __init__ stores config and performs no I/O."""

    def test_init_is_lazy(self):
        """D-08: __init__ stores config and does NOT load data."""
        cfg = EvalConfig(data_path="x.mat")
        factory = DataFactory(cfg)
        assert factory._real_dataset is None
        assert factory._synthetic_dataset is None
        assert factory._target_dataset is None
        assert factory.config is cfg


# ---------------------------------------------------------------------------
# TestLoadReal — D-09 caching + dispatch by data_format (Plan 17-02)
# ---------------------------------------------------------------------------


class TestLoadReal:
    """D-09 caching + dispatch by data_format."""

    def _make_mock_ds(self):
        """Build a minimal mock dataset for patching."""
        return {0: zRegPointCloud(pos=torch.zeros(3, 3), label=None, id=torch.arange(3))}

    def test_dispatches_tracklets(self):
        """data_format='tracklets' calls load_data_from_tracklets with device=cfg.device."""
        cfg = EvalConfig(data_path="x.mat", data_format="tracklets")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
            result = factory.load_real()
        m.assert_called_once_with("x.mat", device=cfg.device)
        # _standardize produces a new dict (pos scaled); check keys are preserved
        assert set(result.keys()) == set(mock_ds.keys())

    def test_dispatches_csv(self):
        """data_format='csv' calls load_shah_from_csv with device=cfg.device (Pitfall 5)."""
        cfg = EvalConfig(data_path="x.csv", data_format="csv")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_shah_from_csv", return_value=mock_ds) as m:
            factory.load_real()
        m.assert_called_once_with("x.csv", device=cfg.device)

    def test_caches(self):
        """D-09: second load_real() call returns same reference and does not re-invoke loader."""
        cfg = EvalConfig(data_path="x.mat", data_format="tracklets")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
            first = factory.load_real()
            second = factory.load_real()
        assert first is second
        assert m.call_count == 1


# ---------------------------------------------------------------------------
# TestLoadTarget — D-09 caching + dispatch by data_format for target (Plan 30-01)
# ---------------------------------------------------------------------------


class TestLoadTarget:
    """load_target() caching, format dispatch, and EvalConfigError guard (D-05)."""

    def _make_mock_ds(self):
        """Build a minimal mock dataset for patching."""
        return {0: zRegPointCloud(pos=torch.zeros(3, 3), label=None, id=torch.arange(3))}

    def test_dispatches_tracklets(self):
        """data_format='tracklets' calls load_data_from_tracklets with target_data_path and device=cfg.device."""
        cfg = EvalConfig(data_path="x.mat", target_data_path="y.mat", data_format="tracklets")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
            result = factory.load_target()
        m.assert_called_once_with("y.mat", device=cfg.device)
        # _standardize produces a new dict (pos scaled); check keys are preserved
        assert set(result.keys()) == set(mock_ds.keys())

    def test_dispatches_csv(self):
        """data_format='csv' calls load_shah_from_csv with target_data_path and device=cfg.device (Pitfall 5)."""
        cfg = EvalConfig(data_path="x.csv", target_data_path="y.csv", data_format="csv")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_shah_from_csv", return_value=mock_ds) as m:
            factory.load_target()
        m.assert_called_once_with("y.csv", device=cfg.device)

    def test_caches(self):
        """D-09: second load_target() call returns same reference; loader invoked exactly once."""
        cfg = EvalConfig(data_path="x.mat", target_data_path="y.mat", data_format="tracklets")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
            first = factory.load_target()
            second = factory.load_target()
        assert first is second
        assert m.call_count == 1

    def test_raises_eval_config_error_when_target_path_none(self):
        """D-05: load_target() raises EvalConfigError (not ValueError) when target_data_path is None."""
        cfg = EvalConfig(data_path="x.mat", target_data_path=None)
        factory = DataFactory(cfg)
        with pytest.raises(EvalConfigError, match="target_data_path"):
            factory.load_target()


# ---------------------------------------------------------------------------
# TestEvalConfigTargetDataFormat — HETERO-01 field acceptance (Plan 32-01)
# ---------------------------------------------------------------------------


class TestEvalConfigTargetDataFormat:
    """EvalConfig accepts target_data_format field; None default is backward-compatible."""

    def test_accepts_csv(self):
        """target_data_format='csv' is accepted without EvalConfigError."""
        cfg = EvalConfig(data_path="x.mat", target_data_format="csv")
        assert cfg.target_data_format == "csv"

    def test_accepts_tracklets(self):
        """target_data_format='tracklets' is accepted without EvalConfigError."""
        cfg = EvalConfig(data_path="x.mat", target_data_format="tracklets")
        assert cfg.target_data_format == "tracklets"

    def test_default_is_none(self):
        """target_data_format defaults to None (backward compatible — existing configs unaffected)."""
        cfg = EvalConfig(data_path="x.mat")
        assert cfg.target_data_format is None


# ---------------------------------------------------------------------------
# TestLoadTargetFormatDispatch — HETERO-01 format-override dispatch (Plan 32-01)
# ---------------------------------------------------------------------------


class TestLoadTargetFormatDispatch:
    """load_target() uses target_data_format when set; falls back to data_format when None."""

    def _make_mock_ds(self):
        return {0: zRegPointCloud(pos=torch.zeros(3, 3), label=None, id=torch.arange(3))}

    def test_target_format_csv_overrides_source_tracklets(self):
        """target_data_format='csv' routes to CSV loader even though data_format='tracklets'."""
        cfg = EvalConfig(
            data_path="x.mat",
            data_format="tracklets",
            target_data_path="y.csv",
            target_data_format="csv",
        )
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_shah_from_csv", return_value=mock_ds) as m:
            result = factory.load_target()
        m.assert_called_once_with("y.csv", device="cpu")
        # _standardize produces a new dict (pos scaled); check keys are preserved
        assert set(result.keys()) == set(mock_ds.keys())

    def test_target_format_tracklets_overrides_source_csv(self):
        """target_data_format='tracklets' routes to tracklets loader even though data_format='csv'."""
        cfg = EvalConfig(
            data_path="x.csv",
            data_format="csv",
            target_data_path="y.mat",
            target_data_format="tracklets",
        )
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
            result = factory.load_target()
        m.assert_called_once_with("y.mat", device="cpu")
        # _standardize produces a new dict (pos scaled); check keys are preserved
        assert set(result.keys()) == set(mock_ds.keys())

    def test_none_falls_back_to_data_format_csv(self):
        """target_data_format=None falls back to data_format='csv' (backward compat)."""
        cfg = EvalConfig(
            data_path="x.csv",
            data_format="csv",
            target_data_path="y.csv",
            target_data_format=None,
        )
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_shah_from_csv", return_value=mock_ds) as m:
            factory.load_target()
        m.assert_called_once_with("y.csv", device="cpu")

    def test_none_falls_back_to_data_format_tracklets(self):
        """target_data_format=None falls back to data_format='tracklets' (backward compat)."""
        cfg = EvalConfig(
            data_path="x.mat",
            data_format="tracklets",
            target_data_path="y.mat",
            target_data_format=None,
        )
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
            factory.load_target()
        m.assert_called_once_with("y.mat", device="cpu")


# ---------------------------------------------------------------------------
# TestGenerateTarget — D-01/D-02/D-03 generate_target() (Plan 31-01)
# ---------------------------------------------------------------------------


class TestGenerateTarget:
    """D-01/D-02/D-03: generate_target() augment dispatch, config save/restore, instance state."""

    def _make_small_ds(self):
        """2-frame dataset with 5 points each, no fps-idx."""
        torch.manual_seed(7)
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.rand(5, 3),
                label=None,
                id=torch.arange(5),
            )
            pc["fps-idx"] = None
            ds[i] = pc
        return ds

    def test_init_attributes_are_none(self):
        """D-03: _synthetic_target, _source_dataset, and _transform_spec initialise to None."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        assert factory._synthetic_target is None
        assert factory._source_dataset is None
        assert factory._transform_spec is None
        assert factory._correspondence_idx is None  # Phase 56 D-03/D-04

    def test_generate_target_resets_correspondence_idx_when_noise_only(self):
        """Phase 56 D-03: noise-only generate_target() leaves _correspondence_idx None
        (never touches drop_points/sample_new_points)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        factory.generate_target(ds, {"type": "noise", "sigma": 0.1})
        assert factory._correspondence_idx is None

    def test_generate_target_with_noise_returns_distinct_dataset(self):
        """D-01: noise transform produces pos tensors different from input dataset."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        out = factory.generate_target(ds, {"type": "noise", "sigma": 0.1})
        for k in ds:
            assert not torch.equal(out[k]["pos"], ds[k]["pos"])

    def test_generate_target_stores_instance_state(self):
        """D-03: _synthetic_target, _source_dataset, _transform_spec set after generate_target."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        spec = {"type": "noise", "sigma": 0.1}
        out = factory.generate_target(ds, spec)
        assert factory._synthetic_target is out
        assert factory._source_dataset is ds
        assert factory._transform_spec == spec

    def test_config_restored_after_generate_target(self):
        """D-02: augmentation_params restored to original value after generate_target."""
        cfg = EvalConfig(data_path="x", augmentation_params={"sigma": 0.5})
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        factory.generate_target(ds, {"type": "noise", "sigma": 0.1})
        assert factory.config.augmentation_params == {"sigma": 0.5}

    def test_config_restored_when_augment_raises(self):
        """D-02: augmentation_params restored even when augment() raises (try/finally guarantee)."""
        from unittest.mock import patch as _patch
        cfg = EvalConfig(data_path="x", augmentation_params={"sigma": 0.5})
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        with _patch.object(factory, "augment", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                factory.generate_target(ds, {"type": "noise", "sigma": 0.1})
        assert factory.config.augmentation_params == {"sigma": 0.5}

    def test_raises_on_none_transform_spec(self):
        """generate_target(ds, None) raises ValueError."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        with pytest.raises(ValueError):
            factory.generate_target(ds, None)

    def test_raises_on_empty_transform_spec(self):
        """generate_target(ds, {}) raises ValueError."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        with pytest.raises(ValueError):
            factory.generate_target(ds, {})

    def test_raises_on_type_only_spec(self):
        """generate_target(ds, {'type': 'rigid'}) raises ValueError (augment_params empty after stripping)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        with pytest.raises(ValueError):
            factory.generate_target(ds, {"type": "rigid"})

    def test_type_key_is_stripped_before_dispatch(self):
        """D-02: 'type' key stripped before augment dispatch; noise spec succeeds despite 'type' present."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        # This must NOT raise, even though 'type' is not a recognised augment key
        out = factory.generate_target(ds, {"type": "noise", "sigma": 0.1})
        assert isinstance(out, dict)

    def test_rigid_transform_changes_positions(self):
        """generate_target with rigid rotation_deg changes output pos from input pos."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_small_ds()
        out = factory.generate_target(
            ds, {"type": "rigid", "rotation_deg": 30.0, "rotation_axis": [0, 0, 1]}
        )
        for k in ds:
            assert not torch.equal(out[k]["pos"], ds[k]["pos"])


# ---------------------------------------------------------------------------
# TestGenerateSubsamplePair — Phase 57 GT-04/GT-05: generate_subsample_pair()
# ---------------------------------------------------------------------------


class TestGenerateSubsamplePair:
    """generate_subsample_pair() — subsample-pair generation with tracked correspondence."""

    def _make_ds_100pts(self):
        """Single-frame, 100-point dataset with populated label and id fields."""
        pc = zRegPointCloud(
            pos=torch.randn(100, 3),
            label=torch.arange(100, dtype=torch.long),
            id=torch.arange(100),
        )
        pc["fps-idx"] = None
        return {0: pc}

    def test_list_seed_raises_value_error(self):
        """seed as a list is rejected (D-06/D-07 — optimizer.py-level concern)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        with pytest.raises(ValueError, match="seed"):
            factory.generate_subsample_pair(ds, {"seed": [1, 2]})

    def test_missing_dataset_when_not_synthesize_raises_value_error(self):
        """dataset=None with synthesize not set (defaults False) raises ValueError naming 'synthesize'."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        with pytest.raises(ValueError, match="synthesize"):
            factory.generate_subsample_pair(None, {"source_fraction": 0.5})

    def test_source_fraction_zero_raises_value_error(self):
        """source_fraction=0.0 is out of the (0.0, 1.0] range."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        with pytest.raises(ValueError, match="source_fraction"):
            factory.generate_subsample_pair(ds, {"source_fraction": 0.0})

    def test_source_fraction_above_one_raises_value_error(self):
        """source_fraction=1.1 is out of the (0.0, 1.0] range."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        with pytest.raises(ValueError, match="source_fraction"):
            factory.generate_subsample_pair(ds, {"source_fraction": 1.1})

    def test_target_fraction_zero_raises_value_error(self):
        """target_fraction=0.0 is out of the (0.0, 1.0] range."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        with pytest.raises(ValueError, match="target_fraction"):
            factory.generate_subsample_pair(ds, {"target_fraction": 0.0})

    def test_target_fraction_above_one_raises_value_error(self):
        """target_fraction=1.1 is out of the (0.0, 1.0] range."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        with pytest.raises(ValueError, match="target_fraction"):
            factory.generate_subsample_pair(ds, {"target_fraction": 1.1})

    def test_run_alignment_true_applies_real_geometric_transform(self):
        """D-03: with run_alignment=True (default), the target view's retained
        points are NOT bitwise-identical to the corresponding base positions —
        proving a real perturbation (rotation/scale) was layered on top of the
        subsampling, not just an identity subsample."""
        cfg = EvalConfig(data_path="x")  # run_alignment=True by default
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        source, target = factory.generate_subsample_pair(
            ds, {"source_fraction": 0.8, "target_fraction": 0.8, "seed": 42}
        )
        corr = factory._correspondence_idx[0]
        valid = corr >= 0
        pre_perturbation_expected = ds[0]["pos"][corr[valid]]
        assert not torch.allclose(pre_perturbation_expected, target[0]["pos"][valid])

    def test_run_alignment_true_source_and_target_are_different_selections(self):
        """The seed+1 target-seed offset produces genuinely different retained
        points between source and target views (not the same subsample twice)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        source, target = factory.generate_subsample_pair(
            ds, {"source_fraction": 0.8, "target_fraction": 0.8, "seed": 42}
        )
        assert (
            source[0]["pos"].shape[0] != target[0]["pos"].shape[0]
            or not torch.equal(source[0]["pos"], target[0]["pos"])
        )

    def test_run_alignment_false_no_perturbation_layered(self):
        """D-03: with run_alignment=False, the target view's retained-point
        positions exactly equal the corresponding base positions (no rotation/
        scale/noise applied) — source and target stay in the same coordinate
        frame."""
        cfg = EvalConfig(data_path="x", run_alignment=False)
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        source, target = factory.generate_subsample_pair(
            ds, {"source_fraction": 0.8, "target_fraction": 0.8, "seed": 42}
        )
        corr = factory._correspondence_idx[0]
        valid = corr >= 0
        assert torch.equal(ds[0]["pos"][corr[valid]], target[0]["pos"][valid])

    def test_instance_state_after_explicit_dataset_call(self):
        """After a non-synthesize call, _subsample_source_view/_source_dataset/
        _synthetic_target are set per the producer contract."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        source, target = factory.generate_subsample_pair(
            ds, {"source_fraction": 0.8, "target_fraction": 0.8, "seed": 42}
        )
        assert factory._subsample_source_view is source
        assert factory._source_dataset is ds
        assert factory._synthetic_target is target

    def test_get_synthetic_ground_truth_matches_target_length_and_values(self):
        """get_synthetic_ground_truth() after generate_subsample_pair returns a
        per-frame tensor whose length equals the target view's point count and
        whose non-sentinel entries equal the base dataset's label field gathered
        at the tracked correspondence (mirrors Phase 56 gather-correctness)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        source, target = factory.generate_subsample_pair(
            ds, {"source_fraction": 0.8, "target_fraction": 0.8, "seed": 42}
        )
        gt = factory.get_synthetic_ground_truth()
        assert gt[0].shape[0] == target[0]["pos"].shape[0]
        corr = factory._correspondence_idx[0]
        valid = corr >= 0
        assert torch.equal(gt[0][valid], ds[0][cfg.ground_truth_field][corr[valid]])

    def test_synthesize_mode_produces_labeled_pair_without_dataset(self):
        """synthesize=True mode produces a labeled source/target pair with no
        dataset argument required."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        source, target = factory.generate_subsample_pair(
            None, {"synthesize": True, "seed": 7, "n_classes": 3}
        )
        assert source[0]["label"] is not None
        assert target[0]["label"] is not None

    def test_reproducibility_across_fresh_factory_instances(self):
        """Two calls with identical transform_spec (including identical seed)
        on two separate, fresh DataFactory instances produce bitwise-identical
        source/target positions."""
        cfg = EvalConfig(data_path="x")
        ds = self._make_ds_100pts()
        spec = {"source_fraction": 0.8, "target_fraction": 0.8, "seed": 42}

        factory_a = DataFactory(cfg)
        source_a, target_a = factory_a.generate_subsample_pair(ds, dict(spec))

        factory_b = DataFactory(cfg)
        source_b, target_b = factory_b.generate_subsample_pair(ds, dict(spec))

        assert torch.equal(source_a[0]["pos"], source_b[0]["pos"])
        assert torch.equal(target_a[0]["pos"], target_b[0]["pos"])

    def test_synthesize_explicit_ball_shape(self):
        """synthesize=True with shape='ball' explicitly set skips auto-assignment (498->500 arc)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        source, target = factory.generate_subsample_pair(
            None, {"synthesize": True, "seed": 1, "n_classes": 3, "shape": "ball", "n_points": 50}
        )
        assert source[0]["pos"].shape[1] == 3
        assert target[0]["pos"].shape[1] == 3

    def test_synthesize_explicit_bowl_shape(self):
        """synthesize=True with shape='bowl' explicitly set skips auto-assignment (498->500 arc)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        source, target = factory.generate_subsample_pair(
            None, {"synthesize": True, "seed": 2, "n_classes": 3, "shape": "bowl", "n_points": 50}
        )
        assert source[0]["pos"].shape[1] == 3

    def test_synthesize_unknown_shape_raises_value_error(self):
        """synthesize=True with an unknown shape raises ValueError (line 505)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        with pytest.raises(ValueError, match="unknown shape"):
            factory.generate_subsample_pair(
                None, {"synthesize": True, "seed": 1, "n_classes": 3, "shape": "cube", "n_points": 50}
            )

    def test_run_alignment_with_explicit_perturb_keys_uses_them(self):
        """Explicit rotation_deg in transform_spec skips auto-perturb generation (536->547 arc)."""
        cfg = EvalConfig(data_path="x", run_alignment=True)
        factory = DataFactory(cfg)
        source, target = factory.generate_subsample_pair(
            None,
            {
                "synthesize": True,
                "seed": 5,
                "n_classes": 3,
                "n_points": 50,
                "source_fraction": 0.8,
                "target_fraction": 0.8,
                "rotation_deg": 30.0,
                "rotation_axis": [0.0, 0.0, 1.0],
                "scale_factor": 1.0,
            },
        )
        assert source[0]["pos"].shape[1] == 3
        assert target[0]["pos"].shape[1] == 3


# ---------------------------------------------------------------------------
# TestGenerateSynthetic — caching + seed reproducibility (Plan 17-02)
# ---------------------------------------------------------------------------


class TestGenerateSynthetic:
    """Caching and seed reproducibility for generate_synthetic()."""

    def test_seed_reproducible(self):
        """Two DataFactory instances with equal config produce bitwise-equal trajectories."""
        cfg = EvalConfig(data_path="x", n_synthetic=3)
        f_a = DataFactory(cfg)
        f_b = DataFactory(cfg)
        traj_a = f_a.generate_synthetic()
        traj_b = f_b.generate_synthetic()
        assert len(traj_a) == 3
        for i in traj_a:
            assert torch.equal(traj_a[i]["pos"], traj_b[i]["pos"])

    def test_caches(self):
        """D-09: second generate_synthetic() call returns same object reference."""
        cfg = EvalConfig(data_path="x", n_synthetic=2)
        factory = DataFactory(cfg)
        first = factory.generate_synthetic()
        second = factory.generate_synthetic()
        assert first is second


# ---------------------------------------------------------------------------
# TestAugment — augmentation_params dispatch, 4 cases (Plan 17-02)
# ---------------------------------------------------------------------------


class TestAugment:
    """augmentation_params dispatch: no-op, sigma, n_outliers, and both."""

    def _baseline(self):
        """Build a small synthetic trajectory with 2 frames."""
        cfg = EvalConfig(data_path="x", n_synthetic=2)
        factory = DataFactory(cfg)
        return factory.generate_synthetic()

    def test_empty_params_noop(self):
        """augment with augmentation_params={} returns input unchanged (identity)."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._baseline()
        out = factory.augment(ds)
        assert out is ds

    def test_sigma_only(self):
        """augment with {'sigma': 0.01} applies add_gaussian_noise — positions differ from input."""
        cfg = EvalConfig(data_path="x", augmentation_params={"sigma": 0.01})
        factory = DataFactory(cfg)
        ds = self._baseline()
        out = factory.augment(ds)
        for i in ds:
            assert not torch.equal(out[i]["pos"], ds[i]["pos"])

    def test_outliers_only(self):
        """augment with {'n_outliers': 5} appends 5 points per frame (add_outliers contract)."""
        cfg = EvalConfig(data_path="x", augmentation_params={"n_outliers": 5})
        factory = DataFactory(cfg)
        ds = self._baseline()
        baseline_n = ds[0]["pos"].shape[0]
        out = factory.augment(ds)
        for i in ds:
            assert out[i]["pos"].shape[0] == baseline_n + 5

    def test_both(self):
        """augment with both keys applies noise THEN outliers (composition order)."""
        cfg = EvalConfig(data_path="x", augmentation_params={"sigma": 0.01, "n_outliers": 5})
        factory = DataFactory(cfg)
        ds = self._baseline()
        baseline_n = ds[0]["pos"].shape[0]
        out = factory.augment(ds)
        # Outliers added: shape grows by 5
        for i in ds:
            assert out[i]["pos"].shape[0] == baseline_n + 5
        # Noise applied first: positions of first baseline_n rows differ from input
        for i in ds:
            assert not torch.equal(out[i]["pos"][:baseline_n], ds[i]["pos"])


# ---------------------------------------------------------------------------
# TestPrepareSplit — FRAME-02 split-shape gate (Plan 17-02)
# ---------------------------------------------------------------------------


class TestPrepareSplit:
    """FRAME-02 split-shape gate: disjoint keys, sorted order, 1-frame guard."""

    def _make_ds(self, n):
        """Build a dataset with n frames, each with a single zero point."""
        return {i: zRegPointCloud(pos=torch.zeros(2, 3), label=None, id=torch.arange(2)) for i in range(n)}

    def test_split_returns_two_dicts_with_disjoint_keys(self):
        """FRAME-02 gate 2: 10-frame dataset with val_split=0.2 → 8 train + 2 val, disjoint, covering all keys."""
        ds = self._make_ds(10)
        cfg = EvalConfig(data_path="x", val_split=0.2)
        factory = DataFactory(cfg)
        train, val = factory.prepare_split(ds)
        assert isinstance(train, dict) and isinstance(val, dict)
        assert len(train) == 8 and len(val) == 2
        assert set(train.keys()).isdisjoint(set(val.keys()))
        assert set(train.keys()) | set(val.keys()) == set(ds.keys())

    def test_split_keys_are_sorted(self):
        """Both returned dicts have ascending-sorted keys (D-05)."""
        ds = self._make_ds(10)
        cfg = EvalConfig(data_path="x", val_split=0.2)
        factory = DataFactory(cfg)
        train, val = factory.prepare_split(ds)
        assert list(train.keys()) == sorted(train.keys())
        assert list(val.keys()) == sorted(val.keys())

    def test_single_frame_returns_empty_val(self):
        """D-07: 1-frame dataset returns (dataset, {}) silently."""
        ds = {0: zRegPointCloud(pos=torch.zeros(2, 3), label=None, id=torch.arange(2))}
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        train, val = factory.prepare_split(ds)
        assert train == ds
        assert val == {}


# ---------------------------------------------------------------------------
# TestGetGroundTruth — D-10/D-11 id extraction (Plan 17-02)
# ---------------------------------------------------------------------------


class TestGetGroundTruth:
    """D-10/D-11: id extraction from in-memory dataset and external GT path."""

    def test_extracts_id_from_pc(self):
        """D-10, D-11, Phase 56 D-02: returns {i: ds[i]['id']} when ground_truth_path
        is None and ground_truth_field is explicitly overridden to 'id'."""
        ds = {
            i: zRegPointCloud(pos=torch.zeros(2, 3), label=None, id=torch.tensor([i, i + 100]))
            for i in range(3)
        }
        cfg = EvalConfig(data_path="x", ground_truth_field="id")
        factory = DataFactory(cfg)
        gt = factory.get_ground_truth(ds)
        assert set(gt.keys()) == set(ds.keys())
        for i in ds:
            assert torch.equal(gt[i], ds[i]["id"])

    def test_extracts_label_from_pc_by_default(self):
        """Phase 56 D-01: with default config (ground_truth_field='label'),
        get_ground_truth returns pc['label'] when the dataset has a populated,
        distinct label field."""
        ds = {
            i: zRegPointCloud(
                pos=torch.zeros(2, 3),
                label=torch.tensor([i, i + 100]),
                id=torch.tensor([i * 1000, i * 1000 + 1]),
            )
            for i in range(3)
        }
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        gt = factory.get_ground_truth(ds)
        assert set(gt.keys()) == set(ds.keys())
        for i in ds:
            assert torch.equal(gt[i], ds[i]["label"])

    def test_external_gt_path(self):
        """D-10, Phase 56 D-02: when ground_truth_path is set, loads via same loader
        as data_format, and reads the 'id' field explicitly overridden here."""
        cfg = EvalConfig(
            data_path="x.mat", data_format="tracklets", ground_truth_path="gt.mat",
            ground_truth_field="id",
        )
        factory = DataFactory(cfg)
        external_ds = {0: zRegPointCloud(pos=torch.zeros(3, 3), label=None, id=torch.arange(3))}
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(external_ds, {})) as m:
            gt = factory.get_ground_truth({})  # in-memory ds is irrelevant when external path set
        m.assert_called_once_with("gt.mat", device="cpu")
        for i in external_ds:
            assert torch.equal(gt[i], external_ds[i]["id"])


# ---------------------------------------------------------------------------
# TestGetSyntheticGroundTruth — D-04/D-05/D-06 id-or-ordinal GT (Plan 31-01)
# ---------------------------------------------------------------------------


class TestGetSyntheticGroundTruth:
    """D-04/D-05/D-06: get_synthetic_ground_truth() id-or-ordinal fallback, RuntimeError guard, torch.long dtype."""

    def _make_ds_with_ids(self):
        """2-frame source dataset with pc['id'] = torch.arange(5, dtype=torch.float32)."""
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.randn(5, 3),
                label=None,
                id=torch.arange(5, dtype=torch.float32),
            )
            pc["fps-idx"] = None
            ds[i] = pc
        return ds

    def _make_ds_no_ids(self):
        """2-frame source dataset with pc['id'] = None."""
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.randn(5, 3),
                label=None,
                id=None,
            )
            pc["fps-idx"] = None
            ds[i] = pc
        return ds

    def test_raises_runtime_error_before_generate_target(self):
        """get_synthetic_ground_truth() raises RuntimeError when _synthetic_target is None."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        with pytest.raises(RuntimeError, match="generate_target"):
            factory.get_synthetic_ground_truth()

    def test_returns_id_tensors_when_source_has_ids(self):
        """D-04: returns pc['id'] cast to torch.long when source frames have ids.

        Phase 56 note: fixture has label=None, so under the new
        ground_truth_field='label' default this actually exercises the
        ordinal-fallback path (D-05) — it passes because
        torch.arange(5).to(torch.long) coincidentally equals the ordinal
        fallback tensor. Kept for regression coverage of that coincidence;
        see test_extracts_id_from_pc in TestGetGroundTruth for an explicit
        'id'-override test.
        """
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_with_ids()
        factory.generate_target(ds, {"type": "noise", "sigma": 0.01})
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            expected = ds[k]["id"].to(torch.long)
            assert torch.equal(gt[k], expected)

    def test_returns_ordinal_fallback_when_source_id_is_none(self):
        """D-05: returns torch.arange(n, dtype=torch.long) per frame when pc['id'] is None.

        Phase 56 note: fixture also has label=None, so under the new
        ground_truth_field='label' default this exercises the ordinal
        fallback via the 'label' field being None (same fallback code path
        as before, just reached via a different field name).
        """
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_no_ids()
        factory.generate_target(ds, {"type": "noise", "sigma": 0.01})
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            n = ds[k]["pos"].shape[0]
            expected = torch.arange(n, dtype=torch.long)
            assert torch.equal(gt[k], expected)

    def test_dtype_is_torch_long_for_id_path(self):
        """D-06: returned tensors have dtype torch.long when source ids are float32.

        Phase 56 note: fixture has label=None, so this exercises the ordinal
        fallback (torch.long by construction) rather than the id field itself
        under the new ground_truth_field='label' default.
        """
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_with_ids()
        factory.generate_target(ds, {"type": "noise", "sigma": 0.01})
        gt = factory.get_synthetic_ground_truth()
        for k in gt:
            assert gt[k].dtype == torch.long

    def test_dtype_is_torch_long_for_ordinal_path(self):
        """D-06: returned tensors have dtype torch.long for ordinal fallback path."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_no_ids()
        factory.generate_target(ds, {"type": "noise", "sigma": 0.01})
        gt = factory.get_synthetic_ground_truth()
        for k in gt:
            assert gt[k].dtype == torch.long

    def test_keys_match_source_dataset_keys(self):
        """Returned dict keys exactly equal source dataset keys."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_with_ids()
        factory.generate_target(ds, {"type": "noise", "sigma": 0.01})
        gt = factory.get_synthetic_ground_truth()
        assert set(gt.keys()) == set(ds.keys())

    def _make_ds_20pts_with_labels(self):
        """2-frame, 20-point source dataset with a populated, distinct label field."""
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.randn(20, 3),
                label=torch.arange(20, dtype=torch.long),
                id=None,
            )
            pc["fps-idx"] = None
            ds[i] = pc
        return ds

    def test_gather_matches_target_length_after_dropout(self):
        """Phase 56 D-03/D-04 (GT-02): after dropout_fraction, gt[k] length matches
        the target's actual post-dropout point count (gathered by tracked
        correspondence), not the original 20-point source count. Values are
        gathered from the retained original positions, proving correctness
        beyond just length."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_20pts_with_labels()
        factory.generate_target(ds, {"type": "noise", "dropout_fraction": 0.3, "augment_seed": 42})
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            target_n = factory._synthetic_target[k]["pos"].shape[0]
            assert target_n < 20  # dropout actually removed points
            assert gt[k].shape[0] == target_n
            # gathered values equal the original labels at the tracked correspondence
            corr = factory._correspondence_idx[k]
            assert torch.equal(gt[k], ds[k]["label"][corr])

    def test_gather_appends_sentinel_for_new_points(self):
        """GT-02: after n_new_points, gt[k] length = original + n_new, with the
        first `n_original` entries equal to the source field and a -1 sentinel
        for the newly-added points (no original-source correspondence)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_20pts_with_labels()
        factory.generate_target(ds, {"type": "extend", "n_new_points": 4, "augment_seed": 3})
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            assert gt[k].shape[0] == 24  # 20 + 4
            assert torch.equal(gt[k][:20], ds[k][factory.config.ground_truth_field])
            assert (gt[k][20:] == -1).all()

    def test_gather_combined_dropout_and_new_points(self):
        """GT-02: combined dropout_fraction + n_new_points on a 20-point source
        returns gt[k].shape[0] == round(20*0.5) + 3 == 13, with the last 3
        entries sentinel -1."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_20pts_with_labels()
        factory.generate_target(
            ds, {"dropout_fraction": 0.5, "n_new_points": 3, "augment_seed": 7}
        )
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            assert gt[k].shape[0] == 13
            assert (gt[k][-3:] == -1).all()

    def test_compute_f1_accepts_gathered_length_without_shape_error(self):
        """GT-02: the length produced by get_synthetic_ground_truth is directly
        usable by compute_f1 without further truncation — no ValueError for
        shape mismatch when y_pred is constructed with the same length."""
        from zreg.evaluation.label_transfer import compute_f1

        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        ds = self._make_ds_20pts_with_labels()
        factory.generate_target(
            ds, {"dropout_fraction": 0.5, "n_new_points": 3, "augment_seed": 7}
        )
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            y_true = gt[k]
            y_pred = torch.zeros_like(y_true)
            score = compute_f1(y_true, y_pred)  # must not raise ValueError
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestScale — scale(dataset, factor) (Plan 27-01)
# ---------------------------------------------------------------------------


class TestScale:
    """scale(dataset, factor) multiplies pos and preserves other fields."""

    def _make_ds(self):
        """3-frame dataset with known pos values and fps-idx populated."""
        ds = {}
        for i in range(3):
            pc = zRegPointCloud(
                pos=torch.ones(4, 3) * (i + 1),
                label=None,
                id=torch.arange(4),
            )
            pc["fps-idx"] = torch.arange(4)
            ds[i] = pc
        return ds

    def test_pos_scaled(self):
        """pos tensors are multiplied by factor in every frame."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds()
        out = factory.scale(ds, 2.0)
        for i in ds:
            assert torch.allclose(out[i]["pos"], ds[i]["pos"] * 2.0)

    def test_input_not_mutated(self):
        """Input dataset pos tensors are not modified in place."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds()
        original_pos = {i: ds[i]["pos"].clone() for i in ds}
        factory.scale(ds, 3.0)
        for i in ds:
            assert torch.equal(ds[i]["pos"], original_pos[i])

    def test_factor_zero(self):
        """scale(ds, 0.0) produces all-zero pos tensors."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds()
        out = factory.scale(ds, 0.0)
        for i in ds:
            assert torch.allclose(out[i]["pos"], torch.zeros_like(ds[i]["pos"]))

    def test_fps_idx_preserved(self):
        """fps-idx field is passed through unchanged after scale."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds()
        out = factory.scale(ds, 2.0)
        for i in ds:
            assert torch.equal(out[i]["fps-idx"], ds[i]["fps-idx"])


# ---------------------------------------------------------------------------
# TestRotate — rotate(dataset, rotation_matrix) (Plan 27-01)
# ---------------------------------------------------------------------------


class TestRotate:
    """rotate(dataset, R) applies rigid rotation via apply_rigid."""

    def _make_ds(self):
        """Single-frame dataset with two known points."""
        return {
            0: zRegPointCloud(
                pos=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                label=None,
                id=torch.arange(2),
            )
        }

    def _identity_R(self):
        """Return 3x3 identity rotation matrix."""
        return torch.eye(3, dtype=torch.float32)

    def _rot90z(self):
        """Return 90-degree rotation around Z axis: [[0,-1,0],[1,0,0],[0,0,1]]."""
        return torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    def test_identity_is_noop(self):
        """rotate(ds, I) leaves pos unchanged (identity rotation)."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds()
        out = factory.rotate(ds, self._identity_R())
        assert torch.allclose(out[0]["pos"], ds[0]["pos"], atol=1e-5)

    def test_rotation_changes_pos(self):
        """rotate(ds, R_90z) changes pos of non-origin points."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds()
        out = factory.rotate(ds, self._rot90z())
        assert not torch.allclose(out[0]["pos"], ds[0]["pos"], atol=1e-5)

    def test_deep_copy_contract(self):
        """rotate returns a new dict — out[0] is not ds[0]."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds()
        out = factory.rotate(ds, self._identity_R())
        assert out[0] is not ds[0]

    def test_rotation_correctness(self):
        """90-degree Z rotation maps (1,0,0) -> (0,1,0) within atol=1e-5."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        single_pt = {0: zRegPointCloud(pos=torch.tensor([[1.0, 0.0, 0.0]]), label=None, id=torch.arange(1))}
        out = factory.rotate(single_pt, self._rot90z())
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(out[0]["pos"], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# TestDropPoints — drop_points(dataset, fraction, seed) (Plan 27-01)
# ---------------------------------------------------------------------------


class TestDropPoints:
    """drop_points(dataset, fraction) removes fraction of points per frame."""

    def _make_ds_100pts(self):
        """2-frame dataset, each with 100 points, populated fps-idx."""
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.zeros(100, 3),
                label=torch.zeros(100, dtype=torch.long),
                id=torch.arange(100),
            )
            pc["fps-idx"] = torch.arange(100)
            ds[i] = pc
        return ds

    def test_removes_fraction(self):
        """drop_points(ds, 0.3, seed=42) keeps round(100*0.7)=70 points per frame."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.drop_points(ds, 0.3, seed=42)
        for i in ds:
            assert out[i]["pos"].shape[0] == 70

    def test_fraction_zero_noop(self):
        """drop_points(ds, 0.0) returns same point count as input."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.drop_points(ds, 0.0)
        for i in ds:
            assert out[i]["pos"].shape[0] == 100

    def test_id_shape_consistent(self):
        """id field shape matches pos shape after drop_points."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.drop_points(ds, 0.3, seed=42)
        for i in ds:
            assert out[i]["id"].shape[0] == out[i]["pos"].shape[0]

    def test_fps_idx_shape_consistent(self):
        """fps-idx field shape matches pos shape after drop_points."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.drop_points(ds, 0.3, seed=42)
        for i in ds:
            assert out[i]["fps-idx"].shape[0] == out[i]["pos"].shape[0]

    def test_input_not_mutated(self):
        """Input dataset remains at 100 points after drop_points call."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        factory.drop_points(ds, 0.3, seed=42)
        assert ds[0]["pos"].shape[0] == 100

    def test_correspondence_idx_matches_id_field_on_fresh_factory(self):
        """Phase 56 D-03: on a fresh factory, _correspondence_idx[i] equals the
        idx tensor produced by torch.randperm — verified via ds['id']=arange(100)
        indexed by the same idx, so _correspondence_idx[i] == out[i]['id'] by
        construction."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.drop_points(ds, 0.3, seed=42)
        for i in ds:
            assert factory._correspondence_idx[i].shape[0] == 70
            assert torch.equal(factory._correspondence_idx[i], out[i]["id"])

    def test_drop_points_composes_prior_correspondence_idx(self):
        """Phase 56 D-03: second drop_points call without resetting _correspondence_idx
        composes indices (line 1262: new_corr[i] = self._correspondence_idx[i][idx])."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        # First call: _correspondence_idx is None → fresh index set
        out1 = factory.drop_points(ds, 0.2, seed=10)
        corr1 = {i: factory._correspondence_idx[i].clone() for i in ds}
        # Second call WITHOUT resetting: composes the new sub-index into the prior one
        out2 = factory.drop_points(out1, 0.25, seed=20)
        for i in ds:
            # Composed index must be a subset of the first-pass index
            assert factory._correspondence_idx[i].shape[0] == out2[i]["pos"].shape[0]
            assert factory._correspondence_idx[i].shape[0] < corr1[i].shape[0]


# ---------------------------------------------------------------------------
# TestSampleNewPoints — sample_new_points(dataset, n_extra, seed) (Plan 27-01)
# ---------------------------------------------------------------------------


class TestSampleNewPoints:
    """sample_new_points(dataset, n_extra) appends n_extra uniform-in-bbox points per frame."""

    def _make_ds_100pts(self):
        """2-frame dataset, each with 100 uniform random points, fps-idx populated."""
        torch.manual_seed(0)
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.rand(100, 3),
                label=torch.zeros(100, dtype=torch.long),
                id=torch.arange(100),
            )
            pc["fps-idx"] = torch.arange(100)
            ds[i] = pc
        return ds

    def test_adds_n_extra(self):
        """sample_new_points(ds, 10, seed=42) produces 110 points per frame."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.sample_new_points(ds, 10, seed=42)
        for i in ds:
            assert out[i]["pos"].shape[0] == 110

    def test_new_pts_within_bbox(self):
        """New points lie within the per-frame bounding box."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.sample_new_points(ds, 10, seed=42)
        for i in ds:
            bbox_min = ds[i]["pos"].min(dim=0).values
            bbox_max = ds[i]["pos"].max(dim=0).values
            new_pts = out[i]["pos"][100:]
            assert (new_pts >= bbox_min - 1e-6).all()
            assert (new_pts <= bbox_max + 1e-6).all()

    def test_id_shape_consistent(self):
        """id field has correct length; new entries are sentinel -1."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.sample_new_points(ds, 10, seed=42)
        for i in ds:
            assert out[i]["id"].shape[0] == 110
            assert (out[i]["id"][100:] == -1).all()

    def test_input_not_mutated(self):
        """Input dataset remains at 100 points after sample_new_points call."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        factory.sample_new_points(ds, 10, seed=42)
        assert ds[0]["pos"].shape[0] == 100

    def test_fps_idx_extended(self):
        """fps-idx field length matches pos length after sample_new_points."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.sample_new_points(ds, 10, seed=42)
        for i in ds:
            assert out[i]["fps-idx"].shape[0] == 110

    def test_correspondence_idx_fresh_factory(self):
        """Phase 56 D-03: on a fresh factory, _correspondence_idx[i] is
        arange(100) followed by 10 sentinel -1 entries."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        factory.sample_new_points(ds, 10, seed=42)
        for i in ds:
            corr = factory._correspondence_idx[i]
            assert corr.shape[0] == 110
            assert torch.equal(corr[:100], torch.arange(100, dtype=torch.long))
            assert (corr[100:] == -1).all()

    def test_correspondence_idx_chains_after_drop_points(self):
        """Phase 56 D-03/D-04: chaining drop_points then sample_new_points on the
        SAME factory instance composes correspondence correctly (matches
        augment()'s dispatch order: dropout_fraction before n_new_points)."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        dropped = factory.drop_points(ds, 0.3, seed=42)
        after_drop_corr = {i: t.clone() for i, t in factory._correspondence_idx.items()}
        extended = factory.sample_new_points(dropped, 5, seed=1)
        for i in ds:
            corr = factory._correspondence_idx[i]
            assert corr.shape[0] == 75  # 70 + 5
            assert torch.equal(corr[:70], after_drop_corr[i])
            assert (corr[70:] == -1).all()
        assert extended[0]["pos"].shape[0] == 75


# ---------------------------------------------------------------------------
# TestAugmentExtended — augment() with four new dispatch keys (Plan 27-02)
# ---------------------------------------------------------------------------


class TestAugmentExtended:
    """augment() extended dispatch: scale_factor, rotation_deg, dropout_fraction, n_new_points."""

    def _make_ds_100pts(self):
        """2-frame dataset, each with 100 random points, fps-idx set to None."""
        torch.manual_seed(99)
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.rand(100, 3),
                label=torch.zeros(100, dtype=torch.long),
                id=torch.arange(100),
            )
            pc["fps-idx"] = None
            ds[i] = pc
        return ds

    def test_scale_factor_key(self):
        """augment({'scale_factor': 2.0}) multiplies every frame's pos by 2.0."""
        cfg = EvalConfig(data_path="x", augmentation_params={"scale_factor": 2.0})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        original_pos = {i: ds[i]["pos"].clone() for i in ds}
        out = factory.augment(ds)
        for i in ds:
            assert torch.allclose(out[i]["pos"], original_pos[i] * 2.0)

    def test_rotation_deg_zero_noop(self):
        """augment({'rotation_deg': 0.0}) leaves pos unchanged (zero rotation is identity)."""
        cfg = EvalConfig(data_path="x", augmentation_params={"rotation_deg": 0.0})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        original_pos = {i: ds[i]["pos"].clone() for i in ds}
        out = factory.augment(ds)
        for i in ds:
            assert torch.allclose(out[i]["pos"], original_pos[i], atol=1e-5)

    def test_dropout_fraction_key(self):
        """augment({'dropout_fraction': 0.5}) on 100-point frames produces 50 points per frame."""
        cfg = EvalConfig(data_path="x", augmentation_params={"dropout_fraction": 0.5})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.augment(ds)
        for i in ds:
            assert out[i]["pos"].shape[0] == 50

    def test_n_new_points_key(self):
        """augment({'n_new_points': 10}) on 100-point frames produces 110 points per frame."""
        cfg = EvalConfig(data_path="x", augmentation_params={"n_new_points": 10})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.augment(ds)
        for i in ds:
            assert out[i]["pos"].shape[0] == 110

    def test_scale_then_dropout_chain(self):
        """augment({'scale_factor': 2.0, 'dropout_fraction': 0.5}): scale applied first, then dropout."""
        cfg = EvalConfig(data_path="x", augmentation_params={"scale_factor": 2.0, "dropout_fraction": 0.5})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.augment(ds)
        for i in ds:
            # Dropout: 50 points remain
            assert out[i]["pos"].shape[0] == 50
            # Scale was applied: out positions != original positions (scale 2.0 applied before dropout)
            assert not torch.allclose(out[i]["pos"], ds[i]["pos"][:50])

    def test_all_four_new_keys(self):
        """augment with all four new keys: final count == round(100*0.8)+5 == 85."""
        cfg = EvalConfig(
            data_path="x",
            augmentation_params={
                "scale_factor": 1.5,
                "rotation_deg": 45.0,
                "dropout_fraction": 0.2,
                "n_new_points": 5,
            },
        )
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.augment(ds)
        expected_count = round(100 * 0.8) + 5  # 80 + 5 = 85
        for i in ds:
            assert out[i]["pos"].shape[0] == expected_count

    def test_existing_sigma_unaffected(self):
        """augment({'sigma': 0.01}) still applies gaussian noise — existing path unaffected."""
        cfg = EvalConfig(data_path="x", augmentation_params={"sigma": 0.01})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        original_pos = {i: ds[i]["pos"].clone() for i in ds}
        out = factory.augment(ds)
        for i in ds:
            assert not torch.equal(out[i]["pos"], original_pos[i])

    def test_existing_outliers_unaffected(self):
        """augment({'n_outliers': 5}) on 100-point frames produces 105 points — existing path unaffected."""
        cfg = EvalConfig(data_path="x", augmentation_params={"n_outliers": 5})
        factory = DataFactory(cfg)
        ds = self._make_ds_100pts()
        out = factory.augment(ds)
        for i in ds:
            assert out[i]["pos"].shape[0] == 105


# ---------------------------------------------------------------------------
# Coverage gap tests — added to reach 100% coverage
# ---------------------------------------------------------------------------


class TestEvalConfigFromYAMLInvalidSyntax:
    """eval/config.py:198 — yaml.YAMLError branch in from_yaml."""

    def test_invalid_yaml_syntax_raises_eval_config_error(self, tmp_path):
        """YAML with invalid syntax raises EvalConfigError (yaml.YAMLError branch)."""
        p = tmp_path / "bad_syntax.yaml"
        # 'key: :' is invalid YAML syntax — triggers yaml.YAMLError
        p.write_text("key: :\n")
        with pytest.raises(EvalConfigError):
            EvalConfig.from_yaml(p)


class TestLoadRealUnknownFormat:
    """eval/data_factory.py:114 — ValueError for unknown data_format in load_real."""

    def test_unknown_data_format_raises_value_error(self):
        """load_real() raises ValueError when data_format is not 'tracklets' or 'csv'."""
        cfg = EvalConfig(data_path="x", data_format="unknown")
        factory = DataFactory(cfg)
        with pytest.raises(ValueError, match="unknown data_format"):
            factory.load_real()


class TestLoadTargetUnknownFormat:
    """eval/data_factory.py:165 — ValueError for unknown format in load_target."""

    def test_unknown_data_format_raises_value_error(self):
        """load_target() raises ValueError when resolved format is neither 'tracklets' nor 'csv'."""
        cfg = EvalConfig(data_path="x", target_data_path="y", data_format="unknown")
        factory = DataFactory(cfg)
        with pytest.raises(ValueError, match="unknown data_format"):
            factory.load_target()


class TestGetGroundTruthCSVFormat:
    """eval/data_factory.py:446 — load_shah_from_csv branch in get_ground_truth."""

    def test_csv_format_uses_load_shah_from_csv(self):
        """When data_format='csv' and ground_truth_path is set, load_shah_from_csv is
        called (Phase 56 D-02: ground_truth_field explicitly overridden to 'id')."""
        cfg = EvalConfig(
            data_path="x.csv",
            data_format="csv",
            ground_truth_path="gt.csv",
            ground_truth_field="id",
        )
        factory = DataFactory(cfg)
        external_ds = {0: zRegPointCloud(pos=torch.zeros(3, 3), label=None, id=torch.arange(3))}
        with patch("eval.data_factory.load_shah_from_csv", return_value=external_ds) as m:
            gt = factory.get_ground_truth({})
        m.assert_called_once_with("gt.csv", device="cpu")
        for i in external_ds:
            assert torch.equal(gt[i], external_ds[i]["id"])


class TestSampleNewPointsWithNoneLabelAndId:
    """eval/data_factory.py:671 — _extend(t) returns None when t is None."""

    def test_sample_new_points_with_none_color_and_id(self):
        """sample_new_points with label=None and id=None still works; _extend returns None."""
        from zreg.data_generation import generate_trajectory
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        # generate_trajectory returns label=None, id=None
        ds = generate_trajectory(n_points=20, n_frames=2, seed=0)
        out = factory.sample_new_points(ds, 5, seed=42)
        for i in ds:
            assert out[i]["pos"].shape[0] == 25
            assert out[i]["label"] is None
            assert out[i]["id"] is None

    def test_sample_new_points_with_2d_color_extends_rows(self):
        """data_factory.py:671 — _extend 2-D branch: label shape (N,3) gets zeros appended."""
        cfg = EvalConfig(data_path="x", augmentation_params={})
        factory = DataFactory(cfg)
        n = 20
        n_extra = 5
        pc = zRegPointCloud(
            pos=torch.randn(n, 3),
            label=torch.rand(n, 3),  # 2-D label
            id=torch.arange(n, dtype=torch.long),
        )
        ds = {0: pc}
        out = factory.sample_new_points(ds, n_extra, seed=42)
        assert out[0]["label"].shape == (n + n_extra, 3)
        # appended rows should be zeros
        assert out[0]["label"][n:].sum().item() == 0.0


# ---------------------------------------------------------------------------
# TestDataPreprocessing — DataFactory._standardize() behavior (Phase 43)
# ---------------------------------------------------------------------------


def _make_dataset(n_frames=3, n_points=50, seed=7):
    """Build a small dataset with non-zero mean/std for standardization testing.

    Uses torch.manual_seed for reproducibility.  pos is drawn from
    N(10, 5^2) so mean ≈ 10, std ≈ 5 — effect of standardization is
    clearly measurable.

    Parameters
    ----------
    n_frames : int
        Number of frames in the returned dict.
    n_points : int
        Points per frame.
    seed : int
        Torch manual seed.

    Returns
    -------
    dict[int, zRegPointCloud]
    """
    torch.manual_seed(seed)
    result = {}
    for i in range(n_frames):
        pos = torch.randn(n_points, 3) * 5 + 10
        label = torch.zeros(n_points, dtype=torch.long)
        id_ = torch.arange(n_points, dtype=torch.long)
        pc = zRegPointCloud(pos=pos, label=label, id=id_)
        pc["fps-idx"] = None
        result[i] = pc
    return result


class TestDataPreprocessing:
    """DataFactory._standardize() behavior and integration tests.

    Covers requirements DATA-02-01 through DATA-02-07:
    - DATA-02-01: z-score standardization (mean≈0, std≈1)
    - DATA-02-02: min-max normalization ([0, 1])
    - DATA-02-03: robust scaling with clipping
    - DATA-02-04: only pos field is scaled; label/id/fps-idx unchanged
    - DATA-02-05: load_target without prior load_real computes own stats
    - DATA-02-06: zero-std stability (eps=1e-8)
    - DATA-02-07: paired mode — source and target share coordinate space
    """

    def _make_factory(self, method="standardize", threshold=3.0, preprocessing=True):
        """Build a DataFactory with the specified preprocessing config."""
        if preprocessing:
            from eval.config import DataPreprocessingConfig
            dp = DataPreprocessingConfig(method=method, robust_outlier_threshold=threshold)
        else:
            dp = None
        cfg = EvalConfig(data_path="x", data_preprocessing=dp)
        from eval.data_factory import DataFactory
        return DataFactory(cfg)

    def test_standardize_returns_near_zero_mean(self):
        """After standardize, per-dimension mean is ≈ 0 (abs < 0.01)."""
        factory = self._make_factory(method="standardize")
        dataset = _make_dataset()
        result = factory._standardize(dataset)
        all_pos = torch.cat([pc["pos"] for pc in result.values()], dim=0)
        assert all_pos.mean(dim=0).abs().max().item() < 0.01

    def test_standardize_returns_near_unit_std(self):
        """After standardize, per-dimension std is within 0.05 of 1.0."""
        factory = self._make_factory(method="standardize")
        dataset = _make_dataset()
        result = factory._standardize(dataset)
        all_pos = torch.cat([pc["pos"] for pc in result.values()], dim=0)
        std = all_pos.std(dim=0)
        assert (std - 1.0).abs().max().item() < 0.05

    def test_normalize_range_zero_to_one(self):
        """After normalize, all pos values are in [0, 1] and extremes hit 0 and 1."""
        factory = self._make_factory(method="normalize")
        dataset = _make_dataset()
        result = factory._standardize(dataset)
        all_pos = torch.cat([pc["pos"] for pc in result.values()], dim=0)
        assert all_pos.min().item() >= 0.0 - 1e-6
        assert all_pos.max().item() <= 1.0 + 1e-6
        # min per dim should be ≈ 0, max per dim ≈ 1
        assert all_pos.min(dim=0).values.abs().max().item() < 1e-4
        assert (all_pos.max(dim=0).values - 1.0).abs().max().item() < 1e-4

    def test_robust_clips_outlier_threshold(self):
        """After robust scaling with threshold=1.0, all abs(pos) <= 1.0 + eps."""
        factory = self._make_factory(method="robust", threshold=1.0)
        dataset = _make_dataset()
        result = factory._standardize(dataset)
        all_pos = torch.cat([pc["pos"] for pc in result.values()], dim=0)
        assert all_pos.abs().max().item() <= 1.0 + 1e-6

    def test_zero_std_no_error(self):
        """Constant pos (zero std) does not raise — eps=1e-8 prevents ZeroDivisionError."""
        factory = self._make_factory(method="standardize")
        # All points identical in all dimensions
        pc = zRegPointCloud(
            pos=torch.ones(10, 3) * 5.0,
            label=torch.zeros(10, dtype=torch.long),
            id=torch.arange(10, dtype=torch.long),
        )
        pc["fps-idx"] = None
        dataset = {0: pc}
        result = factory._standardize(dataset)
        all_pos = torch.cat([pc["pos"] for pc in result.values()], dim=0)
        assert torch.isfinite(all_pos).all()

    def test_standardize_empty_dataset_no_crash(self):
        """Empty dataset returns empty dict without raising RuntimeError (IN-02 / CR-01 regression)."""
        factory = self._make_factory(method="standardize")
        result = factory._standardize({})
        assert result == {}

    def test_standardize_single_point_no_nan(self):
        """Single point across all frames: Bessel-corrected std is NaN but guard replaces it with 0 (IN-03 / WR-01 regression)."""
        factory = self._make_factory(method="standardize")
        pc = zRegPointCloud(
            pos=torch.tensor([[1.0, 2.0, 3.0]]),
            label=torch.zeros(1, dtype=torch.long),
            id=torch.zeros(1, dtype=torch.long),
        )
        pc["fps-idx"] = None
        result = factory._standardize({0: pc})
        assert torch.isfinite(result[0]["pos"]).all()

    def test_preprocessing_none_passthrough(self):
        """When data_preprocessing=None, _standardize returns same dict object (identity)."""
        factory = self._make_factory(preprocessing=False)
        dataset = _make_dataset()
        result = factory._standardize(dataset)
        assert result is dataset

    def test_label_id_fps_idx_unchanged(self):
        """_standardize preserves label, id, and fps-idx tensors unchanged."""
        factory = self._make_factory(method="standardize")
        dataset = _make_dataset(n_frames=1, n_points=20)
        orig_label = dataset[0]["label"].clone()
        orig_id = dataset[0]["id"].clone()
        result = factory._standardize(dataset)
        assert torch.equal(result[0]["label"], orig_label)
        assert torch.equal(result[0]["id"], orig_id)
        assert result[0]["fps-idx"] is None

    def test_stats_stored_after_standardize(self):
        """_standardize with stats=None stores computed stats in _preprocessing_stats."""
        factory = self._make_factory(method="standardize")
        dataset = _make_dataset()
        factory._standardize(dataset)
        stats = factory._preprocessing_stats
        assert isinstance(stats, dict)
        expected_keys = {"mean", "std", "median", "iqr", "min", "max"}
        assert set(stats.keys()) == expected_keys
        for key in expected_keys:
            assert isinstance(stats[key], torch.Tensor)
            assert stats[key].shape == (3,)

    def test_paired_mode_reuses_source_stats(self):
        """_standardize(target, stats=...) does not overwrite _preprocessing_stats cache."""
        factory = self._make_factory(method="standardize")
        source_ds = _make_dataset(seed=7)
        target_ds = _make_dataset(seed=99)
        # First call: compute and cache source stats
        factory._standardize(source_ds)
        source_stats = factory._preprocessing_stats
        # Second call: pass source stats explicitly for target
        factory._standardize(target_ds, stats=source_stats)
        # Cache should still be the same source stats object
        assert factory._preprocessing_stats is source_stats

    def test_robust_default_threshold_no_error(self):
        """Robust scaling with default threshold=3.0 does not raise (DATA-02-03 smoke)."""
        factory = self._make_factory(method="robust", threshold=3.0)
        dataset = _make_dataset()
        result = factory._standardize(dataset)
        all_pos = torch.cat([pc["pos"] for pc in result.values()], dim=0)
        # Clipped to [-3, 3] with default threshold
        assert all_pos.abs().max().item() <= 3.0 + 1e-6

    def test_load_target_without_prior_load_real_computes_own_stats(self):
        """load_target() on a fresh factory computes own stats (DATA-02-05)."""
        cfg = EvalConfig(
            data_path="x.mat",
            target_data_path="y.mat",
            data_format="tracklets",
        )
        from eval.data_factory import DataFactory
        factory = DataFactory(cfg)
        mock_ds = _make_dataset(n_frames=2, n_points=30, seed=11)
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})):
            factory.load_target()
        stats = factory._preprocessing_stats
        assert stats is not None
        # Verify stats were derived from mock_ds, not a sentinel or stale value.
        # mock_ds is drawn from N(10, 5^2) so global mean should be near 10 per dim.
        expected_mean = torch.cat(
            [pc["pos"] for pc in mock_ds.values()], dim=0
        ).float().mean(dim=0)
        assert torch.allclose(stats["mean"], expected_mean, atol=1e-4)

    def test_load_real_and_load_target_share_coordinate_space(self):
        """load_real() + load_target() share source stats so displacement is bounded (DATA-02-07)."""
        cfg = EvalConfig(
            data_path="x.mat",
            target_data_path="y.mat",
            data_format="tracklets",
        )
        from eval.data_factory import DataFactory
        factory = DataFactory(cfg)
        # Source dataset: mean ≈ 10
        source_ds = _make_dataset(n_frames=2, n_points=40, seed=7)
        # Target dataset: mean ≈ 50 (very different range)
        torch.manual_seed(42)
        target_ds = {
            0: zRegPointCloud(
                pos=torch.randn(40, 3) * 5 + 50,
                label=torch.zeros(40, dtype=torch.long),
                id=torch.arange(40, dtype=torch.long),
            )
        }
        target_ds[0]["fps-idx"] = None

        with patch("eval.data_factory.load_data_from_tracklets") as m:
            m.side_effect = [(source_ds, {}), (target_ds, {})]
            src_result = factory.load_real()
            tgt_result = factory.load_target()

        src_pos = torch.cat([pc["pos"] for pc in src_result.values()], dim=0)
        tgt_pos = torch.cat([pc["pos"] for pc in tgt_result.values()], dim=0)
        src_mean = src_pos.mean(dim=0)
        tgt_mean = tgt_pos.mean(dim=0)
        # Both are scaled by source stats; raw mean diff was ~40; after sharing stats it's smaller
        assert (src_mean - tgt_mean).abs().max().item() < 10.0


class TestSubsampleToMax:
    """DataFactory._subsample_to_max coverage (data_factory.py:704-718)."""

    def test_oversized_frames_truncated(self):
        """Frames exceeding max_points_per_frame are truncated to the limit."""
        cfg = EvalConfig(data_path="x", max_points_per_frame=3)
        factory = DataFactory(cfg)
        dataset = {
            0: zRegPointCloud(pos=torch.randn(10, 3), label=None, id=None),
            1: zRegPointCloud(pos=torch.randn(2, 3), label=None, id=None),
        }
        result = factory._subsample_to_max(dataset)
        assert result[0]["pos"].shape[0] == 3
        assert result[1]["pos"].shape[0] == 2


# ---------------------------------------------------------------------------
# TestDataFactoryDeviceGuard — Phase 53 GPU-02 / D-03 availability guard
# ---------------------------------------------------------------------------


class TestDataFactoryDeviceGuard:
    """DataFactory.__init__ raises RuntimeError when requested device is unavailable — D-03."""

    def test_cuda_unavailable_raises_runtime_error(self):
        """device='cuda' + torch.cuda.is_available()=False → RuntimeError."""
        cfg = EvalConfig(data_path="x.mat", device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="torch.cuda.is_available\\(\\) is False"):
                DataFactory(cfg)

    def test_cuda_0_unavailable_raises_runtime_error(self):
        """device='cuda:0' + torch.cuda.is_available()=False → RuntimeError."""
        cfg = EvalConfig(data_path="x.mat", device="cuda:0")
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="torch.cuda.is_available\\(\\) is False"):
                DataFactory(cfg)

    def test_mps_unavailable_raises_runtime_error(self):
        """device='mps' + torch.backends.mps.is_available()=False → RuntimeError."""
        cfg = EvalConfig(data_path="x.mat", device="mps")
        with patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="torch.backends.mps.is_available\\(\\) is False"):
                DataFactory(cfg)

    def test_cpu_skips_guard(self):
        """device='cpu' constructs without error regardless of CUDA availability."""
        cfg = EvalConfig(data_path="x.mat", device="cpu")
        with patch("torch.cuda.is_available", return_value=False):
            factory = DataFactory(cfg)
        assert factory.config.device == "cpu"

    def test_cuda_available_constructs_successfully(self):
        """device='cuda' + torch.cuda.is_available()=True → constructs, factory.config.device=='cuda'."""
        cfg = EvalConfig(data_path="x.mat", device="cuda")
        with patch("torch.cuda.is_available", return_value=True):
            factory = DataFactory(cfg)
        assert factory.config.device == "cuda"


# ---------------------------------------------------------------------------
# Phase 62 (DATA-01 U4-3, DATA-02 U4-5): device-following GT gather,
# outlier correspondence tracking, rotation device contract
# ---------------------------------------------------------------------------

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_labelled_ds(n=20, n_frames=2, seed=1234):
    """Frames with pos ~ U[0,1)^3 and label = arange(n) (label == source index)."""
    g = torch.Generator().manual_seed(seed)
    ds = {}
    for i in range(n_frames):
        pc = zRegPointCloud(pos=torch.rand(n, 3, generator=g), label=torch.arange(n), id=None)
        pc["fps-idx"] = None
        ds[i] = pc
    return ds


def _ds_to(ds, device):
    out = {}
    for k, pc in ds.items():
        new = zRegPointCloud(
            pos=pc["pos"].to(device),
            label=pc["label"].to(device) if pc["label"] is not None else None,
            id=pc["id"].to(device) if pc["id"] is not None else None,
        )
        new["fps-idx"] = None
        out[k] = new
    return out


class TestGetSyntheticGroundTruthDevice:
    """U4-3 / D-01: the GT gather builds every buffer on the source frame's pos device."""

    def _inject(self, factory, ds, corr):
        factory._source_dataset = ds
        factory._synthetic_target = ds
        factory._correspondence_idx = corr

    def test_meta_device_with_correspondence_and_sentinels(self):
        meta = torch.device("meta")
        factory = DataFactory(EvalConfig(data_path="x", device="cpu"))
        pc = zRegPointCloud(
            pos=torch.empty(10, 3, device=meta),
            label=torch.empty(10, dtype=torch.long, device=meta),
            id=None,
        )
        corr = torch.empty(7, dtype=torch.long, device=meta)  # values unknown (may be -1)
        self._inject(factory, {0: pc}, {0: corr})
        gt = factory.get_synthetic_ground_truth()
        assert gt[0].device.type == "meta"
        assert gt[0].dtype == torch.long
        assert gt[0].shape == (7,)

    def test_meta_device_ordinal_fallback(self):
        meta = torch.device("meta")
        factory = DataFactory(EvalConfig(data_path="x", device="cpu"))
        pc = zRegPointCloud(pos=torch.empty(10, 3, device=meta), label=None, id=None)
        corr = torch.empty(4, dtype=torch.long, device=meta)
        self._inject(factory, {0: pc}, {0: corr})
        gt = factory.get_synthetic_ground_truth()
        assert gt[0].device.type == "meta"
        assert gt[0].dtype == torch.long
        assert gt[0].shape == (4,)

    def test_meta_device_without_correspondence_ordinal_fallback(self):
        meta = torch.device("meta")
        factory = DataFactory(EvalConfig(data_path="x", device="cpu"))
        pc = zRegPointCloud(pos=torch.empty(10, 3, device=meta), label=None, id=None)
        self._inject(factory, {0: pc}, None)
        gt = factory.get_synthetic_ground_truth()
        assert gt[0].device.type == "meta"
        assert gt[0].shape == (10,)

    def test_cpu_gather_values_with_sentinel(self):
        factory = DataFactory(EvalConfig(data_path="x"))
        pc = zRegPointCloud(pos=torch.rand(5, 3), label=torch.arange(5) * 10, id=None)
        self._inject(factory, {0: pc}, {0: torch.tensor([2, -1, 0, 4])})
        gt = factory.get_synthetic_ground_truth()
        assert torch.equal(gt[0], torch.tensor([20, -1, 0, 40]))
        assert gt[0].dtype == torch.long

    def test_empty_source_frame_all_sentinel(self):
        factory = DataFactory(EvalConfig(data_path="x"))
        pc = zRegPointCloud(pos=torch.empty(0, 3), label=torch.empty(0, dtype=torch.long), id=None)
        self._inject(factory, {0: pc}, {0: torch.full((3,), -1, dtype=torch.long)})
        gt = factory.get_synthetic_ground_truth()
        assert torch.equal(gt[0], torch.full((3,), -1, dtype=torch.long))

    @_CUDA
    def test_cuda_dropout_new_points_gt_on_cuda(self):
        from zreg.evaluation.label_transfer import compute_f1

        factory = DataFactory(EvalConfig(data_path="x", device="cuda"))
        ds = _ds_to(_make_labelled_ds(), "cuda")
        factory.generate_target(ds, {"dropout_fraction": 0.3, "n_new_points": 3})
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            assert gt[k].device.type == "cuda"
            assert gt[k].dtype == torch.long
            compute_f1(gt[k], gt[k])


class TestAugmentCorrespondence:
    """U4-5 / D-02: outlier injection is tracked in the correspondence map."""

    # Pre-fix (HEAD f5523a0) correspondence for _make_labelled_ds() with
    # {"n_outliers": 5, "dropout_fraction": 0.3}: drop_points retained these
    # positions of the 25-point (20 + 5 outliers) frames. Positions >= 20
    # are outliers and must now map to -1; the retained subset itself (and
    # therefore the RNG stream) must be unchanged.
    _PRE_FIX_RETAINED = {
        0: [0, 1, 2, 3, 6, 8, 9, 10, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        1: [0, 1, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 23, 24],
    }
    # sha256 of the target pos bytes (frames in key order) at HEAD f5523a0.
    _PRE_FIX_POS_SHA256 = "abed7ecf18f4e8f795ba4cdde9c49cbfdb2d70ff7b77da835252bc5aa19b7f93"

    def _assert_gt_points_match(self, ds, tgt, gt):
        for k in ds:
            assert gt[k].shape[0] == tgt[k]["pos"].shape[0]
            for j in torch.nonzero(gt[k] >= 0).flatten().tolist():
                assert torch.equal(tgt[k]["pos"][j], ds[k]["pos"][gt[k][j]])

    def test_outliers_plus_dropout_gt_valid(self):
        n, n_out = 20, 5
        factory = DataFactory(EvalConfig(data_path="x"))
        ds = _make_labelled_ds(n=n)
        tgt = factory.generate_target(ds, {"n_outliers": n_out, "dropout_fraction": 0.3})
        gt = factory.get_synthetic_ground_truth()
        self._assert_gt_points_match(ds, tgt, gt)
        for k in ds:
            expected = torch.tensor(
                [p if p < n else -1 for p in self._PRE_FIX_RETAINED[k]], dtype=torch.long
            )
            assert torch.equal(gt[k], expected)
            assert torch.equal(factory._correspondence_idx[k], expected)

    def test_outliers_plus_dropout_rng_unchanged(self):
        import hashlib

        factory = DataFactory(EvalConfig(data_path="x"))
        tgt = factory.generate_target(
            _make_labelled_ds(), {"n_outliers": 5, "dropout_fraction": 0.3}
        )
        h = hashlib.sha256()
        for k in sorted(tgt):
            h.update(tgt[k]["pos"].contiguous().numpy().tobytes())
        assert h.hexdigest() == self._PRE_FIX_POS_SHA256

    def test_outlier_only_gt_length_and_sentinels(self):
        n, n_out = 20, 5
        factory = DataFactory(EvalConfig(data_path="x"))
        ds = _make_labelled_ds(n=n)
        tgt = factory.generate_target(ds, {"n_outliers": n_out})
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            assert gt[k].shape[0] == n + n_out == tgt[k]["pos"].shape[0]
            assert torch.equal(gt[k][:n], torch.arange(n))
            assert torch.all(gt[k][n:] == -1)
        self._assert_gt_points_match(ds, tgt, gt)

    def test_outliers_plus_new_points(self):
        n, n_out, n_new = 20, 5, 4
        factory = DataFactory(EvalConfig(data_path="x"))
        ds = _make_labelled_ds(n=n)
        tgt = factory.generate_target(ds, {"n_outliers": n_out, "n_new_points": n_new})
        gt = factory.get_synthetic_ground_truth()
        for k in ds:
            assert gt[k].shape[0] == n + n_out + n_new
            assert torch.all(gt[k][n:] == -1)
        self._assert_gt_points_match(ds, tgt, gt)

    def test_outliers_dropout_new_points(self):
        factory = DataFactory(EvalConfig(data_path="x"))
        ds = _make_labelled_ds(n=30)
        tgt = factory.generate_target(
            ds, {"n_outliers": 6, "dropout_fraction": 0.4, "n_new_points": 3}
        )
        gt = factory.get_synthetic_ground_truth()
        self._assert_gt_points_match(ds, tgt, gt)
        for k in ds:
            assert torch.all(gt[k][-3:] == -1)

    def test_direct_augment_does_not_compose_stale_map(self):
        """62-REVIEW WR-07: augment() after a dropout generate_target starts a fresh map."""
        factory = DataFactory(EvalConfig(data_path="x"))
        ds = _make_labelled_ds(n=30)
        factory.generate_target(ds, {"dropout_fraction": 0.4})
        stale = factory._correspondence_idx
        assert stale is not None and stale[next(iter(ds))].shape[0] < 30
        factory.config.augmentation_params = {"n_outliers": 4}
        out = factory.augment(ds)
        for k in ds:
            corr = factory._correspondence_idx[k]
            assert corr.shape[0] == out[k]["pos"].shape[0] == 34
            assert torch.equal(corr[:30], torch.arange(30))
            assert torch.all(corr[30:] == -1)

    def test_direct_augment_without_count_change_clears_stale_map(self):
        """62-REVIEW WR-07: a count-preserving augment leaves no stale map behind."""
        factory = DataFactory(EvalConfig(data_path="x"))
        ds = _make_labelled_ds(n=30)
        factory.generate_target(ds, {"dropout_fraction": 0.4})
        factory.config.augmentation_params = {"sigma": 0.01}
        factory.augment(ds)
        assert factory._correspondence_idx is None

    def test_sample_new_points_map_unchanged(self):
        """sample_new_points still appends n_extra -1 sentinels to an identity map."""
        factory = DataFactory(EvalConfig(data_path="x"))
        ds = _make_labelled_ds(n=6)
        factory._correspondence_idx = None
        factory.sample_new_points(ds, 2, seed=0)
        for k in ds:
            assert torch.equal(
                factory._correspondence_idx[k], torch.tensor([0, 1, 2, 3, 4, 5, -1, -1])
            )


class TestAugmentRotationDevice:
    """Review cycle 2 MEDIUM: the rotation step follows the input device.

    augment step 4 builds R on the CPU; _apply_matrix (zreg.data_generation.
    transforms) casts it to each frame's pos dtype/device. These tests pin
    that contract.
    """

    _ROT = {"rotation_deg": 30.0, "rotation_axis": [0.0, 0.0, 1.0]}

    def test_meta_rotation_stays_on_meta(self):
        meta = torch.device("meta")
        factory = DataFactory(EvalConfig(data_path="x", augmentation_params=dict(self._ROT)))
        ds = {}
        for i in range(2):
            pc = zRegPointCloud(
                pos=torch.empty(10, 3, device=meta),
                label=torch.empty(10, dtype=torch.long, device=meta),
                id=None,
            )
            pc["fps-idx"] = None
            ds[i] = pc
        out = factory.augment(ds)
        for k in ds:
            assert out[k]["pos"].device.type == "meta"
            assert out[k]["pos"].dtype == torch.float32

    @_CUDA
    def test_cuda_rotation_matches_cpu(self):
        ds_cpu = _make_labelled_ds()
        ds_cuda = _ds_to(ds_cpu, "cuda")
        out_cpu = DataFactory(
            EvalConfig(data_path="x", augmentation_params=dict(self._ROT))
        ).augment(ds_cpu)
        out_cuda = DataFactory(
            EvalConfig(data_path="x", device="cuda", augmentation_params=dict(self._ROT))
        ).augment(ds_cuda)
        for k in ds_cpu:
            assert out_cuda[k]["pos"].device.type == "cuda"
            assert torch.allclose(out_cuda[k]["pos"].cpu(), out_cpu[k]["pos"], atol=1e-5)

    @_CUDA
    def test_cuda_rotation_dropout_new_points_gt(self):
        ds_cpu = _make_labelled_ds()
        ds = _ds_to(ds_cpu, "cuda")
        factory = DataFactory(EvalConfig(data_path="x", device="cuda"))
        tgt = factory.generate_target(
            ds, {"rotation_deg": 30.0, "dropout_fraction": 0.3, "n_new_points": 3}
        )
        gt = factory.get_synthetic_ground_truth()
        rotated = DataFactory(
            EvalConfig(data_path="x", device="cuda", augmentation_params=dict(self._ROT))
        ).augment(ds)
        for k in ds:
            assert tgt[k]["pos"].device.type == "cuda"
            assert gt[k].device.type == "cuda"
            for j in torch.nonzero(gt[k] >= 0).flatten().tolist():
                assert torch.allclose(tgt[k]["pos"][j], rotated[k]["pos"][gt[k][j]], atol=1e-5)

    def test_cpu_rotation_dropout_new_points_gt(self):
        """CPU twin of the CUDA test above (runs everywhere)."""
        ds = _make_labelled_ds()
        factory = DataFactory(EvalConfig(data_path="x"))
        tgt = factory.generate_target(
            ds, {"rotation_deg": 30.0, "dropout_fraction": 0.3, "n_new_points": 3}
        )
        gt = factory.get_synthetic_ground_truth()
        rotated = DataFactory(
            EvalConfig(data_path="x", augmentation_params={"rotation_deg": 30.0})
        ).augment(ds)
        for k in ds:
            assert gt[k].shape[0] == tgt[k]["pos"].shape[0]
            for j in torch.nonzero(gt[k] >= 0).flatten().tolist():
                assert torch.allclose(tgt[k]["pos"][j], rotated[k]["pos"][gt[k][j]], atol=1e-5)


class TestGenerateSubsamplePairLabelGeneration:
    """U4-8: the synthesize path honours config.label_generation (D-13 precedence).

    explicit transform_spec n_labels/n_classes > config.label_generation >
    6-class fallback, mirroring generate_training_triple.
    """

    _SPEC = {"synthesize": True, "seed": 3, "n_points": 150, "shape": "ball"}

    def _base_labels(self, factory):
        return factory._source_dataset[0]["label"]

    def _expected(self, factory, **kwargs):
        from zreg.data_generation import generate_labels

        base = {0: zRegPointCloud(pos=factory._source_dataset[0]["pos"])}
        return generate_labels(base, seed=3, **kwargs)[0]["label"]

    def test_config_label_generation_applies(self):
        from eval.config import LabelGenerationConfig

        cfg = EvalConfig(data_path="x", label_generation=LabelGenerationConfig(n_labels=2))
        factory = DataFactory(cfg)
        src, tgt = factory.generate_subsample_pair(None, dict(self._SPEC))
        for view in (src, tgt):
            assert set(view[0]["label"].unique().tolist()) <= {0, 1}
        assert torch.equal(
            self._base_labels(factory),
            self._expected(factory, n_labels=2, mode="deterministic"),
        )

    @pytest.mark.parametrize("key,value", [("n_labels", 4), ("n_classes", 3)])
    def test_explicit_spec_value_wins(self, key, value):
        from eval.config import LabelGenerationConfig

        cfg = EvalConfig(data_path="x", label_generation=LabelGenerationConfig(n_labels=2))
        factory = DataFactory(cfg)
        factory.generate_subsample_pair(None, {**self._SPEC, key: value})
        assert torch.equal(self._base_labels(factory), self._expected(factory, n_labels=value))
        assert int(self._base_labels(factory).max()) >= 2

    def test_fallback_six_classes(self):
        factory = DataFactory(EvalConfig(data_path="x"))
        factory.generate_subsample_pair(None, dict(self._SPEC))
        assert torch.equal(self._base_labels(factory), self._expected(factory, n_labels=6))
