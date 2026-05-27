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
from zreg.dataset import zRegPointCloud

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
        assert factory.config is cfg


# ---------------------------------------------------------------------------
# TestLoadReal — D-09 caching + dispatch by data_format (Plan 17-02)
# ---------------------------------------------------------------------------


class TestLoadReal:
    """D-09 caching + dispatch by data_format."""

    def _make_mock_ds(self):
        """Build a minimal mock dataset for patching."""
        return {0: zRegPointCloud(pos=torch.zeros(3, 3), color=None, id=torch.arange(3))}

    def test_dispatches_tracklets(self):
        """data_format='tracklets' calls load_data_from_tracklets with device='cpu'."""
        cfg = EvalConfig(data_path="x.mat", data_format="tracklets")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
            result = factory.load_real()
        m.assert_called_once_with("x.mat", device="cpu")
        assert result is mock_ds

    def test_dispatches_csv(self):
        """data_format='csv' calls load_shah_from_csv with explicit device='cpu' (Pitfall 5)."""
        cfg = EvalConfig(data_path="x.csv", data_format="csv")
        factory = DataFactory(cfg)
        mock_ds = self._make_mock_ds()
        with patch("eval.data_factory.load_shah_from_csv", return_value=mock_ds) as m:
            factory.load_real()
        m.assert_called_once_with("x.csv", device="cpu")

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
        return {i: zRegPointCloud(pos=torch.zeros(2, 3), color=None, id=torch.arange(2)) for i in range(n)}

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
        ds = {0: zRegPointCloud(pos=torch.zeros(2, 3), color=None, id=torch.arange(2))}
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
        """D-10, D-11: returns {i: ds[i]['id']} when ground_truth_path is None."""
        ds = {
            i: zRegPointCloud(pos=torch.zeros(2, 3), color=None, id=torch.tensor([i, i + 100]))
            for i in range(3)
        }
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        gt = factory.get_ground_truth(ds)
        assert set(gt.keys()) == set(ds.keys())
        for i in ds:
            assert torch.equal(gt[i], ds[i]["id"])

    def test_external_gt_path(self):
        """D-10: when ground_truth_path is set, loads via same loader as data_format."""
        cfg = EvalConfig(data_path="x.mat", data_format="tracklets", ground_truth_path="gt.mat")
        factory = DataFactory(cfg)
        external_ds = {0: zRegPointCloud(pos=torch.zeros(3, 3), color=None, id=torch.arange(3))}
        with patch("eval.data_factory.load_data_from_tracklets", return_value=(external_ds, {})) as m:
            gt = factory.get_ground_truth({})  # in-memory ds is irrelevant when external path set
        m.assert_called_once_with("gt.mat", device="cpu")
        for i in external_ds:
            assert torch.equal(gt[i], external_ds[i]["id"])
