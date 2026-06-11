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
                color=None,
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
                color=None,
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
        single_pt = {0: zRegPointCloud(pos=torch.tensor([[1.0, 0.0, 0.0]]), color=None, id=torch.arange(1))}
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
                color=torch.zeros(100, dtype=torch.long),
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
                color=torch.zeros(100, dtype=torch.long),
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
