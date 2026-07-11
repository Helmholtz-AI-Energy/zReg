"""Wave-0 tests for Phase 46 Plan 03's seed-driven training-triple pipeline.

Mandated by 45-DESIGN.md's "Downstream Wave-0 Test Obligations" and
46-RESEARCH.md's "Validation Architecture" Phase-Requirements -> Test Map.
Covers, across seven test classes:

- TestAugmentSeedThreading      — Task 1 (Pitfall 3): augment() forwards a
                                   caller-supplied "augment_seed" to its four
                                   internal stochastic sub-calls; omitting the
                                   key preserves exact legacy (seed=42) behavior.
- TestPointCountRegime          — D-01: source/target clouds have 100-300 points.
- TestSeedReproducibilityAndVariety — D-02: same seed -> bitwise-equal results;
                                   different seeds -> non-equal source positions.
- TestSeedSplit                 — D-03: split_seeds() disjoint train/val ranges.
- TestLabelVsIdDiscipline        — 45-DESIGN.md carry-forward: labels come only
                                   from pc["label"], never pc["id"].
- TestNoFalseCaching             — Pitfall 2: no singleton-cache reuse across seeds.
- TestGeometryCoverage           — D-04 discretion: both ball and bowl geometries
                                   are exercised across a seed range.

Task lifecycle note (per plan review): Task 1 introduces this file with its own
TestAugmentSeedThreading class (RED, then GREEN after augment() is extended).
Task 2 extends the same file with its own classes for split_seeds()/
generate_training_triple()/generate_training_set() (RED, then GREEN). Task 3
finalizes the file by adding the remaining mandated classes not yet covered.
"""

# zreg.dataset/zreg.generators MUST precede import torch on macOS ARM to avoid
# a libomp SIGABRT (enforced in tests/conftest.py:20-24; matches the convention
# in tests/test_data_factory.py).
from zreg.dataset import zRegPointCloud
from zreg.generators import add_gaussian_noise, generate_trajectory

import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory, split_seeds


# ---------------------------------------------------------------------------
# TestAugmentSeedThreading — Task 1 (Pitfall 3)
# ---------------------------------------------------------------------------


class TestAugmentSeedThreading:
    """augment() forwards a caller-supplied "augment_seed" to its stochastic
    sub-calls; omitting the key reproduces exact legacy (seed=42) behavior."""

    def _baseline(self):
        """2-frame, 50-point synthetic trajectory (no labels needed here)."""
        return generate_trajectory(n_points=50, n_frames=2, seed=0)

    def test_sigma_augment_seed_varies_noise_pattern(self):
        """Different augment_seed values produce non-bitwise-equal noise patterns
        (not just magnitude-scaled) for the same sigma."""
        traj = self._baseline()
        cfg_a = EvalConfig(data_path="x", augmentation_params={"sigma": 1.0, "augment_seed": 7})
        cfg_b = EvalConfig(data_path="x", augmentation_params={"sigma": 1.0, "augment_seed": 8})
        out_a = DataFactory(cfg_a).augment(traj)
        out_b = DataFactory(cfg_b).augment(traj)
        assert not torch.equal(out_a[0]["pos"], out_b[0]["pos"])

    def test_dropout_augment_seed_varies_kept_points(self):
        """Different augment_seed values drop DIFFERENT points for the same
        dropout_fraction (kept-point sets differ)."""
        traj = self._baseline()
        cfg_a = EvalConfig(
            data_path="x", augmentation_params={"dropout_fraction": 0.3, "augment_seed": 7}
        )
        cfg_b = EvalConfig(
            data_path="x", augmentation_params={"dropout_fraction": 0.3, "augment_seed": 8}
        )
        out_a = DataFactory(cfg_a).augment(traj)
        out_b = DataFactory(cfg_b).augment(traj)
        # Same keep-count (deterministic from fraction) but different point sets.
        assert out_a[0]["pos"].shape == out_b[0]["pos"].shape
        assert not torch.equal(out_a[0]["pos"], out_b[0]["pos"])

    def test_omitting_augment_seed_reproduces_legacy_seed_42(self):
        """augmentation_params without "augment_seed" reproduces the exact
        legacy result (seed=42 default), matching add_gaussian_noise directly."""
        traj = self._baseline()
        cfg = EvalConfig(data_path="x", augmentation_params={"sigma": 1.0})
        out = DataFactory(cfg).augment(traj)
        reference = add_gaussian_noise(traj, sigma=1.0, seed=42)
        for i in traj:
            assert torch.equal(out[i]["pos"], reference[i]["pos"])

    def test_augment_seed_alone_is_a_noop(self):
        """A params dict of only {"augment_seed": 5} triggers no dispatch step
        and returns the input dataset unchanged (by identity)."""
        traj = self._baseline()
        cfg = EvalConfig(data_path="x", augmentation_params={"augment_seed": 5})
        out = DataFactory(cfg).augment(traj)
        assert out is traj


# ---------------------------------------------------------------------------
# TestSeedSplit — Task 2 / D-03
# ---------------------------------------------------------------------------


class TestSeedSplit:
    """split_seeds() returns disjoint train/val integer ranges (D-03)."""

    def test_default_base_seed(self):
        """split_seeds(200, 50, base_seed=0) returns (range(0,200), range(200,250))."""
        train, val = split_seeds(200, 50, base_seed=0)
        assert train == range(0, 200)
        assert val == range(200, 250)

    def test_disjoint(self):
        """Train and val seed sets never overlap."""
        train, val = split_seeds(200, 50, base_seed=0)
        assert set(train).isdisjoint(set(val))
        assert min(val) >= 200

    def test_rejects_non_positive_n_train(self):
        """n_train < 1 raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            split_seeds(0, 10)

    def test_rejects_negative_n_val(self):
        """n_val < 0 raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            split_seeds(10, -1)


# ---------------------------------------------------------------------------
# TestPointCountRegime — Task 2 / D-01
# ---------------------------------------------------------------------------


class TestPointCountRegime:
    """Generated source and target clouds have 100-300 points/frame."""

    def test_source_and_target_point_counts(self):
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        triple = factory.generate_training_triple(seed=0)
        assert 100 <= triple.source_cloud["pos"].shape[0] <= 300
        assert 100 <= triple.target_cloud["pos"].shape[0] <= 300


# ---------------------------------------------------------------------------
# TestSeedReproducibilityAndVariety — Task 2 / D-02
# ---------------------------------------------------------------------------


class TestSeedReproducibilityAndVariety:
    """Same seed -> bitwise-equal results; different seeds -> non-equal source pos."""

    def test_same_seed_is_reproducible(self):
        cfg = EvalConfig(data_path="x")
        t1 = DataFactory(cfg).generate_training_triple(seed=1)
        t2 = DataFactory(cfg).generate_training_triple(seed=1)
        assert torch.equal(t1.source_cloud["pos"], t2.source_cloud["pos"])
        assert torch.equal(t1.target_cloud["pos"], t2.target_cloud["pos"])

    def test_different_seeds_differ(self):
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        t1 = factory.generate_training_triple(seed=1)
        t2 = factory.generate_training_triple(seed=2)
        assert not torch.equal(t1.source_cloud["pos"], t2.source_cloud["pos"])

    def test_generate_training_set_batches_seeds(self):
        """generate_training_set(range(0,4)) returns 4 triples with seeds [0,1,2,3]."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        triples = factory.generate_training_set(range(0, 4))
        assert [t.seed for t in triples] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# TestLabelVsIdDiscipline — 45-DESIGN.md carry-forward
# ---------------------------------------------------------------------------


class TestLabelVsIdDiscipline:
    """Training-triple class targets come only from pc["label"], never pc["id"]."""

    def test_source_id_is_none_and_label_is_long(self):
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        triple = factory.generate_training_triple(seed=0, n_classes=6)
        assert triple.source_cloud["id"] is None
        assert triple.source_labels.dtype == torch.long
        assert triple.source_labels.min() >= 0
        assert triple.source_labels.max() < 6

    def test_target_labels_in_range_no_sentinel(self):
        """target labels are non-None, torch.long, in [0, n_classes) — never -1
        (n_new_points is excluded from the transform_spec, so no sentinel-filled
        points ever appear in the target)."""
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        triple = factory.generate_training_triple(seed=0, n_classes=6)
        assert triple.target_labels is not None
        assert triple.target_labels.dtype == torch.long
        assert triple.target_labels.min() >= 0
        assert triple.target_labels.max() < 6


# ---------------------------------------------------------------------------
# TestNoFalseCaching — Task 2 / Pitfall 2
# ---------------------------------------------------------------------------


class TestNoFalseCaching:
    """generate_training_triple() does not reuse the _synthetic_dataset singleton
    cache; N distinct seeds produce N distinct results."""

    def test_five_seeds_five_distinct_results(self):
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        triples = [factory.generate_training_triple(s) for s in range(5)]
        positions = [t.source_cloud["pos"] for t in triples]
        assert all(
            not torch.equal(positions[i], positions[j])
            for i in range(5)
            for j in range(i + 1, 5)
        )
        assert factory._synthetic_dataset is None


# ---------------------------------------------------------------------------
# TestGeometryCoverage — Task 2/3 / D-04 discretion
# ---------------------------------------------------------------------------


class TestGeometryCoverage:
    """Both ball and bowl geometries are exercised over seeds 0..9 (even=ball,
    odd=bowl per the documented rule); geometric containment is verified."""

    def test_both_geometries_appear(self):
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        triples = [factory.generate_training_triple(s) for s in range(10)]
        # Even seeds -> ball (all points within radius of origin);
        # odd seeds -> bowl (all points at z <= 0, per the documented rule).
        saw_ball = False
        saw_bowl = False
        for s, t in zip(range(10), triples):
            pos = t.source_cloud["pos"]
            if s % 2 == 0:
                saw_ball = True
                radius = pos.norm(dim=1).max()
                assert radius <= 1.0 + 1e-4
            else:
                saw_bowl = True
                assert torch.all(pos[:, 2] <= 1e-4)
        assert saw_ball and saw_bowl
