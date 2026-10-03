"""Tests for zreg.generators module.

Four test classes cover all seven public symbols across Plans 01 and 02:
- TestGenerateTrajectory  — generate_trajectory factory
- TestTransformWrappers   — apply_rigid, apply_affine wrappers
- TestCorruptionWrappers  — add_gaussian_noise, add_outliers wrappers
- TestLabelUtilities      — generate_labels, remove_labels utilities

Each class verifies: return shape/dtype contracts (D-07), immutability
(D-03), seed reproducibility, ValueError raises, and downstream
compatibility with zreg.metrics.compute_f1 (D-07 dtype contract).
"""

import copy

import pytest
import torch

from zreg.data_generation import (
    generate_trajectory,
    apply_rigid,
    apply_affine,
    add_gaussian_noise,
    add_outliers,
    generate_labels,
    remove_labels,
    LabelComponentSpec,
    LabelSpec,
)
from zreg.data_generation.generators import sample_ball, sample_bowl
from zreg.core.dataset import zRegPointCloud
from zreg.core.transforms import RigidTransformation, AffineTransformation
from zreg.evaluation.label_transfer import compute_f1


# ---------------------------------------------------------------------------
# TestGenerateTrajectory
# ---------------------------------------------------------------------------


class TestGenerateTrajectory:
    """Tests for the generate_trajectory factory function."""

    def test_returns_dict_with_correct_keys(self):
        """3-frame trajectory has keys {0, 1, 2} and all values are zRegPointCloud."""
        traj = generate_trajectory(n_points=10, n_frames=3, seed=0)
        assert set(traj.keys()) == {0, 1, 2}
        for pc in traj.values():
            assert isinstance(pc, zRegPointCloud)

    def test_pos_shape(self):
        """Every frame's pos has shape (n_points, 3)."""
        n_points = 15
        traj = generate_trajectory(n_points=n_points, n_frames=4, seed=1)
        for pc in traj.values():
            assert pc["pos"].shape == (n_points, 3)

    def test_color_id_unset(self):
        """Every frame's label and id is None (factory leaves them unset)."""
        traj = generate_trajectory(n_points=10, n_frames=3, seed=2)
        for pc in traj.values():
            assert pc["label"] is None
            assert pc["id"] is None

    def test_seed_reproducibility(self):
        """Same seed produces torch.equal pos across two calls."""
        traj_a = generate_trajectory(n_points=20, n_frames=3, seed=42)
        traj_b = generate_trajectory(n_points=20, n_frames=3, seed=42)
        for i in traj_a:
            assert torch.equal(traj_a[i]["pos"], traj_b[i]["pos"])

    def test_seed_none_does_not_crash(self):
        """generate_trajectory(10, 3, seed=None) returns without raising."""
        traj = generate_trajectory(10, 3, seed=None)
        assert len(traj) == 3

    def test_seed_none_is_nondeterministic(self):
        """Two consecutive seed=None calls produce DIFFERENT pos tensors."""
        traj_a = generate_trajectory(50, 2, seed=None)
        traj_b = generate_trajectory(50, 2, seed=None)
        # It would be astronomically unlikely for these to be equal
        assert not all(torch.equal(traj_a[i]["pos"], traj_b[i]["pos"]) for i in traj_a)

    def test_invalid_n_points(self):
        """generate_trajectory(0, 3) raises ValueError matching 'n_points'."""
        with pytest.raises(ValueError, match="n_points"):
            generate_trajectory(0, 3)

    def test_invalid_n_frames(self):
        """generate_trajectory(10, 0) raises ValueError matching 'n_frames'."""
        with pytest.raises(ValueError, match="n_frames"):
            generate_trajectory(10, 0)


# ---------------------------------------------------------------------------
# TestTransformWrappers
# ---------------------------------------------------------------------------


class TestTransformWrappers:
    """Tests for the apply_rigid and apply_affine wrapper functions."""

    def test_apply_rigid_identity_preserves_pos(self):
        """Identity RigidTransformation() leaves pos unchanged (atol=1e-5)."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=10)
        result = apply_rigid(traj, RigidTransformation())
        for i in traj:
            assert torch.allclose(result[i]["pos"], traj[i]["pos"], atol=1e-5)

    def test_apply_affine_identity_preserves_pos(self):
        """AffineTransformation(t=zeros(3)) leaves pos unchanged (atol=1e-5)."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=11)
        # AffineTransformation() defaults to the identity (t = zeros); the
        # explicit t=torch.zeros(3) keeps the intent visible.
        result = apply_affine(traj, AffineTransformation(t=torch.zeros(3)))
        for i in traj:
            assert torch.allclose(result[i]["pos"], traj[i]["pos"], atol=1e-5)

    def test_apply_rigid_returns_new_dict(self):
        """Result dict is not the same object; pos tensors have distinct data_ptr."""
        traj = generate_trajectory(n_points=10, n_frames=2, seed=12)
        result = apply_rigid(traj, RigidTransformation())
        assert result is not traj
        for i in traj:
            assert result[i]["pos"].data_ptr() != traj[i]["pos"].data_ptr()

    def test_apply_rigid_does_not_mutate_input(self):
        """Deepcopy snapshot of traj remains torch.equal to traj after apply_rigid call."""
        traj = generate_trajectory(n_points=15, n_frames=3, seed=13)
        snapshot = copy.deepcopy(traj)
        apply_rigid(traj, RigidTransformation())
        assert all(torch.equal(traj[i]["pos"], snapshot[i]["pos"]) for i in traj)

    def test_apply_affine_does_not_mutate_input(self):
        """Deepcopy snapshot of traj remains torch.equal to traj after apply_affine call."""
        traj = generate_trajectory(n_points=15, n_frames=3, seed=14)
        snapshot = copy.deepcopy(traj)
        apply_affine(traj, AffineTransformation(t=torch.zeros(3)))
        assert all(torch.equal(traj[i]["pos"], snapshot[i]["pos"]) for i in traj)

    def test_apply_rigid_preserves_other_fields(self, trajectory_data):
        """Output frames' label and id are torch.equal to input's when fields are set."""
        result = apply_rigid(trajectory_data, RigidTransformation())
        for i in trajectory_data:
            assert torch.equal(result[i]["label"], trajectory_data[i]["label"])
            assert torch.equal(result[i]["id"], trajectory_data[i]["id"])

    def test_apply_rigid_translation_applied(self):
        """Non-identity RigidTransformation(t=[1,2,3]) shifts every output pos by (1,2,3)."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=15)
        t = torch.tensor([1.0, 2.0, 3.0])
        result = apply_rigid(traj, RigidTransformation(t=t))
        for i in traj:
            diff = result[i]["pos"] - traj[i]["pos"]
            expected = t.expand_as(traj[i]["pos"])
            assert torch.allclose(diff, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# TestCorruptionWrappers
# ---------------------------------------------------------------------------


class TestCorruptionWrappers:
    """Tests for the add_gaussian_noise and add_outliers wrapper functions."""

    def test_add_gaussian_noise_sigma_zero_is_identity(self):
        """add_gaussian_noise(traj, sigma=0.0) yields pos torch.equal to input's."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=20)
        result = add_gaussian_noise(traj, sigma=0.0)
        for i in traj:
            assert torch.equal(result[i]["pos"], traj[i]["pos"])

    def test_add_gaussian_noise_changes_pos(self):
        """add_gaussian_noise(sigma=0.1) produces pos NOT torch.equal, same shape."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=21)
        result = add_gaussian_noise(traj, sigma=0.1, seed=42)
        for i in traj:
            assert result[i]["pos"].shape == traj[i]["pos"].shape
            assert not torch.equal(result[i]["pos"], traj[i]["pos"])

    def test_add_gaussian_noise_reproducible(self):
        """Same seed produces torch.equal output across two calls."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=22)
        result_a = add_gaussian_noise(traj, sigma=0.1, seed=99)
        result_b = add_gaussian_noise(traj, sigma=0.1, seed=99)
        for i in traj:
            assert torch.equal(result_a[i]["pos"], result_b[i]["pos"])

    def test_add_gaussian_noise_does_not_mutate_input(self):
        """Deepcopy snapshot remains torch.equal to input after add_gaussian_noise."""
        traj = generate_trajectory(n_points=15, n_frames=3, seed=23)
        snapshot = copy.deepcopy(traj)
        add_gaussian_noise(traj, sigma=0.1, seed=42)
        assert all(torch.equal(traj[i]["pos"], snapshot[i]["pos"]) for i in traj)

    def test_add_gaussian_noise_invalid_sigma(self):
        """sigma=-0.1 raises ValueError matching 'sigma'."""
        traj = generate_trajectory(n_points=10, n_frames=2, seed=24)
        with pytest.raises(ValueError, match="sigma"):
            add_gaussian_noise(traj, sigma=-0.1)

    def test_add_outliers_increases_pos_length(self, trajectory_data):
        """Every output frame's pos.shape[0] == input.shape[0] + n_outliers."""
        n_outliers = 5
        result = add_outliers(trajectory_data, n_outliers=n_outliers, seed=42)
        for i in trajectory_data:
            expected = trajectory_data[i]["pos"].shape[0] + n_outliers
            assert result[i]["pos"].shape[0] == expected

    def test_add_outliers_extends_id_with_sentinel(self, trajectory_data):
        """When input has id of length N, output id has length N+5 with last 5 == -1."""
        n_outliers = 5
        result = add_outliers(trajectory_data, n_outliers=n_outliers, seed=42)
        for i in trajectory_data:
            n = trajectory_data[i]["id"].shape[0]
            assert result[i]["id"].shape[0] == n + n_outliers
            assert torch.all(result[i]["id"][-n_outliers:] == -1)

    def test_add_outliers_extends_2d_color(self, trajectory_data):
        """When input has 2-D label of shape (N, 3), output has (N+5, 3) with zero appended rows."""
        n_outliers = 5
        result = add_outliers(trajectory_data, n_outliers=n_outliers, seed=42)
        for i in trajectory_data:
            n = trajectory_data[i]["label"].shape[0]
            assert result[i]["label"].shape == (n + n_outliers, 3)
            assert torch.allclose(
                result[i]["label"][-n_outliers:].float(),
                torch.zeros(n_outliers, 3),
            )

    def test_add_outliers_does_not_mutate_input(self, trajectory_data):
        """Deepcopy snapshot remains torch.equal to input after add_outliers."""
        snapshot = copy.deepcopy(trajectory_data)
        add_outliers(trajectory_data, n_outliers=5, seed=42)
        for i in trajectory_data:
            assert torch.equal(trajectory_data[i]["pos"], snapshot[i]["pos"])

    def test_add_outliers_invalid_n_outliers(self):
        """n_outliers=-1 raises ValueError matching 'n_outliers'."""
        traj = generate_trajectory(n_points=10, n_frames=2, seed=25)
        with pytest.raises(ValueError, match="n_outliers"):
            add_outliers(traj, n_outliers=-1)

    def test_add_outliers_negative_scale_raises(self):
        """corruption.py:105 — scale < 0 raises ValueError matching 'scale'."""
        traj = generate_trajectory(n_points=10, n_frames=2, seed=26)
        with pytest.raises(ValueError, match="scale"):
            add_outliers(traj, n_outliers=2, scale=-1.0)

    def test_add_outliers_zero_is_noop(self):
        """n_outliers=0 returns frames with the same pos.shape[0] as input."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=26)
        result = add_outliers(traj, n_outliers=0, seed=42)
        for i in traj:
            assert result[i]["pos"].shape[0] == traj[i]["pos"].shape[0]
            assert torch.equal(result[i]["pos"], traj[i]["pos"])


# ---------------------------------------------------------------------------
# TestLabelUtilities
# ---------------------------------------------------------------------------


class TestLabelUtilities:
    """Tests for the generate_labels and remove_labels utility functions."""

    def test_generate_labels_dtype_and_shape(self):
        """Every frame's pc['label'] is torch.long, shape (N,)."""
        n_points = 30
        traj = generate_trajectory(n_points=n_points, n_frames=4, seed=30)
        labelled = generate_labels(traj, n_labels=4, seed=42)
        for pc in labelled.values():
            assert pc["label"].dtype == torch.long
            assert pc["label"].shape == (n_points,)

    def test_generate_labels_value_range(self):
        """Every frame's pc['label'].min() >= 0 and .max() < n_labels."""
        n_labels = 4
        traj = generate_trajectory(n_points=50, n_frames=3, seed=31)
        labelled = generate_labels(traj, n_labels=n_labels, seed=42)
        for pc in labelled.values():
            assert pc["label"].min().item() >= 0
            assert pc["label"].max().item() < n_labels

    def test_generate_labels_reproducible(self):
        """Same seed produces torch.equal label output across two calls."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=32)
        labelled_a = generate_labels(traj, n_labels=3, seed=55)
        labelled_b = generate_labels(traj, n_labels=3, seed=55)
        for i in traj:
            assert torch.equal(labelled_a[i]["label"], labelled_b[i]["label"])

    def test_generate_labels_does_not_mutate_input(self):
        """Input frames still have label is None after generate_labels call."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=33)
        # Verify input has label=None before the call
        for pc in traj.values():
            assert pc["label"] is None
        generate_labels(traj, n_labels=3, seed=42)
        # Input must still have label=None after the call
        for pc in traj.values():
            assert pc["label"] is None

    def test_generate_labels_invalid_n_labels(self):
        """n_labels=0 raises ValueError matching 'n_labels'."""
        traj = generate_trajectory(n_points=10, n_frames=2, seed=34)
        with pytest.raises(ValueError, match="n_labels"):
            generate_labels(traj, n_labels=0)

    def test_generate_labels_compatible_with_compute_f1(self):
        """compute_f1(labelled[0]['label'], labelled[0]['label']) returns 1.0 (D-07 dtype contract)."""
        traj = generate_trajectory(n_points=50, n_frames=3, seed=35)
        labelled = generate_labels(traj, n_labels=4, seed=42)
        # Perfect prediction: y_true == y_pred should yield F1 == 1.0
        score = compute_f1(
            y_true=labelled[0]["label"],
            y_pred=labelled[0]["label"],
        )
        assert abs(score - 1.0) < 1e-6

    def test_remove_labels_sets_color_none(self):
        """Every frame's pc['label'] is None in the output."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=36)
        labelled = generate_labels(traj, n_labels=3, seed=42)
        # Confirm labels are set
        for pc in labelled.values():
            assert pc["label"] is not None
        unlabelled = remove_labels(labelled)
        for pc in unlabelled.values():
            assert pc["label"] is None

    def test_remove_labels_does_not_mutate_input(self):
        """Input frames still have their label tensor after remove_labels call."""
        traj = generate_trajectory(n_points=20, n_frames=3, seed=37)
        labelled = generate_labels(traj, n_labels=3, seed=42)
        # Capture the label tensors before the call
        original_colors = {i: labelled[i]["label"].clone() for i in labelled}
        remove_labels(labelled)
        # Input must still have its label tensors
        for i in labelled:
            assert labelled[i]["label"] is not None
            assert torch.equal(labelled[i]["label"], original_colors[i])


# ---------------------------------------------------------------------------
# TestLabelRegionShapes
# ---------------------------------------------------------------------------


class TestLabelRegionShapes:
    """Per-shape (voronoi/blob/cone) correctness and mixture-within-label (D-11) tests."""

    def _single_point_traj(self, xyz):
        """Single-frame trajectory with one point at `xyz`."""
        pc = zRegPointCloud(pos=torch.tensor([xyz], dtype=torch.float32))
        return {0: pc}

    def _empty_traj(self):
        """Single-frame trajectory with zero points."""
        pc = zRegPointCloud(pos=torch.zeros((0, 3), dtype=torch.float32))
        return {0: pc}

    def test_voronoi_point_at_center_gets_matching_label(self):
        """A point exactly at label 0's voronoi center gets label 0 when label 1's center is far away."""
        traj = self._single_point_traj([0.0, 0.0, 0.0])
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="voronoi", center=[0.0, 0.0, 0.0])]),
            LabelSpec(label_id=1, components=[LabelComponentSpec(shape="voronoi", center=[100.0, 100.0, 100.0])]),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        assert result[0]["label"].item() == 0

    def test_blob_point_at_center_gets_matching_label(self):
        """A point exactly at label 0's blob center (sigma=1.0) gets label 0 when label 1's blob center is far away."""
        traj = self._single_point_traj([0.0, 0.0, 0.0])
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="blob", center=[0.0, 0.0, 0.0], sigma=1.0)]),
            LabelSpec(
                label_id=1,
                components=[LabelComponentSpec(shape="blob", center=[100.0, 100.0, 100.0], sigma=1.0)],
            ),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        assert result[0]["label"].item() == 0

    def test_cone_point_at_pole_gets_matching_label(self):
        """A point exactly at label 0's cone pole (sigma=20.0) gets label 0 when label 1's pole points opposite."""
        traj = self._single_point_traj([0.0, 0.0, 1.0])
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="cone", center=[0.0, 0.0, 1.0], sigma=20.0)]),
            LabelSpec(
                label_id=1, components=[LabelComponentSpec(shape="cone", center=[0.0, 0.0, -1.0], sigma=20.0)]
            ),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        assert result[0]["label"].item() == 0

    def test_mixture_within_label_beats_competing_single_component(self):
        """A label with two components (one blob near the point, one far) still wins over a competing
        single-component label whose one component is far — proving multi-component mixture
        aggregation works (D-11)."""
        traj = self._single_point_traj([0.0, 0.0, 0.0])
        label_specs = [
            LabelSpec(
                label_id=0,
                components=[
                    LabelComponentSpec(shape="blob", center=[0.0, 0.0, 0.0], sigma=1.0),
                    LabelComponentSpec(shape="blob", center=[50.0, 50.0, 50.0], sigma=1.0),
                ],
            ),
            LabelSpec(
                label_id=1,
                components=[LabelComponentSpec(shape="blob", center=[50.0, 50.0, 50.0], sigma=1.0)],
            ),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        assert result[0]["label"].item() == 0

    def test_empty_frame_voronoi_no_raise(self):
        """generate_labels with label_specs (voronoi) on a zero-point frame produces a (0,)
        torch.long label tensor without raising."""
        traj = self._empty_traj()
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="voronoi", center=[0.0, 0.0, 0.0])]),
            LabelSpec(label_id=1, components=[LabelComponentSpec(shape="voronoi", center=[10.0, 0.0, 0.0])]),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        assert result[0]["label"].dtype == torch.long
        assert result[0]["label"].shape == (0,)

    def test_empty_frame_blob_cone_no_raise(self):
        """generate_labels with label_specs (blob/cone) on a zero-point frame produces a (0,)
        torch.long label tensor without raising."""
        traj = self._empty_traj()
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="blob", center=[0.0, 0.0, 0.0], sigma=1.0)]),
            LabelSpec(label_id=1, components=[LabelComponentSpec(shape="cone", center=[0.0, 0.0, 1.0], sigma=20.0)]),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        assert result[0]["label"].dtype == torch.long
        assert result[0]["label"].shape == (0,)


# ---------------------------------------------------------------------------
# TestLabelAssignmentModes
# ---------------------------------------------------------------------------


class TestLabelAssignmentModes:
    """Deterministic (argmax) vs probabilistic (sampled) assignment-mode behavior tests (D-12)."""

    def test_deterministic_mode_is_argmax_and_reproducible(self):
        """Deterministic mode assigns the unambiguously closer label to every point, bitwise
        reproducibly across repeated calls with the same seed."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        traj = {0: zRegPointCloud(pos=pos)}
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="voronoi", center=[0.0, 0.0, 0.0])]),
            LabelSpec(
                label_id=1,
                components=[LabelComponentSpec(shape="voronoi", center=[1000.0, 1000.0, 1000.0])],
            ),
        ]
        for _ in range(3):
            result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=7)
            assert torch.equal(result[0]["label"], torch.zeros(3, dtype=torch.long))

    def test_probabilistic_mode_samples_valid_label(self):
        """Probabilistic mode with two blob labels of very different distances yields a label_id
        that is one of the two valid ids."""
        pos = torch.tensor([[0.0, 0.0, 0.0]])
        traj = {0: zRegPointCloud(pos=pos)}
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="blob", center=[0.0, 0.0, 0.0], sigma=1.0)]),
            LabelSpec(
                label_id=1,
                components=[LabelComponentSpec(shape="blob", center=[10.0, 10.0, 10.0], sigma=1.0)],
            ),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="probabilistic", seed=3)
        assert result[0]["label"].item() in (0, 1)

    def test_probabilistic_mode_majority_favors_nearer_label(self):
        """Across many points concentrated near label 0's blob center, label 0 is assigned to a
        majority of them (statistical, not exact-count, assertion)."""
        torch.manual_seed(123)
        pos = torch.randn(200, 3) * 0.1  # concentrated near the origin
        traj = {0: zRegPointCloud(pos=pos)}
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="blob", center=[0.0, 0.0, 0.0], sigma=1.0)]),
            LabelSpec(
                label_id=1,
                components=[LabelComponentSpec(shape="blob", center=[20.0, 20.0, 20.0], sigma=1.0)],
            ),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="probabilistic", seed=9)
        labels = result[0]["label"]
        assert (labels == 0).float().mean().item() > 0.5

    def test_probabilistic_mode_with_n_labels_raises(self):
        """Probabilistic mode with n_labels (no label_specs) raises ValueError mentioning 'mode'
        (D-10 — the simple path does not support probabilistic)."""
        traj = generate_trajectory(n_points=10, n_frames=2, seed=40)
        with pytest.raises(ValueError, match="mode"):
            generate_labels(traj, n_labels=3, mode="probabilistic")

    def test_probabilistic_voronoi_missing_temperature_raises(self):
        """Probabilistic mode with a label_specs voronoi component missing temperature raises
        ValueError mentioning 'temperature'."""
        traj = {0: zRegPointCloud(pos=torch.tensor([[0.0, 0.0, 0.0]]))}
        label_specs = [
            LabelSpec(label_id=0, components=[LabelComponentSpec(shape="voronoi", center=[0.0, 0.0, 0.0])]),
            LabelSpec(label_id=1, components=[LabelComponentSpec(shape="voronoi", center=[5.0, 0.0, 0.0])]),
        ]
        with pytest.raises(ValueError, match="temperature"):
            generate_labels(traj, label_specs=label_specs, mode="probabilistic", seed=0)

    def test_label_specs_arbitrary_label_ids_round_trip(self):
        """label_specs with non-contiguous/arbitrary label_id values ({2, 7}) round-trips
        correctly — output label values are exactly the configured label_ids, not positional
        indices."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]])
        traj = {0: zRegPointCloud(pos=pos)}
        label_specs = [
            LabelSpec(label_id=2, components=[LabelComponentSpec(shape="voronoi", center=[0.0, 0.0, 0.0])]),
            LabelSpec(label_id=7, components=[LabelComponentSpec(shape="voronoi", center=[50.0, 50.0, 50.0])]),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        labels = result[0]["label"]
        assert set(labels.tolist()) <= {2, 7}
        assert labels[0].item() == 2
        assert labels[1].item() == 7


# ---------------------------------------------------------------------------
# TestLabelGenerationD07Regression
# ---------------------------------------------------------------------------


class TestLabelGenerationD07Regression:
    """Regression tests proving region/component centers are resolved once per trajectory,
    not redrawn per frame (D-07's exact bug fix)."""

    def test_n_labels_path_identical_pos_across_frames_yields_identical_labels(self):
        """Two frames with an identical single point at the same pos produce an identical label
        under the n_labels path — only guaranteed if centers are resolved once per trajectory."""
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        traj = {0: zRegPointCloud(pos=pos.clone()), 1: zRegPointCloud(pos=pos.clone())}
        result = generate_labels(traj, n_labels=5, mode="deterministic", seed=0)
        assert torch.equal(result[0]["label"], result[1]["label"])

    def test_label_specs_path_identical_pos_across_frames_yields_identical_labels(self):
        """Three frames with an identical single point at the same pos produce identical labels
        under the label_specs multi-component mixture path."""
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        traj = {i: zRegPointCloud(pos=pos.clone()) for i in range(3)}
        label_specs = [
            LabelSpec(
                label_id=0,
                components=[
                    LabelComponentSpec(shape="blob", center=[1.0, 2.0, 3.0], sigma=1.0),
                    LabelComponentSpec(shape="cone", center=[0.0, 0.0, 1.0], sigma=20.0),
                ],
            ),
            LabelSpec(
                label_id=1,
                components=[LabelComponentSpec(shape="voronoi", center=[-5.0, -5.0, -5.0])],
            ),
        ]
        result = generate_labels(traj, label_specs=label_specs, mode="deterministic", seed=0)
        assert torch.equal(result[0]["label"], result[1]["label"])
        assert torch.equal(result[1]["label"], result[2]["label"])


# ---------------------------------------------------------------------------
# TestSampleBall
# ---------------------------------------------------------------------------


class TestSampleBall:
    """Tests for the sample_ball single-frame solid-ball sampler."""

    def test_sample_ball_shape_and_dtype(self):
        """sample_ball(200, seed=0) returns a (200, 3) float32 torch.Tensor."""
        pos = sample_ball(n_points=200, seed=0)
        assert isinstance(pos, torch.Tensor)
        assert pos.shape == (200, 3)
        assert pos.dtype == torch.float32

    def test_sample_ball_points_inside_default_radius(self):
        """All points lie inside the default radius=1.0 sphere."""
        pos = sample_ball(n_points=200, seed=0)
        assert pos.norm(dim=1).max().item() <= 1.0 + 1e-5

    def test_sample_ball_seed_reproducibility(self):
        """Same seed produces bitwise-equal tensors across two calls."""
        a = sample_ball(n_points=150, seed=7)
        b = sample_ball(n_points=150, seed=7)
        assert torch.equal(a, b)

    def test_sample_ball_different_seeds_differ(self):
        """Different seeds produce non-equal tensors."""
        a = sample_ball(n_points=150, seed=7)
        b = sample_ball(n_points=150, seed=8)
        assert not torch.equal(a, b)

    def test_sample_ball_invalid_n_points(self):
        """sample_ball(0) raises ValueError matching 'n_points'."""
        with pytest.raises(ValueError, match="n_points"):
            sample_ball(n_points=0)

    def test_sample_ball_non_default_radius_respected(self):
        """radius=2.5 produces points within 2.5 but with max norm > 1.0."""
        pos = sample_ball(n_points=500, seed=1, radius=2.5)
        max_norm = pos.norm(dim=1).max().item()
        assert max_norm <= 2.5 + 1e-5
        assert max_norm > 1.0


# ---------------------------------------------------------------------------
# TestSampleBowl
# ---------------------------------------------------------------------------


class TestSampleBowl:
    """Tests for the sample_bowl single-frame lower-hemisphere-shell sampler."""

    def test_sample_bowl_shape_and_dtype(self):
        """sample_bowl(300, seed=1) returns a (300, 3) float32 torch.Tensor."""
        pos = sample_bowl(n_points=300, seed=1)
        assert isinstance(pos, torch.Tensor)
        assert pos.shape == (300, 3)
        assert pos.dtype == torch.float32

    def test_sample_bowl_satisfies_in_bowl_predicate(self):
        """Every returned point satisfies the bowl predicate for R=1.0, d=0.5."""
        pos = sample_bowl(n_points=300, seed=1).numpy()
        R = 1.0
        d = 0.5
        norm_sq = (pos**2).sum(axis=1)
        carve_sq = pos[:, 0] ** 2 + pos[:, 1] ** 2 + (pos[:, 2] - d) ** 2
        assert (norm_sq <= R**2 + 1e-4).all()
        assert (carve_sq > R**2 + d**2 - 1e-4).all()
        assert (pos[:, 2] <= 1e-6).all()

    def test_sample_bowl_lower_hemisphere(self):
        """All returned points have z <= 1e-6."""
        pos = sample_bowl(n_points=300, seed=1)
        assert (pos[:, 2] <= 1e-6).all()

    def test_sample_bowl_seed_reproducibility(self):
        """Same seed produces bitwise-equal tensors across two calls."""
        a = sample_bowl(n_points=120, seed=3)
        b = sample_bowl(n_points=120, seed=3)
        assert torch.equal(a, b)

    def test_sample_bowl_different_seeds_differ(self):
        """Different seeds produce non-equal tensors."""
        a = sample_bowl(n_points=120, seed=3)
        b = sample_bowl(n_points=120, seed=4)
        assert not torch.equal(a, b)

    def test_sample_bowl_invalid_n_points(self):
        """sample_bowl(0) raises ValueError matching 'n_points'."""
        with pytest.raises(ValueError, match="n_points"):
            sample_bowl(n_points=0)

    def test_sample_ball_bowl_package_export(self):
        """from zreg.data_generation import sample_ball, sample_bowl succeeds."""
        from zreg.data_generation import sample_ball as pkg_sample_ball
        from zreg.data_generation import sample_bowl as pkg_sample_bowl

        assert pkg_sample_ball is sample_ball
        assert pkg_sample_bowl is sample_bowl


# ---------------------------------------------------------------------------
# Coverage gap tests for generators
# ---------------------------------------------------------------------------


class TestGeneratorsCoverageGaps:
    """Coverage gaps: seed=None branches in corruption.py and labels.py."""

    def test_add_gaussian_noise_seed_none(self):
        """corruption.py:53 — seed=None skips torch.manual_seed call."""
        traj = generate_trajectory(n_points=20, n_frames=2, seed=0)
        # Should not raise; just runs with caller's RNG state
        result = add_gaussian_noise(traj, sigma=0.01, seed=None)
        assert len(result) == 2
        for i in traj:
            assert result[i]["pos"].shape == traj[i]["pos"].shape

    def test_add_outliers_seed_none(self):
        """corruption.py:106 — seed=None skips torch.manual_seed call."""
        traj = generate_trajectory(n_points=20, n_frames=2, seed=0)
        result = add_outliers(traj, n_outliers=3, seed=None)
        for i in traj:
            assert result[i]["pos"].shape[0] == 23

    def test_add_outliers_with_fps_idx(self):
        """corruption.py:142-148 — fps-idx extended when pc['fps-idx'] is not None."""
        from zreg.core.dataset import zRegPointCloud
        traj = generate_trajectory(n_points=20, n_frames=1, seed=0)
        traj[0]["fps-idx"] = torch.arange(20)
        result = add_outliers(traj, n_outliers=5, seed=42)
        assert result[0]["fps-idx"] is not None
        assert result[0]["fps-idx"].shape[0] == 25

    def test_generate_labels_seed_none(self):
        """labels.py:72 — seed=None skips torch.manual_seed call."""
        traj = generate_trajectory(n_points=10, n_frames=2, seed=0)
        result = generate_labels(traj, n_labels=3, seed=None)
        for i in traj:
            assert result[i]["label"] is not None
            assert result[i]["label"].shape[0] == 10
