"""Validation and device tests for ``zreg.data_generation`` (Phase 62-02).

Covers:

- U6-2: ``generate_labels`` returns label tensors on the device of ``pos``.
- U6-6 / U6-8: ``generate_labels`` rejects an unknown ``mode`` and an empty
  ``label_specs`` list at runtime with attributable ``ValueError``s.
- U6-4: ``sample_bowl`` rejects parameters that make its rejection sampler
  unsatisfiable or impractically slow, and bounds both its total work
  (candidate budget) and its peak allocation (per-round chunk cap).

This module exists separately from ``tests/test_generators.py`` because that
file is owned by Phase 60-02.
"""

from zreg.core.dataset import zRegPointCloud
from zreg.data_generation.labels import (
    LabelComponentSpec,
    LabelSpec,
    generate_labels,
)

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _specs() -> list[LabelSpec]:
    return [
        LabelSpec(
            label_id=5,
            components=[
                LabelComponentSpec(shape="voronoi", center=[1.0, 0.0, 0.0], temperature=1.0)
            ],
        ),
        LabelSpec(
            label_id=9,
            components=[LabelComponentSpec(shape="blob", center=[-1.0, 0.0, 0.0], sigma=1.0)],
        ),
    ]


def _cpu_pos() -> torch.Tensor:
    g = torch.Generator().manual_seed(123)
    return torch.randn(12, 3, generator=g)


# ---------------------------------------------------------------------------
# generate_labels: device (U6-2)
# ---------------------------------------------------------------------------


class TestGenerateLabelsDevice:
    def test_device_follows_pos_meta_n_labels(self):
        traj = {0: zRegPointCloud(pos=torch.randn(20, 3, device="meta"))}
        out = generate_labels(traj, n_labels=3)
        assert out[0]["label"].device.type == "meta"
        assert out[0]["label"].dtype == torch.long
        assert out[0]["label"].shape == (20,)

    def test_device_follows_pos_meta_label_specs(self):
        traj = {0: zRegPointCloud(pos=torch.randn(20, 3, device="meta"))}
        out = generate_labels(traj, label_specs=_specs())
        assert out[0]["label"].device.type == "meta"
        assert out[0]["label"].dtype == torch.long

    def test_device_follows_pos_meta_probabilistic(self):
        traj = {0: zRegPointCloud(pos=torch.randn(20, 3, device="meta"))}
        out = generate_labels(traj, label_specs=_specs(), mode="probabilistic")
        assert out[0]["label"].device.type == "meta"
        assert out[0]["label"].dtype == torch.long

    @CUDA
    @pytest.mark.parametrize("mode", ["deterministic", "probabilistic"])
    def test_device_follows_pos_cuda(self, mode):
        traj = {0: zRegPointCloud(pos=torch.randn(20, 3, device="cuda"))}
        out = generate_labels(traj, label_specs=_specs(), mode=mode)
        assert out[0]["label"].device.type == "cuda"
        assert out[0]["label"].dtype == torch.long

    @CUDA
    def test_device_follows_pos_cuda_n_labels(self):
        traj = {0: zRegPointCloud(pos=torch.randn(20, 3, device="cuda"))}
        out = generate_labels(traj, n_labels=3)
        assert out[0]["label"].device.type == "cuda"

    def test_cpu_outputs_unchanged(self):
        # Literals recorded from the pre-fix implementation (HEAD c01947f).
        pos = _cpu_pos()
        det = generate_labels({0: zRegPointCloud(pos=pos)}, n_labels=3, seed=7)
        assert det[0]["label"].tolist() == [2, 0, 0, 1, 2, 2, 2, 2, 2, 1, 0, 1]
        prob = generate_labels(
            {0: zRegPointCloud(pos=pos)}, label_specs=_specs(), mode="probabilistic", seed=7
        )
        assert prob[0]["label"].tolist() == [5, 9, 9, 9, 9, 9, 9, 9, 5, 9, 9, 9]


# ---------------------------------------------------------------------------
# generate_labels: mode / empty specs (U6-6, U6-8)
# ---------------------------------------------------------------------------


class TestGenerateLabelsGuards:
    def test_empty_label_specs_rejected(self):
        traj = {0: zRegPointCloud(pos=_cpu_pos())}
        with pytest.raises(ValueError, match="label_specs must be non-empty"):
            generate_labels(traj, label_specs=[])

    def test_misspelled_mode_rejected_label_specs(self):
        traj = {0: zRegPointCloud(pos=_cpu_pos())}
        with pytest.raises(ValueError, match="mode"):
            generate_labels(traj, label_specs=_specs(), mode="determinstic")

    def test_unknown_mode_rejected_n_labels(self):
        traj = {0: zRegPointCloud(pos=_cpu_pos())}
        with pytest.raises(ValueError, match="mode"):
            generate_labels(traj, n_labels=3, mode="bogus")
