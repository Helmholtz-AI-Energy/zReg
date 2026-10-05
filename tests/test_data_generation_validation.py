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

import os
import re
import subprocess
import sys
import time

from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generators as generators_module
from zreg.data_generation.generators import sample_bowl
from zreg.data_generation.labels import (
    LabelComponentSpec,
    LabelSpec,
    generate_labels,
)

import numpy as np
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


# ---------------------------------------------------------------------------
# sample_bowl: parameter validation (U6-4)
# ---------------------------------------------------------------------------


class TestSampleBowlValidation:
    @pytest.mark.parametrize("d_ratio", [0.0, -0.5, 1e-4, float("nan"), float("inf")])
    def test_bad_d_ratio_rejected(self, d_ratio):
        with pytest.raises(ValueError, match="d_ratio"):
            sample_bowl(10, d_ratio=d_ratio)

    @pytest.mark.parametrize("radius", [0.0, -1.0, float("inf"), float("nan")])
    def test_bad_radius_rejected(self, radius):
        with pytest.raises(ValueError, match="radius"):
            sample_bowl(10, radius=radius)

    def test_d_ratio_at_bound_accepted(self):
        pos = sample_bowl(10, seed=0, d_ratio=1e-3)
        assert pos.shape == (10, 3)

    def test_bowl_zero_d_ratio_does_not_hang_subprocess(self):
        code = (
            "from zreg.data_generation.generators import sample_bowl; "
            "sample_bowl(10, d_ratio=0.0)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=60,
            env=dict(os.environ),
        )
        assert proc.returncode != 0
        assert b"ValueError" in proc.stderr


# ---------------------------------------------------------------------------
# sample_bowl: candidate budget and per-round chunk cap (U6-4, reviews 1-3)
# ---------------------------------------------------------------------------


def _drawn(exc: BaseException) -> int:
    m = re.search(r"drawn=(\d+)", str(exc))
    assert m is not None, str(exc)
    return int(m.group(1))


class TestSampleBowlBudget:
    def test_bowl_budget_smaller_than_first_batch(self, monkeypatch):
        monkeypatch.setattr(generators_module, "_MAX_BOWL_CANDIDATES", 4_000)
        with pytest.raises(RuntimeError, match="candidates") as excinfo:
            sample_bowl(1_000, seed=0, d_ratio=1e-3)
        assert "d_ratio" in str(excinfo.value)
        assert _drawn(excinfo.value) == 4_000

    def test_bowl_budget_not_multiple_of_batch(self, monkeypatch):
        monkeypatch.setattr(generators_module, "_MAX_BOWL_CANDIDATES", 4_500)
        with pytest.raises(RuntimeError, match="d_ratio") as excinfo:
            sample_bowl(100, seed=0, d_ratio=1e-3)
        assert _drawn(excinfo.value) == 4_500

    def test_bowl_n_points_above_budget_rejected(self, monkeypatch):
        monkeypatch.setattr(generators_module, "_MAX_BOWL_CANDIDATES", 4_000)
        t0 = time.perf_counter()
        with pytest.raises(ValueError, match="n_points"):
            sample_bowl(5_000, seed=0)
        assert time.perf_counter() - t0 < 1.0

    def test_bowl_chunk_cap_bounds_peak_allocation(self, monkeypatch):
        # Review cycle 3 MEDIUM: a large n_points must never draw a chunk
        # larger than _MAX_BOWL_BATCH, even though batch = n_points * 10.
        reference = sample_bowl(1_000, seed=0)  # uncapped (batch 10_000)
        monkeypatch.setattr(generators_module, "_MAX_BOWL_BATCH", 3_000)
        sizes: list[int] = []
        real_default_rng = np.random.default_rng

        class _SpyRng:
            def __init__(self, rng):
                self._rng = rng

            def uniform(self, low, high, size):
                sizes.append(size[0])
                return self._rng.uniform(low, high, size)

        monkeypatch.setattr(
            generators_module.np.random,
            "default_rng",
            lambda seed=None: _SpyRng(real_default_rng(seed)),
        )
        pos = sample_bowl(1_000, seed=0)  # batch would be 10_000
        assert pos.shape == (1_000, 3)
        assert sizes, "no candidate draw recorded"
        assert max(sizes) <= 3_000
        # numpy draws uniform candidates sequentially from one stream, so
        # chunking the same draw does not change the accepted points.
        assert torch.equal(pos, reference)

    def test_bowl_valid_output_unchanged(self):
        # Literals recorded from the pre-fix implementation (HEAD c01947f).
        pos = sample_bowl(100, seed=0)
        assert pos.shape == (100, 3)
        assert bool((pos[:, 2] <= 1e-6).all())
        assert torch.equal(pos, sample_bowl(100, seed=0))
        p64 = pos.double()
        assert p64.sum().item() == pytest.approx(-43.12759472243488, abs=0, rel=1e-12)
        assert p64[0].tolist() == [0.37689346075057983, -0.22215715050697327, -0.7298070192337036]

    def test_bowl_small_valid_d_ratio_unchanged(self):
        pos = sample_bowl(500, seed=1, d_ratio=0.05)
        assert pos.shape == (500, 3)
        p64 = pos.double()
        assert p64.sum().item() == pytest.approx(-346.7423937302665, abs=0, rel=1e-12)
        assert p64[0].tolist() == [-0.23701412975788116, 0.007605900522321463, -0.9665543437004089]

    def test_bowl_d_ratio_at_bound_unchanged(self):
        pos = sample_bowl(100, seed=0, d_ratio=1e-3)
        assert pos.double().sum().item() == pytest.approx(-75.19562984383083, abs=0, rel=1e-12)
