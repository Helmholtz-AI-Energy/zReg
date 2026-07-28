"""Tests for zreg.models.pointnet2 — SetAbstraction, FeaturePropagation, PointNet2LabelTransfer.

Wave-0 test file (45-DESIGN.md's "Downstream Wave-0 Test Obligations"): covers
ratio-based SA downsampling + backward, the joint-cloud slicing contract
(``logits[n_source:]``), ratio-based sizing across multiple joint-cloud sizes,
and bowl-sparse robustness (T-47-03 / 47-RESEARCH.md Pitfall 2) using a real
odd-seed ``DataFactory.generate_training_triple`` bowl geometry.
"""

# zreg (and scipy) must be imported before torch/open3d on macOS ARM to avoid
# duplicate libomp initialisation (SIGABRT) -- mirrors tests/conftest.py and
# 47-RESEARCH.md Pitfall 1.
from zreg.core.dataset import zRegPointCloud  # noqa: F401

import pytest
import torch
import torch.nn.functional as F

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from zreg.models.pointnet2 import PointNet2LabelTransfer, SetAbstraction


def _build_joint_feat(n_source: int, n_target: int, n_classes: int) -> torch.Tensor:
    """One-hot(label) on source rows + unknown-flag bit on target rows."""
    source_labels = torch.randint(0, n_classes, (n_source,))
    source_feat = F.one_hot(source_labels, num_classes=n_classes).float()
    source_feat = torch.cat([source_feat, torch.zeros(n_source, 1)], dim=-1)
    target_feat = torch.zeros(n_target, n_classes + 1)
    target_feat[:, -1] = 1.0
    return torch.cat([source_feat, target_feat], dim=0)


class TestSetAbstraction:
    """SA downsamples by ratio and backpropagates gradients."""

    def test_forward_downsamples_by_ratio(self):
        n, in_dim, hidden_dim, ratio = 200, 7, 16, 0.5
        pos = torch.randn(n, 3)
        feat = torch.randn(n, in_dim)
        sa = SetAbstraction(in_dim=in_dim, hidden_dim=hidden_dim, ratio=ratio, radius=0.5, max_neighbors=16)

        sampled_pos, sampled_feat = sa(pos, feat)

        expected_n = max(1, int(n * ratio))
        assert sampled_pos.shape == (expected_n, 3)
        assert sampled_feat.shape == (expected_n, hidden_dim)

    def test_backward_populates_gradients(self):
        n, in_dim = 150, 7
        pos = torch.randn(n, 3)
        feat = torch.randn(n, in_dim, requires_grad=True)
        sa = SetAbstraction(in_dim=in_dim, hidden_dim=16, ratio=0.5, radius=0.5, max_neighbors=16)

        _, sampled_feat = sa(pos, feat)
        sampled_feat.sum().backward()

        assert feat.grad is not None
        assert torch.any(feat.grad != 0)


class TestPointNet2LabelTransferSlicingContract:
    """forward() returns (n_joint, n_classes); logits[n_source:] is the target subset."""

    def test_forward_shape_and_slicing_contract(self):
        n_source, n_target, n_classes = 120, 130, 6
        n_joint = n_source + n_target
        joint_pos = torch.randn(n_joint, 3)
        joint_feat = _build_joint_feat(n_source, n_target, n_classes)
        model = PointNet2LabelTransfer(n_classes=n_classes)

        logits = model(joint_pos, joint_feat)

        assert logits.shape == (n_joint, n_classes)
        target_logits = logits[n_source:]
        assert target_logits.shape == (n_target, n_classes)


class TestRatioBasedSampling:
    """Two different joint-cloud sizes both run without a fixed-count assertion error."""

    @pytest.mark.parametrize("n_joint", [200, 600])
    def test_forward_runs_at_multiple_joint_sizes(self, n_joint):
        n_classes = 6
        n_source = n_joint // 2
        n_target = n_joint - n_source
        joint_pos = torch.randn(n_joint, 3)
        joint_feat = _build_joint_feat(n_source, n_target, n_classes)
        model = PointNet2LabelTransfer(n_classes=n_classes)

        logits = model(joint_pos, joint_feat)

        assert logits.shape == (n_joint, n_classes)
        assert logits[n_source:].shape == (n_target, n_classes)


class TestBowlSparseRobustness:
    """Bowl-shaped (odd-seed) joint cloud runs forward without an empty-neighbour crash."""

    def test_bowl_shaped_joint_cloud_forward_no_crash(self):
        n_classes = 6
        cfg = EvalConfig(data_path="x")
        factory = DataFactory(cfg)
        triple = factory.generate_training_triple(seed=1, n_classes=n_classes)  # odd seed -> bowl

        source_pos = triple.source_cloud["pos"]
        target_pos = triple.target_cloud["pos"]
        n_source, n_target = source_pos.shape[0], target_pos.shape[0]

        source_labels = triple.source_labels
        source_feat = F.one_hot(source_labels, num_classes=n_classes).float()
        source_feat = torch.cat([source_feat, torch.zeros(n_source, 1)], dim=-1)
        target_feat = torch.zeros(n_target, n_classes + 1)
        target_feat[:, -1] = 1.0

        joint_pos = torch.cat([source_pos, target_pos], dim=0)
        joint_feat = torch.cat([source_feat, target_feat], dim=0)

        model = PointNet2LabelTransfer(n_classes=n_classes)
        logits = model(joint_pos, joint_feat)

        assert logits.shape == (n_source + n_target, n_classes)
        assert logits[n_source:].shape == (n_target, n_classes)
        assert torch.isfinite(logits).all()
