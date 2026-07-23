"""Tests for the joint-cloud output-shape contract shared by BOTH label-transfer models.

Wave-0 test file mandated by 45-DESIGN.md's "Downstream Wave-0 Test Obligations" table:
``tests/test_zreg_models_joint_cloud.py`` — "Model output shape == target point count, not
source+target (joint-cloud conditioning contract)". This is the ONE test file that
single-sources the slicing contract for BOTH ``PointNet2LabelTransfer`` (47-02) and
``EGNNLabelTransfer`` (47-03): a joint cloud of ``n_source`` labeled + ``n_target``
unlabeled points must yield exactly ``n_target`` class-logit rows for the supervised
target subset via ``logits[n_source:]`` — never ``n_source + n_target`` rows.

Imports both model classes directly from their submodules (not the package ``__init__``)
to stay decoupled from plan 47-04's ``__init__.py`` rewrite, per this plan's <interfaces>
contract.
"""

# zreg (and scipy) must be imported before torch/open3d/torch_geometric on macOS ARM to
# avoid duplicate libomp initialisation (SIGABRT) -- 47-RESEARCH.md Pitfall 1, mirrors
# tests/conftest.py, tests/test_zreg_models_pointnet2.py, tests/test_egnn_equivariance.py.
from zreg.dataset import zRegPointCloud  # noqa: F401

import pytest
import torch
import torch.nn.functional as F

from zreg.models.egnn import EGNNLabelTransfer
from zreg.models.pointnet2 import PointNet2LabelTransfer

MODEL_CLASSES = [PointNet2LabelTransfer, EGNNLabelTransfer]
SIZE_REGIMES = [(100, 100), (300, 200)]


def _build_joint_cloud(
    n_source: int, n_target: int, n_classes: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a joint cloud: source rows carry one-hot(label), target rows carry the
    unknown-flag bit (45-DESIGN.md Pattern 2 joint-cloud conditioning layout).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(joint_pos [n_source+n_target, 3], joint_feat [n_source+n_target, n_classes+1])``
        with source rows first, target rows last (``torch.cat([source, target])``).
    """
    gen = torch.Generator().manual_seed(seed)
    source_pos = torch.randn(n_source, 3, generator=gen)
    target_pos = torch.randn(n_target, 3, generator=gen)
    joint_pos = torch.cat([source_pos, target_pos], dim=0)

    source_labels = torch.randint(0, n_classes, (n_source,), generator=gen)
    source_feat = F.one_hot(source_labels, num_classes=n_classes).float()
    source_feat = torch.cat([source_feat, torch.zeros(n_source, 1)], dim=-1)

    target_feat = torch.zeros(n_target, n_classes + 1)
    target_feat[:, -1] = 1.0

    joint_feat = torch.cat([source_feat, target_feat], dim=0)
    return joint_pos, joint_feat


class TestJointCloudSlicingContract:
    """Both models: forward over a joint cloud yields exactly n_target target-subset rows.

    Parametrized over both model classes AND two (n_source, n_target) size regimes in the
    100-300 point range, proving the contract holds across variable point counts rather
    than a single fixed size (45-DESIGN.md's core Wave-0 obligation for this file).
    """

    @pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("n_source,n_target", SIZE_REGIMES)
    def test_forward_target_subset_shape(self, model_class, n_source, n_target):
        n_classes = 6
        n_joint = n_source + n_target
        joint_pos, joint_feat = _build_joint_cloud(n_source, n_target, n_classes, seed=0)

        model = model_class(n_classes=n_classes)
        logits = model(joint_pos, joint_feat)

        assert logits.shape == (n_joint, n_classes)

        target_logits = logits[n_source:]
        assert target_logits.shape == (n_target, n_classes)
        assert target_logits.shape[0] == n_target
        assert target_logits.shape[0] != n_joint
