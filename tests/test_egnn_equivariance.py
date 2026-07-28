"""Tests for zreg.models.egnn — EGNNConv E(3) equivariance, feature invariance, gradients.

Wave-0 test file (45-DESIGN.md's "Downstream Wave-0 Test Obligations",
47-RESEARCH.md Pattern 4): reproduces the exact numerical equivariance check
performed live in that session's research run -- rotating input coordinates
by a fixed z-axis rotation before ``forward()`` vs. rotating the *output* of
an unrotated forward by the same rotation, max abs diff ~1e-6 in that
session -- as an automated regression test at ``atol=1e-4``. This is the
single most important correctness property for EGNN (T-47-06): a bug here
silently produces a non-equivariant, theoretically weaker network with no
error raised.
"""

# zreg (and scipy) must be imported before torch/torch_geometric on macOS ARM
# to avoid duplicate libomp initialisation (SIGABRT) -- 47-RESEARCH.md
# Pitfall 1, mirrors src/zreg/models/egnn.py and tests/conftest.py.
from zreg.core.dataset import zRegPointCloud  # noqa: F401

import math

import torch

from zreg.models._ops import build_radius_graph
from zreg.models.egnn import EGNNConv, EGNNLabelTransfer


def _z_rotation(theta_degrees: float) -> torch.Tensor:
    """Rotation matrix about the z-axis by theta_degrees, float32."""
    theta = math.radians(theta_degrees)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _make_graph(n: int, hidden_dim: int, seed: int):
    """Fixed-seed random node features/positions + a near-fully-connected radius graph."""
    gen = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=gen)
    h = torch.randn(n, hidden_dim, generator=gen)
    # radius=5.0 on standard-normal 3D points connects essentially every pair,
    # matching 47-RESEARCH.md Pattern 4's "20-node fully-connected test graph".
    edge_index = build_radius_graph(pos, radius=5.0, max_neighbors=n)
    return h, pos, edge_index


class TestEGNNConvEquivariance:
    """Rotating input pos before forward == rotating an unrotated forward's output pos."""

    def test_coordinates_are_e3_equivariant(self):
        torch.manual_seed(0)
        hidden_dim = 16
        h, pos, edge_index = _make_graph(n=20, hidden_dim=hidden_dim, seed=1)
        R = _z_rotation(37.0)

        layer = EGNNConv(hidden_dim)
        layer.eval()

        with torch.no_grad():
            _, pos_out_unrotated = layer(h, pos, edge_index)
            pos_out_then_rotated = pos_out_unrotated @ R.T

            pos_rotated_input = pos @ R.T
            _, pos_out_rotated_input = layer(h, pos_rotated_input, edge_index)

        assert torch.allclose(pos_out_rotated_input, pos_out_then_rotated, atol=1e-4)

    def test_features_are_invariant_under_rotation(self):
        torch.manual_seed(0)
        hidden_dim = 16
        h, pos, edge_index = _make_graph(n=20, hidden_dim=hidden_dim, seed=1)
        R = _z_rotation(51.0)

        layer = EGNNConv(hidden_dim)
        layer.eval()

        with torch.no_grad():
            h_out_unrotated, _ = layer(h, pos, edge_index)
            pos_rotated_input = pos @ R.T
            h_out_rotated_input, _ = layer(h, pos_rotated_input, edge_index)

        assert torch.allclose(h_out_unrotated, h_out_rotated_input, atol=1e-4)


class TestEGNNConvGradients:
    """forward() + backward() populate gradients on both h and pos."""

    def test_forward_backward_populates_gradients(self):
        torch.manual_seed(0)
        hidden_dim = 16
        h, pos, edge_index = _make_graph(n=20, hidden_dim=hidden_dim, seed=2)
        h.requires_grad_(True)
        pos.requires_grad_(True)

        layer = EGNNConv(hidden_dim)
        h_new, pos_new = layer(h, pos, edge_index)
        (h_new.sum() + pos_new.sum()).backward()

        assert h.grad is not None
        assert pos.grad is not None
        assert torch.any(h.grad != 0)
        assert torch.any(pos.grad != 0)


class TestEGNNLabelTransferEndToEnd:
    """EGNNLabelTransfer.forward on a joint cloud: shape contract + backward succeeds."""

    def test_forward_shape_and_backward(self):
        torch.manual_seed(0)
        n_classes = 6
        n_source, n_target = 120, 130
        n_joint = n_source + n_target
        joint_pos = torch.randn(n_joint, 3)
        joint_feat = torch.zeros(n_joint, n_classes + 1)
        joint_feat[:n_source, 0] = 1.0
        joint_feat[n_source:, -1] = 1.0

        model = EGNNLabelTransfer(n_classes=n_classes)
        logits = model(joint_pos, joint_feat)

        assert logits.shape == (n_joint, n_classes)
        target_logits = logits[n_source:]
        assert target_logits.shape == (n_target, n_classes)

        target_logits.sum().backward()
