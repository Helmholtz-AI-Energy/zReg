"""Hand-rolled, joint-cloud eGNN: EGNNConv, EGNNLabelTransfer.

Implements 47-RESEARCH.md's Pattern 4 (executed and numerically
equivariance-verified live in that session's research run, max abs diff
~1e-6): an `EGNNConv(MessagePassing)` layer implementing the Satorras,
Hoogeboom, Welling (2021, arXiv:2102.09844) E(n)-equivariant update equations
via a **single** `propagate()` call. `message()` computes the edge MLP output
`m_ij` (from `h_i`, `h_j`, squared relative distance) AND the coordinate
message `coord_mlp(m_ij) * (pos_i - pos_j)`, concatenated into one tensor. A
custom `aggregate()` override splits the concatenated tensor and reduces each
half independently via `torch_geometric.utils.scatter` -- `sum` for the
feature message (per the paper), `mean` for the coordinate message (keeps
updates scale-reasonable regardless of neighbour count).

This is the exact verified structure from 47-RESEARCH.md Pattern 4 -- do not
restructure it. Rotating input coordinates by an arbitrary rotation R before
`forward()` yields output coordinates rotated by the same R (E(3)
equivariance); node features `h` are invariant under the same rotation. A
subtle change to this structure silently breaks equivariance with no error
raised (T-47-06).

`EGNNLabelTransfer` composes: an input Linear embedding (`n_classes + 1 ->
hidden_dim`) for the joint-cloud conditioning feature (45-DESIGN.md Pattern
2 -- one-hot label for source rows, "unknown" flag bit for target rows), a
radius graph over the joint cloud via `_ops.build_radius_graph` (the
cross-cloud message-passing mechanism -- message-passing propagates
labelled-node signal to unlabelled nodes through edges that cross the
source/target boundary), a stack of `n_layers` `EGNNConv` layers, and a final
Linear readout to `n_classes` logits per point.

Slicing contract: matches `pointnet2.PointNet2LabelTransfer` -- this model
has no notion of `n_source`; `forward` returns logits for every joint point
and callers recover the supervised subset via `logits[n_source:]` (the joint
cloud is built by `torch.cat([source, target])`, so target rows are always
the last rows by construction).
"""

# zreg (and scipy) must be imported before torch/torch_geometric on macOS ARM
# to avoid duplicate libomp initialisation (SIGABRT) -- 47-RESEARCH.md
# Pitfall 1, mirrors src/zreg/models/pointnet2.py and tests/conftest.py.
from zreg.core.dataset import zRegPointCloud  # noqa: F401

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from ._ops import build_radius_graph

__all__ = ["EGNNConv", "EGNNLabelTransfer"]


class EGNNConv(MessagePassing):
    """One E(n)-equivariant message-passing layer.

    Verified this session (47-RESEARCH.md Pattern 4, executed and
    equivariance-checked, not just read): rotating input `pos` by an
    arbitrary rotation before `forward()` produces output `pos` rotated by
    the exact same rotation (max abs diff ~1e-6, float32 precision) --
    `h` is invariant under the same rotation. Uses ONE `propagate()` call
    whose `message()` emits `[h-message | coord-message]` concatenated, and
    a custom `aggregate()` override reduces each half independently (sum for
    h, mean for coord) before `forward()` applies the residual connections.

    Parameters
    ----------
    hidden_dim : int
        Node feature dimension (input and output).
    """

    def __init__(self, hidden_dim: int):
        super().__init__(aggr=None, flow="source_to_target")  # custom aggregate()
        self.hidden_dim = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU())

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        """Apply one equivariant message-passing update.

        Parameters
        ----------
        h : torch.Tensor
            [N, hidden_dim] node features.
        pos : torch.Tensor
            [N, 3] node positions.
        edge_index : torch.Tensor
            [2, E] directed edge index (source_to_target flow).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(h_new [N, hidden_dim], pos_new [N, 3])`` with feature and
            position residual connections applied.
        """
        out = self.propagate(edge_index, h=h, pos=pos, size=(h.size(0), h.size(0)))
        agg_m, agg_coord = out.split([self.hidden_dim, 3], dim=-1)
        h_new = h + self.node_mlp(torch.cat([h, agg_m], dim=-1))
        pos_new = pos + agg_coord
        return h_new, pos_new

    def message(self, h_i, h_j, pos_i, pos_j):
        rel = pos_i - pos_j
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))
        coord_msg = self.coord_mlp(m_ij) * rel
        return torch.cat([m_ij, coord_msg], dim=-1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_part, c_part = inputs.split([self.hidden_dim, 3], dim=-1)
        agg_m = scatter(m_part, index, dim=0, dim_size=dim_size, reduce="sum")
        agg_c = scatter(c_part, index, dim=0, dim_size=dim_size, reduce="mean")
        return torch.cat([agg_m, agg_c], dim=-1)


class EGNNLabelTransfer(nn.Module):
    """Joint-cloud eGNN label-transfer model: embed -> N x EGNNConv -> readout.

    Builds a radius graph over the joint (source+target-concatenated) cloud
    via ``_ops.build_radius_graph`` -- this IS the cross-cloud message
    -passing mechanism (edges crossing the source/target boundary propagate
    labelled-node signal to unlabelled nodes, per 45-RESEARCH.md's "eGNN is a
    more natural fit" reasoning). Stacks ``n_layers`` (default 4, a
    reasonable small-scale default per Satorras et al.'s own molecular-graph
    experiments -- not empirically tuned for this domain, Assumption A1)
    ``EGNNConv`` layers, then a final Linear readout to ``n_classes`` logits
    per point over the FULL joint cloud.

    Slicing contract: this model does NOT store or infer ``n_source`` --
    ``forward`` returns logits for every joint point. Callers slice
    ``logits[n_source:]`` to recover the target-only, supervised subset.

    Constructed on CPU by default; accepts no ``device`` kwarg and never
    calls a bare CUDA device method -- move an instance with the standard
    ``model.to(device)`` ``nn.Module`` API.

    Parameters
    ----------
    n_classes : int
        Number of output label classes.
    hidden_dim : int, optional
        Node feature width used throughout the EGNNConv stack, by default 32.
    n_layers : int, optional
        Number of stacked EGNNConv layers, by default 4.
    radius : float, optional
        Radius-graph search radius (``_ops.build_radius_graph``), by default 0.3.
    max_neighbors : int, optional
        Maximum neighbours kept per node in the radius graph, by default 16.
    """

    def __init__(
        self,
        n_classes: int,
        hidden_dim: int = 32,
        n_layers: int = 4,
        radius: float = 0.3,
        max_neighbors: int = 16,
    ):
        super().__init__()
        in_dim = n_classes + 1  # one-hot label (source) + unknown-flag bit (target)
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.radius = radius
        self.max_neighbors = max_neighbors

        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNConv(hidden_dim) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, joint_pos: torch.Tensor, joint_feat: torch.Tensor) -> torch.Tensor:
        """Per-point classification logits over the full joint cloud.

        Parameters
        ----------
        joint_pos : torch.Tensor
            [n_joint, 3] concatenated source+target positions.
        joint_feat : torch.Tensor
            [n_joint, n_classes + 1] concatenated source+target joint-cloud
            conditioning features (one-hot label / unknown-flag bit).

        Returns
        -------
        torch.Tensor
            [n_joint, n_classes] per-point logits. Caller slices
            ``[n_source:]`` for the target-only supervised subset.
        """
        h = self.embed(joint_feat)
        pos = joint_pos
        edge_index = build_radius_graph(pos, self.radius, self.max_neighbors)

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        return self.readout(h)
