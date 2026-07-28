"""Hand-rolled, joint-cloud PointNet++: SetAbstraction, FeaturePropagation, PointNet2LabelTransfer.

Implements 47-RESEARCH.md's Pattern 3 (verified live in that session's research
run): a `SetAbstraction` layer (FPS -> ball-query -> per-group relative-position
+feature MLP -> max-pool), a `FeaturePropagation` layer (3-NN inverse-distance
interpolation + U-Net skip connection + MLP), and `PointNet2LabelTransfer`
composing 2 SA + 2 symmetric FP layers + a per-point classification head over a
**joint** (source+target-concatenated) cloud.

Joint-cloud conditioning (45-DESIGN.md Pattern 2): the caller builds
``joint_feat`` with ``in_dim = n_classes + 1`` -- one-hot(label) in the first
``n_classes`` dims for source rows, an explicit "unknown" flag bit (dim -1) set
for target rows. This model does not build that feature vector itself; it only
consumes it.

Slicing contract: ``joint_pos``/``joint_feat`` are built by the caller via
``torch.cat([source, target])`` (source rows first, target rows last). This
model has no notion of ``n_source`` and returns logits for every joint point;
callers recover the supervised subset via ``logits[n_source:]``.
"""

import numpy as np
import torch
import torch.nn as nn

from ._ops import ball_query, farthest_point_sample

__all__ = ["SetAbstraction", "FeaturePropagation", "PointNet2LabelTransfer"]


class SetAbstraction(nn.Module):
    """FPS -> ball-query -> per-group [relative_pos | neighbour_feat] MLP -> max-pool.

    Samples ``n_samples = max(1, int(n * ratio))`` centers -- always ratio-based,
    never a fixed absolute count (47-RESEARCH.md Pitfall 3 / Anti-Pattern), since
    ``TrainingTriple`` point counts vary 100-300 per seed. Grouping uses
    ``_ops.ball_query``, whose isolated-point self-fallback (``idx = [i]``) is
    what keeps sparse ``sample_bowl`` regions from crashing this layer's
    max-pool with an empty group (47-RESEARCH.md Pitfall 2 / T-47-03).

    Parameters
    ----------
    in_dim : int
        Input per-point feature dimension.
    hidden_dim : int
        Output feature dimension (also the shared per-group MLP's width).
    ratio : float, optional
        Fraction of points to keep as sampled centers, by default 0.5.
    radius : float, optional
        Ball-query search radius, by default 0.3.
    max_neighbors : int, optional
        Maximum neighbours kept per group, by default 16.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        ratio: float = 0.5,
        radius: float = 0.3,
        max_neighbors: int = 16,
    ):
        super().__init__()
        self.ratio = ratio
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.mlp = nn.Sequential(
            nn.Linear(3 + in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, pos: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Downsample (pos, feat) to (sampled_pos, sampled_feat).

        Parameters
        ----------
        pos : torch.Tensor
            [N, 3] point positions.
        feat : torch.Tensor
            [N, in_dim] per-point features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(sampled_pos [M, 3], sampled_feat [M, hidden_dim])`` where
            ``M = max(1, int(N * ratio))``.
        """
        n = pos.shape[0]
        n_samples = max(1, int(n * self.ratio))
        center_idx = farthest_point_sample(pos, n_samples)
        groups = ball_query(pos, center_idx, self.radius, self.max_neighbors)

        pooled_feats = []
        for center, group_idx in zip(center_idx.tolist(), groups):
            relative_pos = pos[group_idx] - pos[center]  # [G, 3]
            neighbour_feat = feat[group_idx]  # [G, in_dim]
            group_input = torch.cat([relative_pos, neighbour_feat], dim=-1)  # [G, 3+in_dim]
            group_out = self.mlp(group_input)  # [G, hidden_dim]
            pooled, _ = group_out.max(dim=0)  # [hidden_dim]
            pooled_feats.append(pooled)

        sampled_pos = pos[center_idx]
        sampled_feat = torch.stack(pooled_feats, dim=0)
        return sampled_pos, sampled_feat


class FeaturePropagation(nn.Module):
    """3-NN inverse-distance interpolation + U-Net skip concat + MLP.

    Interpolates ``feat_sparse`` (defined at ``pos_sparse``, a coarser
    resolution) back up to ``pos_dense``'s full resolution via standard
    PointNet++ three-nearest-neighbor inverse-distance weighting
    (``open3d.geometry.KDTreeFlann.search_knn_vector_3d(p, 3)``, weights
    ``1/dist`` normalized -- Don't Hand-Roll table, 47-RESEARCH.md), then
    concatenates the corresponding dense-resolution skip feature
    (``feat_dense_skip``) before a shared per-point MLP.

    Parameters
    ----------
    in_dim : int
        Feature dimension of ``feat_sparse`` (the coarse-resolution input).
    skip_dim : int
        Feature dimension of ``feat_dense_skip`` (the U-Net skip connection).
    hidden_dim : int
        Output feature dimension.
    """

    def __init__(self, in_dim: int, skip_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + skip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        pos_dense: torch.Tensor,
        feat_dense_skip: torch.Tensor,
        pos_sparse: torch.Tensor,
        feat_sparse: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate feat_sparse up to pos_dense resolution and fuse with skip.

        Parameters
        ----------
        pos_dense : torch.Tensor
            [N, 3] fine-resolution positions to interpolate onto.
        feat_dense_skip : torch.Tensor
            [N, skip_dim] fine-resolution skip features (from the matching SA
            layer's input).
        pos_sparse : torch.Tensor
            [M, 3] coarse-resolution positions (M <= N).
        feat_sparse : torch.Tensor
            [M, in_dim] coarse-resolution features to interpolate.

        Returns
        -------
        torch.Tensor
            [N, hidden_dim] fused, propagated features.
        """
        import open3d as o3d  # lazy import to avoid libomp conflict on macOS ARM
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            pos_sparse.detach().cpu().numpy().astype(np.float64)
        )
        tree = o3d.geometry.KDTreeFlann(pcd)

        k = min(3, pos_sparse.shape[0])
        pos_dense_np = pos_dense.detach().cpu().numpy().astype(np.float64)

        interpolated = []
        for p in pos_dense_np:
            _, idx, dist2 = tree.search_knn_vector_3d(p, k)
            idx = list(idx)
            dist = torch.as_tensor(
                np.sqrt(np.maximum(dist2, 1e-10)),
                dtype=feat_sparse.dtype,
                device=feat_sparse.device,
            )
            weights = 1.0 / dist.clamp_min(1e-10)
            weights = weights / weights.sum()
            neighbour_feat = feat_sparse[idx]  # [k, in_dim]
            interpolated.append((weights.unsqueeze(-1) * neighbour_feat).sum(dim=0))

        feat_interp = torch.stack(interpolated, dim=0)  # [N, in_dim]
        combined = torch.cat([feat_interp, feat_dense_skip], dim=-1)  # [N, in_dim+skip_dim]
        return self.mlp(combined)


class PointNet2LabelTransfer(nn.Module):
    """Joint-cloud PointNet++ label-transfer model: 2 SA + 2 FP + per-point head.

    Composes ``SetAbstraction`` (ratio 0.5 twice, hidden dims ``hidden_dim`` ->
    ``2*hidden_dim``) with two symmetric ``FeaturePropagation`` layers (U-Net
    skip connections back to each SA layer's input) and a final per-point MLP
    mapping to ``n_classes`` logits, over the FULL joint (source+target
    -concatenated) cloud.

    Slicing contract: this model does NOT store or infer ``n_source`` --
    ``forward`` returns logits for every joint point. Callers slice
    ``logits[n_source:]`` to recover the target-only, supervised subset (the
    joint cloud is built by ``torch.cat([source, target])``, so target rows are
    always the last ``len(target)`` rows by construction).

    Constructed on CPU by default; accepts no ``device`` kwarg and never calls
    a bare CUDA device method -- move an instance with the standard
    ``model.to(device)`` ``nn.Module`` API.

    Parameters
    ----------
    n_classes : int
        Number of output label classes.
    hidden_dim : int, optional
        Base hidden feature width, by default 32.
    radius : float, optional
        Base ball-query radius for the first SA layer (the second SA layer
        uses ``2 * radius`` to match its coarser resolution), by default 0.3.
    max_neighbors : int, optional
        Maximum neighbours per SA group, by default 16.
    """

    def __init__(
        self,
        n_classes: int,
        hidden_dim: int = 32,
        radius: float = 0.3,
        max_neighbors: int = 16,
    ):
        super().__init__()
        in_dim = n_classes + 1  # one-hot label (source) + unknown-flag bit (target)
        self.n_classes = n_classes
        self.in_dim = in_dim

        self.sa1 = SetAbstraction(
            in_dim=in_dim, hidden_dim=hidden_dim, ratio=0.5, radius=radius, max_neighbors=max_neighbors
        )
        self.sa2 = SetAbstraction(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            ratio=0.5,
            radius=radius * 2,
            max_neighbors=max_neighbors,
        )

        self.fp2 = FeaturePropagation(in_dim=hidden_dim * 2, skip_dim=hidden_dim, hidden_dim=hidden_dim)
        self.fp1 = FeaturePropagation(in_dim=hidden_dim, skip_dim=in_dim, hidden_dim=hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

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
        pos0, feat0 = joint_pos, joint_feat
        pos1, feat1 = self.sa1(pos0, feat0)
        pos2, feat2 = self.sa2(pos1, feat1)

        feat1_up = self.fp2(pos1, feat1, pos2, feat2)
        feat0_up = self.fp1(pos0, feat0, pos1, feat1_up)

        return self.head(feat0_up)
