"""Sliced Wasserstein Distance-based registration wrapper for point cloud alignment.

This module implements ``SlicedWassersteinAligner``, a gradient descent-based
wrapper for spatial registration using Sliced Wasserstein Distance variants.
It follows the same normalisation pattern as CPD and ICP registration
(normalise → register → denormalise) to cache and reuse transformation matrices
during the alignment pipeline.

The class supports all 6 SWD variants (swd, aswd, oswd, gswd, pswd, maxswd)
and enforces SO(3) orthogonality periodically via SVD projection to prevent
rotation matrix drift during optimization.
"""

from typing import Optional
import torch
import torch.nn as nn

from zreg.core.dataset import zRegPointCloud
from zreg.core.types import StoredTransform
from zreg.distance_metrics.sw_varients import (
    SlicedWassersteinDistance,
    AdaptiveSlicedWassersteinDistance,
    OrthogonalSlicedWassersteinDistance,
    GeneralisedSlicedWassersteinDistance,
    ProjectedWassersteinDistance,
    MaxSlicedWassersteinDistance,
)
import zreg.utils as utils

__all__ = ["SlicedWassersteinAligner", "SWDTransformation"]


class SWDTransformation:
    """Wrapper for SWD transformation matrix.

    Stores a 4×4 transformation matrix in denormalised space for direct
    application to point clouds during _build_aligned_cloud.
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """Initialize with a 4×4 transformation matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            [4, 4] transformation matrix in denormalised space.
        """
        self.matrix = matrix


class SlicedWassersteinAligner:
    """Iterative SWD-based point cloud registration via gradient descent.

    Performs spatial alignment by minimizing Sliced Wasserstein Distance between
    source and target point clouds. Unlike ICP (discrete nearest-neighbor pairing)
    and CPD (probabilistic EM), SWD uses deterministic gradient-based optimization
    to find the best rigid transformation (rotation + translation).

    Recommended SWD variants for alignment (v1.4):
    - aswd: Adaptive Sliced Wasserstein Distance (recommended, most stable)
    - swd: Sliced Wasserstein Distance (fixed num_projs)
    - oswd: Orthogonal Sliced Wasserstein Distance
    - gswd: Generalised Sliced Wasserstein Distance
    - pswd: Projected Wasserstein Distance

    Not recommended for alignment:
    - maxswd: Max Sliced Wasserstein Distance (nested optimization convergence
      variability; use aswd/gswd/pswd instead). Available for DTW temporal alignment
      via pairwise_distance_matrix.

    This class follows the same normalisation-denormalisation pattern as CPD and ICP
    to ensure consistent transformation matrices across the alignment pipeline
    (Phase 35+ StoredTransform pattern).

    Orthogonality Enforcement
    -------------------------
    To prevent rotation matrix drift from gradient descent, SO(3) orthogonality is
    enforced via SVD projection every 10 iterations. This ensures the rotation
    matrix remains a valid orthogonal matrix throughout optimization.

    Parameters
    ----------
    variant : str, default "aswd"
        SWD variant to use. Recommended: swd, aswd, oswd, gswd, pswd.
        (maxswd not recommended for alignment; deferred to v1.5)
    num_iterations : int, default 50
        Maximum number of gradient descent iterations.
    learning_rate : float, default 1e-3
        Learning rate for Adam optimizer.
    **variant_kwargs
        Additional keyword arguments passed to the SWD distance class constructor.
        Examples: num_projs (for swd), max_sw_num_iters (for maxswd), degree (for gswd).
    """

    # Mapping of variant names to SWD distance classes
    VARIANT_CLASSES = {
        "swd": SlicedWassersteinDistance,
        "aswd": AdaptiveSlicedWassersteinDistance,
        "oswd": OrthogonalSlicedWassersteinDistance,
        "gswd": GeneralisedSlicedWassersteinDistance,
        "pswd": ProjectedWassersteinDistance,
        "maxswd": MaxSlicedWassersteinDistance,
    }

    def __init__(
        self,
        variant: str = "aswd",
        num_iterations: int = 50,
        learning_rate: float = 1e-3,
        **variant_kwargs,
    ) -> None:
        """Initialize SWD registration parameters.

        Parameters
        ----------
        variant : str, default "aswd"
            SWD variant to use.
        num_iterations : int, default 50
            Maximum number of gradient descent iterations.
        learning_rate : float, default 1e-3
            Learning rate for Adam optimizer.
        **variant_kwargs
            Additional keyword arguments for the SWD distance class.

        Raises
        ------
        ValueError
            If variant is not in the supported set.
        """
        if variant not in self.VARIANT_CLASSES:
            raise ValueError(
                f"Unknown variant: {variant}. "
                f"Must be one of: {list(self.VARIANT_CLASSES.keys())}"
            )

        self.variant = variant
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.variant_kwargs = variant_kwargs

    def register(
        self,
        source: zRegPointCloud,
        target: zRegPointCloud,
    ) -> StoredTransform:
        """Register source to target using iterative SWD optimization.

        Workflow:
        1. Extract positions from source and target zRegPointCloud dicts
        2. Extract normalisation context (min/max per-cloud)
        3. Normalise both clouds: (cloud - min) / (max - min) * 2 - 1
        4. Initialize rotation (identity 3×3) and translation (zero 3D vector)
        5. Instantiate SWD metric based on variant selection
        6. Run gradient descent optimization loop:
           - Transform source by current rotation + translation
           - Compute SWD loss between transformed source and target
           - Backward pass to update rotation/translation gradients
           - Enforce SO(3) orthogonality via SVD projection (every 10 iters)
        7. Compute composite denormalisation: D_inv @ T @ D
           where D is the normalisation matrix for source, T is the result
        8. Return StoredTransform with composite matrix and denorm context

        The final matrix operates in denormalised (full-resolution) space, so it can
        be applied directly to source frames without re-normalisation.

        Parameters
        ----------
        source : zRegPointCloud
            Source point cloud (dict with "pos" key containing [N, 3] tensor).
        target : zRegPointCloud
            Target point cloud (dict with "pos" key containing [M, 3] tensor).

        Returns
        -------
        StoredTransform
            Cached transformation with:
            - matrix : [4, 4] composite transformation matrix (in denormalised space)
            - src_min/src_max : source normalisation bounds (scalars)
            - tgt_min/tgt_max : target normalisation bounds (scalars)
        """
        src_pos = source["pos"]
        tgt_pos = target["pos"]

        # Extract normalisation context (min/max) for both clouds
        src_norm, (src_min, src_max) = utils.normalize_point_cloud(src_pos)
        tgt_norm, (tgt_min, tgt_max) = utils.normalize_point_cloud(tgt_pos)

        # Ensure tensors are on the same device
        device = src_norm.device
        dtype = src_norm.dtype

        # Initialize rotation (identity) and translation (zero)
        rotation = torch.eye(3, device=device, dtype=dtype)
        translation = torch.zeros(3, device=device, dtype=dtype)

        # Enable gradient computation
        rotation.requires_grad = True
        translation.requires_grad = True

        # Instantiate SWD metric
        swd_class = self.VARIANT_CLASSES[self.variant]

        # Provide default num_projs for variants that require it
        variant_kwargs = self.variant_kwargs.copy()
        if self.variant in ("swd", "oswd", "gswd", "pswd"):
            if "num_projs" not in variant_kwargs:
                variant_kwargs["num_projs"] = 50  # Sensible default

        swd_metric = swd_class(device=device, **variant_kwargs)

        # Create optimizer
        optimizer = torch.optim.Adam([rotation, translation], lr=self.learning_rate)

        # Optimization loop
        for step in range(self.num_iterations):
            optimizer.zero_grad()

            # Transform source: x_transformed = x_norm @ R^T + t
            # (rotation applied first, then translation)
            src_transformed = src_norm @ rotation.T + translation

            # Compute SWD loss
            # SWD metric will add batch dimension automatically (nobatchdim=True by default)
            loss = swd_metric(src_transformed, tgt_norm)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Enforce orthogonality every 10 iterations
            # SVD projection: R_new = U @ V^T where R = U @ Sigma @ V^T
            if step % 10 == 0:
                U, _, Vt = torch.linalg.svd(rotation.detach())
                rotation.data = U @ Vt

        # Compute denormalized transformation matrix
        T_denorm = self._compute_composite_transform(
            rotation.detach(),
            translation.detach(),
            src_min,
            src_max,
            tgt_min,
            tgt_max,
        )

        return StoredTransform(
            transform=SWDTransformation(T_denorm),
            src_min=src_min,
            src_max=src_max,
            tgt_min=tgt_min,
            tgt_max=tgt_max,
        )

    @staticmethod
    def _compute_composite_transform(
        rotation: torch.Tensor,
        translation: torch.Tensor,
        src_min: torch.Tensor,
        src_max: torch.Tensor,
        tgt_min: torch.Tensor,
        tgt_max: torch.Tensor,
    ) -> torch.Tensor:
        """Compute composite denormalisation matrix for SWD result.

        Given SWD-optimized rotation (3×3) and translation (3,) in normalised space,
        plus normalisation bounds, compute the composite matrix T that operates in
        denormalised space: T = D_inv @ T_norm @ D

        where:
        - D normalises points: x_norm = 2 * (x - src_min) / (src_max - src_min) - 1
        - D_inv denormalises: x = (x_norm + 1) * (src_max - src_min) / 2 + src_min
        - T_norm is the affine matrix in normalised space
        - T is the result (4×4) operating in denormalised space

        This follows the Phase 35/39 denormalization pattern (CPD and ICP).

        Parameters
        ----------
        rotation : torch.Tensor
            [3, 3] rotation matrix (in normalised space).
        translation : torch.Tensor
            [3] translation vector (in normalised space).
        src_min : torch.Tensor
            Scalar minimum value from source normalisation.
        src_max : torch.Tensor
            Scalar maximum value from source normalisation.
        tgt_min : torch.Tensor
            Scalar minimum value from target normalisation (unused in denorm).
        tgt_max : torch.Tensor
            Scalar maximum value from target normalisation (unused in denorm).

        Returns
        -------
        torch.Tensor
            [4, 4] composite transformation matrix (in denormalised space).
        """
        device = rotation.device
        dtype = rotation.dtype

        # Convert scalar tensors to float values
        src_min_val = src_min.item() if src_min.numel() == 1 else src_min
        src_max_val = src_max.item() if src_max.numel() == 1 else src_max

        # Build normalisation matrix D:
        # x_norm = 2 * (x - src_min) / (src_max - src_min) - 1
        # In homogeneous coords: [x_norm, 1] = D @ [x, 1]
        D = torch.eye(4, device=device, dtype=dtype)
        scale = (src_max_val - src_min_val) / 2.0
        D[0, 0] = D[1, 1] = D[2, 2] = 2.0 / (src_max_val - src_min_val)
        D[0, 3] = D[1, 3] = D[2, 3] = -2.0 * src_min_val / (src_max_val - src_min_val) - 1.0

        # Build denormalisation matrix D_inv:
        # x = (x_norm + 1) * (src_max - src_min) / 2 + src_min
        D_inv = torch.eye(4, device=device, dtype=dtype)
        D_inv[0, 0] = D_inv[1, 1] = D_inv[2, 2] = scale
        D_inv[0, 3] = D_inv[1, 3] = D_inv[2, 3] = src_min_val

        # Build affine transformation matrix T in normalised space (4×4 homogeneous):
        # T_norm = [[R, t],
        #           [0, 1]]
        T_norm = torch.eye(4, device=device, dtype=dtype)
        T_norm[:3, :3] = rotation
        T_norm[:3, 3] = translation

        # Composite: T_denorm = D_inv @ T_norm @ D
        composite = D_inv @ T_norm @ D

        return composite
