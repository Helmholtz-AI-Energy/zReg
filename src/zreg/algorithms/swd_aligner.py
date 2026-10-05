"""Sliced Wasserstein Distance-based registration wrapper for point cloud alignment.

This module implements ``SlicedWassersteinAligner``, a gradient descent-based
wrapper for spatial registration using Sliced Wasserstein Distance variants.
It follows the same normalisation pattern as CPD and ICP registration
(normalise → register → denormalise) to cache and reuse transformation matrices
during the alignment pipeline.

The class supports all 6 SWD variants (swd, aswd, oswd, gswd, pswd, maxswd)
and projects the rotation onto SO(3) (SVD with det=+1 correction) after every
optimiser step and once more at return, so the result is always a proper
rotation.
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


def _nearest_rotation(m: torch.Tensor) -> torch.Tensor:
    """Return the proper rotation closest to ``m`` in the Frobenius norm.

    With the SVD ``m = U S V^T`` the result is ``U diag(1, 1, s) V^T`` where
    ``s = -1`` if ``det(U V^T) < 0`` and ``+1`` otherwise (``det == 0`` is
    treated as ``+1``). The sign flip turns a reflection into a rotation, so
    the result always has determinant ``+1``.

    Parameters
    ----------
    m : torch.Tensor
        ``[3, 3]`` matrix (any float dtype, any device).

    Returns
    -------
    torch.Tensor
        ``[3, 3]`` orthogonal matrix with determinant ``+1`` on the dtype and
        device of ``m``.

    Examples
    --------
    >>> import torch
    >>> from zreg.algorithms.swd_aligner import _nearest_rotation
    >>> r = _nearest_rotation(torch.diag(torch.tensor([1.0, 1.0, -1.0])))
    >>> round(torch.det(r).item(), 6)
    1.0
    """
    U, _, Vt = torch.linalg.svd(m)
    d = torch.det(U @ Vt)
    c = torch.ones(3, dtype=m.dtype, device=m.device)
    c[-1] = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    return (U * c.unsqueeze(0)) @ Vt


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
    To prevent rotation matrix drift from gradient descent, the rotation is
    projected onto SO(3) after every optimiser step and once more at return
    (SVD with det=+1 correction, see ``_nearest_rotation``). The returned
    rotation block is therefore orthogonal with determinant +1 to float
    precision, also at large learning rates.

    Point Counts
    ------------
    Source and target may have different point counts. The SW metrics compare
    the sorted 1-D projections through the exact 1-D quantile coupling (legacy
    max(N, M) scaling for the sort-sum variants; PSWD uses a weighted mean),
    see plan 61-01 in ``zreg.distance_metrics.sw_varients``; the aligner does
    no sampling of its own.

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
        2. Compute ONE shared scalar bounds pair (lo, hi) over both clouds
        3. Normalise both clouds with it: 2 * (cloud - lo) / (hi - lo) - 1
           (shared bounds keep the rigid result rigid after denormalisation)
        4. Centre both clouds on their centroids (seeds the centroid offset)
           and initialize rotation (identity 3×3) and residual translation
           (zero 3D vector)
        5. Instantiate SWD metric based on variant selection
        6. Run gradient descent optimization loop:
           - Transform centred source by current rotation + translation
           - Compute SWD loss between transformed source and centred target
           - Backward pass to update rotation/translation gradients
           - Project the rotation onto SO(3) after every optimiser step and
             once more at return (SVD with det=+1 correction)
        7. Compute composite D_inv @ T @ D, where D and D_inv come from
           ``zreg.utils.normalization_matrix`` / ``denormalization_matrix``
           (exact inverse pair) and T is the optimised affine matrix
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
            - src_min/src_max : shared normalisation bounds (scalars)
            - tgt_min/tgt_max : shared normalisation bounds (same as src_*)

        Raises
        ------
        ValueError
            If source or target is not ``(N, 3)``, is empty, contains NaN/inf,
            or has zero extent (all points coincide); raised by
            ``zreg.utils.registration_bounds`` before any registration work.
        """
        src_pos = source["pos"]
        tgt_pos = target["pos"]

        # One shared bounds pair for both clouds: the rigid result found in the
        # shared normalised frame stays rigid after denormalisation.
        lo, hi = utils.registration_bounds(src_pos, tgt_pos)
        src_norm = utils.normalize_point_cloud(src_pos, min_vals=lo, max_vals=hi)[0]
        tgt_norm = utils.normalize_point_cloud(tgt_pos, min_vals=lo, max_vals=hi)[0]

        # Ensure tensors are on the same device
        device = src_norm.device
        dtype = src_norm.dtype

        # Optimise about the centroids. Shared bounds no longer pre-centre each
        # cloud, so starting from a zero translation would leave SWD to learn
        # the whole inter-cloud offset at ~lr per Adam step (a 50-step budget
        # recovers only a fraction of it). Centring both clouds seeds the
        # centroid offset exactly and decouples rotation from translation:
        #   x' = R (x - mu_src) + t + mu_tgt,  with t a residual (init 0).
        src_mean = src_norm.mean(0).detach()
        tgt_mean = tgt_norm.mean(0).detach()
        src_centred = src_norm - src_mean
        tgt_centred = tgt_norm - tgt_mean

        # Initialize rotation (identity) and residual translation (zero)
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
        for _ in range(self.num_iterations):
            optimizer.zero_grad()

            # Transform centred source: x_transformed = x_c @ R^T + t
            # (rotation applied first, then translation)
            src_transformed = src_centred @ rotation.T + translation

            # Compute SWD loss
            # SWD metric will add batch dimension automatically (nobatchdim=True by default)
            loss = swd_metric(src_transformed, tgt_centred)

            # Backward pass
            loss.backward()

            # Keep only the rotational (tangent-space) part of the rotation
            # gradient, G -> R skew(R^T G). Adam rescales each entry
            # separately, so a raw gradient whose symmetric (scale/shear)
            # part dominates yields an update that is almost entirely
            # symmetric; the per-step SO(3) projection below would discard
            # it and the rotation would barely move.
            with torch.no_grad():
                a = rotation.T @ rotation.grad
                rotation.grad.copy_(rotation @ (0.5 * (a - a.T)))

            # Optimizer step
            optimizer.step()

            # Project onto SO(3) after every step. The in-place copy keeps the
            # parameter object Adam holds; det=+1 is enforced (no reflections).
            with torch.no_grad():
                rotation.copy_(_nearest_rotation(rotation.detach()))

        # Fold the centring back into one affine map in the shared normalised
        # frame: x' = R x + (t + mu_tgt - R mu_src).
        rotation_final = _nearest_rotation(rotation.detach())
        translation_final = translation.detach() + tgt_mean - rotation_final @ src_mean

        # Compute denormalized transformation matrix
        T_denorm = self._compute_composite_transform(
            rotation_final,
            translation_final,
            lo,
            hi,
        )

        return StoredTransform(
            transform=SWDTransformation(T_denorm),
            src_min=lo,
            src_max=hi,
            tgt_min=lo,
            tgt_max=hi,
        )

    @staticmethod
    def _compute_composite_transform(
        rotation: torch.Tensor,
        translation: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``D_inv @ T_norm @ D`` in the dtype/device of ``rotation``.

        ``T_norm = [[R, t], [0, 1]]`` is the optimised affine matrix in the
        shared normalised frame; ``D``/``D_inv`` come from the exact inverse
        helper pair in ``zreg.utils``.

        Parameters
        ----------
        rotation : torch.Tensor
            [3, 3] rotation matrix (in normalised space).
        translation : torch.Tensor
            [3] translation vector (in normalised space).
        lo, hi : torch.Tensor
            Shared scalar normalisation bounds.

        Returns
        -------
        torch.Tensor
            [4, 4] composite transformation matrix in original coordinates.
        """
        device = rotation.device
        dtype = rotation.dtype
        D = utils.normalization_matrix(lo, hi, dtype=dtype, device=device)
        D_inv = utils.denormalization_matrix(lo, hi, dtype=dtype, device=device)

        T_norm = torch.eye(4, device=device, dtype=dtype)
        T_norm[:3, :3] = rotation
        T_norm[:3, 3] = translation

        return D_inv @ T_norm @ D
