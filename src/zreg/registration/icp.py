"""ICP (Iterative Closest Point) registration wrapper for point cloud alignment.

This module implements ``ICPRegistration``, a thin wrapper around Open3D's
point-to-point rigid ICP algorithm. It follows the same normalisation pattern
as CPD registration (normalise → register → denormalise) to cache and reuse
transformation matrices during the alignment pipeline.
"""

import open3d as o3d
import torch
import numpy as np

from zreg.dataset import zRegPointCloud
from zreg.types import StoredTransform
import zreg.utils as utils

__all__ = ["ICPRegistration", "ICPTransformation"]


class ICPTransformation:
    """Wrapper for ICP transformation matrix.

    Stores a 4×4 transformation matrix in denormalised space for direct
    application to point clouds during _build_aligned_cloud.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Initialize with a 4×4 transformation matrix.

        Parameters
        ----------
        matrix : np.ndarray
            [4, 4] transformation matrix in denormalised space.
        """
        self.matrix = matrix


class ICPRegistration:
    """Wrap Open3D ICP (point-to-point, rigid) for point cloud registration.

    ICP is a geometry-based registration method that iteratively minimizes the
    distance between corresponding points. Unlike CPD (probabilistic), ICP often
    converges faster and is well-suited for trajectories with good initial
    alignment.

    This class follows the same normalisation-denormalisation pattern as CPD
    to ensure consistent transformation matrices across the alignment pipeline
    (Phase 35+ StoredTransform pattern).

    Parameters
    ----------
    max_iterations : int, default 50
        Maximum number of ICP iterations.
    tolerance : float, default 1e-6
        Convergence tolerance for ICP (relative change in objective).
    """

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6) -> None:
        """Initialize ICP registration parameters.

        Parameters
        ----------
        max_iterations : int, default 50
            Maximum number of ICP iterations.
        tolerance : float, default 1e-6
            Convergence tolerance for relative change in objective.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def register(
        self,
        source: zRegPointCloud,
        target: zRegPointCloud,
    ) -> StoredTransform:
        """Register source to target using Open3D ICP.

        Workflow:
        1. Extract positions from source and target zRegPointCloud dicts
        2. Extract normalisation context (min/max per-cloud)
        3. Normalise both clouds: (cloud - min) / (max - min) * 2 - 1
        4. Run Open3D ICP on normalised tensors
        5. Extract 4×4 transformation matrix from ICP result
        6. Compute composite denormalisation: D_inv @ T @ D
           where D is the normalisation matrix for source, T is ICP result
        7. Return StoredTransform with composite matrix and denorm context

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

        # Convert to Open3D point clouds (torch backend)
        src_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(src_norm))
        tgt_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(tgt_norm))

        # Run ICP on normalised clouds
        icp_result = o3d.t.pipelines.registration.icp(
            source=src_cloud,
            target=tgt_cloud,
            max_correspondence_distance=float('inf'),  # No distance threshold
            init_source_to_target=o3d.core.Tensor.eye(4, o3d.core.float32),
            criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.tolerance,
                relative_rmse=self.tolerance,
                max_iteration=self.max_iterations,
            ),
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        # Extract 4×4 transformation matrix from ICP result (normalised space)
        icp_matrix_norm = icp_result.transformation.numpy().astype('float32')

        # Compute composite denormalisation matrix:
        # Points are normalised as: x_norm = 2 * (x - src_min) / (src_max - src_min) - 1
        # ICP operates in normalised space: x_norm_reg = T_icp @ x_norm
        # We need to return T such that x_reg = T @ x (in denormalised space)
        #
        # Denormalisation reverses the scaling:
        # x = (x_norm + 1) * (src_max - src_min) / 2 + src_min
        #
        # Composite: T = D_inv @ T_icp @ D
        # where D is the normalisation matrix (4×4 homogeneous)
        T_denorm = self._compute_composite_transform(
            icp_matrix_norm,
            src_min,
            src_max,
        )

        # Ensure matrix is float32 for consistency
        final_matrix = T_denorm.astype(np.float32)

        return StoredTransform(
            transform=ICPTransformation(final_matrix),
            src_min=src_min,
            src_max=src_max,
            tgt_min=tgt_min,
            tgt_max=tgt_max,
        )

    @staticmethod
    def _compute_composite_transform(
        icp_matrix: 'numpy.ndarray',
        src_min: torch.Tensor,
        src_max: torch.Tensor,
    ) -> 'numpy.ndarray':
        """Compute composite denormalisation matrix for ICP result.

        Given ICP transformation in normalised space T_icp and normalisation
        bounds, compute the composite matrix T that operates in denormalised space:
        T = D_inv @ T_icp @ D

        where D normalises points and D_inv denormalises.

        Parameters
        ----------
        icp_matrix : numpy.ndarray
            [4, 4] ICP transformation matrix (in normalised space).
        src_min : torch.Tensor
            Scalar minimum value from normalisation.
        src_max : torch.Tensor
            Scalar maximum value from normalisation.

        Returns
        -------
        numpy.ndarray
            [4, 4] composite transformation matrix (in denormalised space).
        """
        import numpy as np

        # Convert tensors to numpy for matrix operations
        src_min_np = src_min.item() if src_min.numel() == 1 else src_min.cpu().numpy()
        src_max_np = src_max.item() if src_max.numel() == 1 else src_max.cpu().numpy()

        # Build normalisation matrix D:
        # x_norm = 2 * (x - src_min) / (src_max - src_min) - 1
        # In homogeneous coords: [x_norm, 1] = D @ [x, 1]
        scale = (src_max_np - src_min_np) / 2.0
        D = np.eye(4, dtype=np.float32)
        D[0, 0] = D[1, 1] = D[2, 2] = 2.0 / (src_max_np - src_min_np)
        D[0, 3] = D[1, 3] = D[2, 3] = -2.0 * src_min_np / (src_max_np - src_min_np) - 1.0

        # Build inverse denormalisation matrix D_inv:
        # x = (x_norm + 1) * (src_max - src_min) / 2 + src_min
        D_inv = np.eye(4, dtype=np.float32)
        D_inv[0, 0] = D_inv[1, 1] = D_inv[2, 2] = scale
        D_inv[0, 3] = D_inv[1, 3] = D_inv[2, 3] = src_min_np

        # Composite: T = D_inv @ T_icp @ D
        composite = D_inv @ icp_matrix @ D

        return composite
