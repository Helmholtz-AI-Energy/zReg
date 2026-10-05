"""ICP (Iterative Closest Point) registration wrapper for point cloud alignment.

This module implements ``ICPRegistration``, a thin wrapper around Open3D's
point-to-point rigid ICP algorithm. It follows the same normalisation pattern
as CPD registration (normalise → register → denormalise) to cache and reuse
transformation matrices during the alignment pipeline.
"""

import torch
import numpy as np

from zreg.core.dataset import zRegPointCloud
from zreg.core.types import StoredTransform
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
        2. Compute ONE shared scalar bounds pair (lo, hi) over both clouds
        3. Normalise both clouds with it: 2 * (cloud - lo) / (hi - lo) - 1
           (shared bounds keep a rigid result rigid after denormalisation)
        4. Hand the normalised clouds and detached bounds to the CPU
        5. Run Open3D ICP, seeded with the normalised centroid offset
        6. Compute composite D_inv @ T_icp @ D in CPU float64, where D and
           D_inv come from ``zreg.utils.normalization_matrix`` /
           ``denormalization_matrix`` (exact inverse pair)
        7. Return StoredTransform with the composite matrix (CPU numpy
           float32, also for CUDA inputs) and the shared bounds

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

        # One shared bounds pair for both clouds: a rigid ICP result in the
        # shared normalised frame stays rigid after denormalisation, and the
        # result is denormalised with the bounds of the frame it lives in.
        lo, hi = utils.registration_bounds(src_pos, tgt_pos)
        src_norm = utils.normalize_point_cloud(src_pos, min_vals=lo, max_vals=hi)[0]
        tgt_norm = utils.normalize_point_cloud(tgt_pos, min_vals=lo, max_vals=hi)[0]

        # Explicit CPU hand-off: Open3D and the composite matrix live on CPU,
        # so detach the bounds to CPU before building the float64 matrices
        # (CUDA inputs would otherwise mix devices).
        lo_cpu = lo.detach().cpu()
        hi_cpu = hi.detach().cpu()
        src_norm_cpu = src_norm.detach().cpu()
        tgt_norm_cpu = tgt_norm.detach().cpu()
        src_norm_np = src_norm_cpu.numpy()
        tgt_norm_np = tgt_norm_cpu.numpy()

        # Convert to Open3D point clouds
        import open3d as o3d  # lazy import to avoid libomp conflict on macOS ARM
        src_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(src_norm_np))
        tgt_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(tgt_norm_np))

        # Seed with the centroid offset: shared bounds no longer pre-centre the
        # clouds, and starting from identity lets the relative-fitness criterion
        # stop early on translated clouds.
        init = np.eye(4, dtype=np.float32)
        init[:3, 3] = (tgt_norm_cpu.mean(0) - src_norm_cpu.mean(0)).to(torch.float32).numpy()

        # Run ICP on normalised clouds
        icp_result = o3d.t.pipelines.registration.icp(
            source=src_cloud,
            target=tgt_cloud,
            max_correspondence_distance=float('inf'),  # No distance threshold
            init_source_to_target=o3d.core.Tensor(init),
            criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.tolerance,
                relative_rmse=self.tolerance,
                max_iteration=self.max_iterations,
            ),
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        # 4x4 ICP result in the shared normalised frame, as CPU float64
        icp_matrix_norm = torch.as_tensor(
            icp_result.transformation.cpu().numpy(), dtype=torch.float64
        )

        # Composite in original coordinates: D_inv @ T_icp @ D (CPU float64)
        composite = self._compute_composite_transform(icp_matrix_norm, lo_cpu, hi_cpu)

        # Keep the numpy float32 return contract (also for CUDA inputs)
        final_matrix = composite.cpu().numpy().astype(np.float32)

        return StoredTransform(
            transform=ICPTransformation(final_matrix),
            src_min=lo,
            src_max=hi,
            tgt_min=lo,
            tgt_max=hi,
        )

    @staticmethod
    def _compute_composite_transform(
        icp_matrix: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``D_inv @ T_icp @ D`` in CPU float64.

        Parameters
        ----------
        icp_matrix : torch.Tensor
            [4, 4] ICP transformation in the shared normalised frame.
        lo, hi : torch.Tensor
            Shared scalar normalisation bounds (moved to CPU by the helpers).

        Returns
        -------
        torch.Tensor
            [4, 4] CPU float64 composite transformation in original coordinates.
        """
        D = utils.normalization_matrix(lo, hi, dtype=torch.float64, device="cpu")
        D_inv = utils.denormalization_matrix(lo, hi, dtype=torch.float64, device="cpu")
        T = torch.as_tensor(icp_matrix, dtype=torch.float64).cpu()
        return D_inv @ T @ D
