"""Dynamic Time Warping for temporal alignment of 3D point cloud trajectories.

This module provides the DynamicTimeWarping class for computing optimal temporal
alignment between two sequences of point clouds.
"""

from pathlib import Path
import logging
import pickle

import torch

from .result import DTWResult
from ..pairwise_distance_matrix import create_pairwise_distance_matrix
from ...core.dataset import zRegPointCloud
from ...distance_metrics import DistanceMetric
from ...core.types import StoredTransform


log = logging.getLogger(__name__)

__all__ = ["DynamicTimeWarping"]


class DynamicTimeWarping:
    """Dynamic Time Warping for temporal alignment of 3D point cloud trajectories.

    This class computes the optimal temporal alignment between two sequences of
    point clouds (trajectories), enabling finding corresponding time points
    between trajectories with different lengths or temporal sampling rates.

    The implementation uses the existing pairwise distance matrix computation
    infrastructure, supporting various distance metrics and spatial registration.

    Parameters
    ----------
    x : dict[int, zRegPointCloud]
        First trajectory (source) - dictionary mapping time indices to point clouds.
    y : dict[int, zRegPointCloud]
        Second trajectory (target) - dictionary mapping time indices to point clouds.
    distance_metric : list[str | DistanceMetric] | str | DistanceMetric, optional
        Distance metric(s) for point cloud comparison. Accepts string identifiers
        ('swd', 'euclidean', 'manhattan', 'minkowski', 'cpd', 'aswd', 'oswd', 'gswd',
        'pswd') or any callable conforming to the `DistanceMetric` protocol
        (``zreg.distances.DistanceMetric``). Default: 'swd'.
    distance_kwargs : list[dict] | dict | None, optional
        Additional kwargs for distance functions. Default: None.
    downsample_method : str | None, optional
        Downsampling method ("random", "uniform", "farthest"). Default: "random".
    cpd_type : str | None, optional
        CPD registration type ("rigid", "affine", "nonrigid").
        If None, no spatial registration is performed. Default: None.
    window : int | None, optional
        Sakoe-Chiba band width for constraining warping path.
        If None, no constraint is applied (full matrix). Default: None.
    normalize : bool, optional
        Whether to normalize point clouds before distance computation. Default: True.
    mpi_distribute : bool, optional
        Whether to distribute computation across MPI processes. Default: False.

    Attributes
    ----------
    result : DTWResult | None
        The result of the DTW computation. None until `compute()` is called.

    Examples
    --------
    >>> dtw = DynamicTimeWarping(
    ...     x=trajectory_x,
    ...     y=trajectory_y,
    ...     distance_metric="swd",
    ...     downsample_method="random",
    ...     cpd_type="rigid",
    ...     window=10,
    ... )
    >>> result = dtw.compute()
    >>> path = result.warping_path
    >>> print(f"DTW distance: {result.distance:.4f}")
    """

    def __init__(
        self,
        x: dict[int, zRegPointCloud],
        y: dict[int, zRegPointCloud],
        distance_metric: list[str | DistanceMetric] | str | DistanceMetric = "swd",
        distance_kwargs: list[dict] | dict | None = None,
        downsample_method: str | None = "random",
        cpd_type: str | None = None,
        window: int | None = None,
        normalize: bool = True,
        mpi_distribute: bool = False,
    ) -> None:
        self.x = x
        self.y = y
        self.distance_metric = distance_metric
        self.distance_kwargs = distance_kwargs
        self.downsample_method = downsample_method
        self.cpd_type = cpd_type
        self.window = window
        self.normalize = normalize
        self.mpi_distribute = mpi_distribute

        # Result storage
        self.result: DTWResult | None = None

        # Internal state
        self._cost_matrix: torch.Tensor | None = None
        self._rotations: torch.Tensor | None = None
        self._stored_transforms: dict[tuple[int, int], StoredTransform] = {}

    def compute(self, metric_index: int = 0) -> DTWResult:
        """Run the full DTW pipeline.

        Computes the cost matrix, accumulated cost matrix, and extracts
        the optimal warping path via backtracing.

        Parameters
        ----------
        metric_index : int, optional
            If multiple distance metrics are used, specify which one to use
            for the DTW path computation. Default: 0 (first metric).

        Returns
        -------
        DTWResult
            Container with cost matrix, accumulated cost, warping path, and distance.
        """
        log.info("Starting DTW computation...")

        # Step 1: Compute pairwise cost matrix
        cost_matrix = self.compute_cost_matrix()

        # Select the metric to use for DTW
        if cost_matrix.ndim == 3:
            cost_for_dtw = cost_matrix[metric_index]
        else:
            cost_for_dtw = cost_matrix

        # Step 2: Compute accumulated cost matrix
        accumulated_cost = self._compute_accumulated_cost(cost_for_dtw)

        # Step 3: Backtrace to find optimal path
        warping_path = self._backtrace(accumulated_cost)

        # Step 4: Extract total distance
        distance = accumulated_cost[-1, -1].item()

        self.result = DTWResult(
            cost_matrix=cost_matrix,
            accumulated_cost=accumulated_cost,
            warping_path=warping_path,
            distance=distance,
            rotations=self._rotations,
            stored_transforms=self._stored_transforms,
        )

        log.info(f"DTW computation complete. Distance: {distance:.4f}, Path length: {len(warping_path)}")

        return self.result

    def compute_cost_matrix(self) -> torch.Tensor:
        """Compute the pairwise distance matrix between all time points.

        Uses the existing `create_pairwise_distance_matrix` function which
        handles distance computation, downsampling, and optional CPD registration.

        Returns
        -------
        torch.Tensor
            Cost matrix of shape (n_metrics, len(x), len(y)) or (len(x), len(y)).
        """
        if self._cost_matrix is not None:
            return self._cost_matrix

        log.info("Computing pairwise distance matrix...")

        pairwise_result = create_pairwise_distance_matrix(
            x=self.x,
            y=self.y,
            window=self.window,
            normalize=self.normalize,
            distance_metric=self.distance_metric,
            distance_kwargs=self.distance_kwargs,
            downsample_method=self.downsample_method,
            cpd_type=self.cpd_type,
            mpi_distribute=self.mpi_distribute,
        )
        self._cost_matrix = pairwise_result.cost_matrix
        self._rotations = pairwise_result.rotations
        self._stored_transforms = pairwise_result.stored_transforms

        return self._cost_matrix

    def _compute_accumulated_cost(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Compute the accumulated cost matrix using dynamic programming.

        The accumulated cost D[i,j] represents the minimum cost to reach
        position (i, j) from (0, 0), considering three possible predecessors:
        - D[i-1, j] (insertion)
        - D[i, j-1] (deletion)
        - D[i-1, j-1] (match)

        Parameters
        ----------
        cost_matrix : torch.Tensor
            Pairwise distance matrix of shape (n, m).

        Returns
        -------
        torch.Tensor
            Accumulated cost matrix of shape (n, m).
        """
        n, m = cost_matrix.shape
        accumulated = torch.full_like(cost_matrix, torch.inf)

        # Initialize first cell
        accumulated[0, 0] = cost_matrix[0, 0]

        # Initialize first column
        for i in range(1, n):
            if not torch.isinf(cost_matrix[i, 0]):
                accumulated[i, 0] = accumulated[i - 1, 0] + cost_matrix[i, 0]

        # Initialize first row
        for j in range(1, m):
            if not torch.isinf(cost_matrix[0, j]):
                accumulated[0, j] = accumulated[0, j - 1] + cost_matrix[0, j]

        # Fill the rest of the matrix
        for i in range(1, n):
            for j in range(1, m):
                if torch.isinf(cost_matrix[i, j]):
                    continue

                accumulated[i, j] = cost_matrix[i, j] + torch.min(
                    torch.stack([
                        accumulated[i - 1, j],      # insertion
                        accumulated[i, j - 1],      # deletion
                        accumulated[i - 1, j - 1],  # match
                    ])
                )

        return accumulated

    def _backtrace(self, accumulated_cost: torch.Tensor) -> list[tuple[int, int]]:
        """Extract the optimal warping path by backtracing through accumulated cost.

        Starts from the end point (n-1, m-1) and traces back to (0, 0)
        by always choosing the predecessor with minimum accumulated cost.

        Parameters
        ----------
        accumulated_cost : torch.Tensor
            Accumulated cost matrix of shape (n, m).

        Returns
        -------
        list[tuple[int, int]]
            Warping path as list of (i, j) index pairs, ordered from start to end.
        """
        n, m = accumulated_cost.shape
        i, j = n - 1, m - 1
        path = [(i, j)]

        while i > 0 or j > 0:
            if i == 0:
                predecessor = accumulated_cost[0, j - 1].item()
                if torch.isinf(accumulated_cost[0, j - 1]):
                    raise ValueError(
                        f"DTW warping path could not be traced — window too tight or "
                        f"disconnected cost matrix. "
                        f"Dead-end at ({i}, {j}), predecessor costs: {predecessor}"
                    )
                j -= 1
            elif j == 0:
                predecessor = accumulated_cost[i - 1, 0].item()
                if torch.isinf(accumulated_cost[i - 1, 0]):
                    raise ValueError(
                        f"DTW warping path could not be traced — window too tight or "
                        f"disconnected cost matrix. "
                        f"Dead-end at ({i}, {j}), predecessor costs: {predecessor}"
                    )
                i -= 1
            else:
                # Find the minimum predecessor
                candidates = torch.tensor([
                    accumulated_cost[i - 1, j],      # came from above (insertion)
                    accumulated_cost[i, j - 1],      # came from left (deletion)
                    accumulated_cost[i - 1, j - 1],  # came from diagonal (match)
                ])
                if torch.isinf(candidates).all():
                    predecessor_values = [
                        accumulated_cost[i - 1, j].item(),
                        accumulated_cost[i, j - 1].item(),
                        accumulated_cost[i - 1, j - 1].item(),
                    ]
                    raise ValueError(
                        f"DTW warping path could not be traced — window too tight or "
                        f"disconnected cost matrix. "
                        f"Dead-end at ({i}, {j}), predecessor costs: {predecessor_values}"
                    )
                argmin = torch.argmin(candidates).item()

                if argmin == 0:
                    i -= 1
                elif argmin == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1

            path.append((i, j))

        # Reverse to get path from start to end
        path.reverse()
        return path

    def get_warping_path(self) -> list[tuple[int, int]]:
        """Get the optimal warping path.

        Returns
        -------
        list[tuple[int, int]]
            Warping path as list of (x_idx, y_idx) pairs.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("DTW has not been computed yet. Call compute() first.")
        return self.result.warping_path

    def get_distance(self) -> float:
        """Get the total DTW distance.

        Returns
        -------
        float
            The accumulated cost at the end of the optimal path.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("DTW has not been computed yet. Call compute() first.")
        return self.result.distance

    def get_aligned_indices(self) -> tuple[list[int], list[int]]:
        """Get aligned index sequences from the warping path.

        This extracts two lists of indices that show how time points
        in x and y are aligned. Useful for resampling trajectories.

        Returns
        -------
        tuple[list[int], list[int]]
            (x_indices, y_indices) where x_indices[k] corresponds to y_indices[k].

        Raises
        ------
        RuntimeError
            If `compute()` has not been called yet.
        """
        path = self.get_warping_path()
        x_indices = [p[0] for p in path]
        y_indices = [p[1] for p in path]
        return x_indices, y_indices

    def get_aligned_trajectory(
        self,
        trajectory: dict[int, zRegPointCloud],
        reference: str = "x",
    ) -> dict[int, zRegPointCloud]:
        """Resample a trajectory according to the warping path.

        This creates a new trajectory where time points are remapped
        according to the DTW alignment.

        Parameters
        ----------
        trajectory : dict[int, zRegPointCloud]
            The trajectory to resample. Should be either x or y.
        reference : str, optional
            Which trajectory to use as reference for the new time indices.
            "x" means the output will have the same time indices as x.
            "y" means the output will have the same time indices as y.
            Default: "x".

        Returns
        -------
        dict[int, zRegPointCloud]
            Resampled trajectory with new time indices.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called yet.
        ValueError
            If reference is not "x" or "y".
        """
        if self.result is None:
            raise RuntimeError("DTW has not been computed yet. Call compute() first.")

        if reference not in ("x", "y"):
            raise ValueError(f"reference must be 'x' or 'y', got '{reference}'")

        x_indices, y_indices = self.get_aligned_indices()

        aligned = {}
        if reference == "x":
            # Map y's time points to x's time indices
            for x_idx, y_idx in zip(x_indices, y_indices):
                if x_idx not in aligned:  # Take first match if multiple
                    aligned[x_idx] = trajectory[y_idx]
        else:
            # Map x's time points to y's time indices
            for x_idx, y_idx in zip(x_indices, y_indices):
                if y_idx not in aligned:
                    aligned[y_idx] = trajectory[x_idx]

        return aligned

    def plot_alignment(
        self,
        save_path: str | Path | None = None,
        figsize: tuple[int, int] = (12, 5),
        metric_index: int = 0,
    ) -> None:
        """Visualize the cost matrix and warping path.

        Creates a figure with two subplots:
        1. Cost matrix with the optimal warping path overlaid
        2. Accumulated cost matrix

        Parameters
        ----------
        save_path : str | Path | None, optional
            Path to save the figure. If None, displays the plot. Default: None.
        figsize : tuple[int, int], optional
            Figure size (width, height) in inches. Default: (12, 5).
        metric_index : int, optional
            If multiple metrics, which one to plot. Default: 0.

        Raises
        ------
        RuntimeError
            If `compute()` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("DTW has not been computed yet. Call compute() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not available. Cannot create plot.")
            return

        cost = self.result.cost_matrix
        if cost.ndim == 3:
            cost = cost[metric_index]

        acc = self.result.accumulated_cost
        path = self.result.warping_path

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot cost matrix with path
        ax1 = axes[0]
        im1 = ax1.imshow(cost.cpu().numpy(), origin="lower", aspect="auto", cmap="viridis")
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax1.plot(path_x, path_y, "r-", linewidth=2, label="Warping path")
        ax1.set_xlabel("Trajectory Y (time)")
        ax1.set_ylabel("Trajectory X (time)")
        ax1.set_title("Cost Matrix with Warping Path")
        ax1.legend()
        plt.colorbar(im1, ax=ax1, label="Distance")

        # Plot accumulated cost matrix
        ax2 = axes[1]
        # Replace inf with nan for better visualization
        acc_viz = acc.clone()
        acc_viz[torch.isinf(acc_viz)] = float("nan")
        im2 = ax2.imshow(acc_viz.cpu().numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax2.plot(path_x, path_y, "r-", linewidth=2, label="Warping path")
        ax2.set_xlabel("Trajectory Y (time)")
        ax2.set_ylabel("Trajectory X (time)")
        ax2.set_title(f"Accumulated Cost (DTW Distance: {self.result.distance:.4f})")
        ax2.legend()
        plt.colorbar(im2, ax=ax2, label="Accumulated Cost")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved alignment plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def save(self, path: str | Path) -> None:
        """Save DTW results to disk.

        Saves the cost matrix, accumulated cost, warping path, distance,
        rotations, and stored_transforms. The main tensor data is saved to
        ``path`` (a .pt file, loadable with ``weights_only=True``). CPD
        transformation objects in ``stored_transforms`` are saved separately to
        ``path + ".transforms.pkl"`` using pickle, because they are not plain
        tensors and would be rejected by ``weights_only=True``.

        Parameters
        ----------
        path : str | Path
            Path to save the results (as a .pt file).

        Raises
        ------
        RuntimeError
            If `compute()` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("DTW has not been computed yet. Call compute() first.")

        path = Path(path)
        # Main tensor data — safe to load with weights_only=True.
        data = {
            "cost_matrix": self.result.cost_matrix,
            "accumulated_cost": self.result.accumulated_cost,
            "warping_path": self.result.warping_path,
            "distance": self.result.distance,
            "rotations": self.result.rotations,
            # Save configuration for reference
            "config": {
                "distance_metric": self.distance_metric,
                "downsample_method": self.downsample_method,
                "cpd_type": self.cpd_type,
                "window": self.window,
                "normalize": self.normalize,
            },
        }
        torch.save(data, path)

        # CPD transformation objects are not plain tensors; save them separately
        # so the main .pt file can be loaded safely with weights_only=True.
        transforms_path = Path(str(path) + ".transforms.pkl")
        with open(transforms_path, "wb") as f:
            pickle.dump(self.result.stored_transforms, f)

        log.info(f"Saved DTW results to {path} (transforms → {transforms_path})")

    @classmethod
    def load(cls, path: str | Path) -> DTWResult:
        """Load DTW results from disk.

        Parameters
        ----------
        path : str | Path
            Path to the saved results (.pt file).

        Returns
        -------
        DTWResult
            The loaded DTW results.
        """
        path = Path(path)
        # weights_only=True is safe because the main file contains only tensors,
        # a list, and primitives — no arbitrary pickle objects.
        data = torch.load(path, weights_only=True)

        # Load companion transforms file if it exists (written by save()).
        transforms_path = Path(str(path) + ".transforms.pkl")
        stored_transforms: dict = {}
        if transforms_path.exists():
            with open(transforms_path, "rb") as f:
                stored_transforms = pickle.load(f)  # noqa: S301

        result = DTWResult(
            cost_matrix=data["cost_matrix"],
            accumulated_cost=data["accumulated_cost"],
            warping_path=data["warping_path"],
            distance=data["distance"],
            rotations=data.get("rotations"),
            stored_transforms=stored_transforms,
        )

        log.info(f"Loaded DTW results from {path}")
        return result

    def set_cost_matrix(self, cost_matrix: torch.Tensor) -> None:
        """Set a precomputed cost matrix.

        Useful when you have already computed the pairwise distance matrix
        externally and want to skip recomputation.

        Parameters
        ----------
        cost_matrix : torch.Tensor
            Precomputed cost matrix of shape (n_metrics, len(x), len(y))
            or (len(x), len(y)).
        """
        self._cost_matrix = cost_matrix
        log.info("Set precomputed cost matrix")
