from pathlib import Path
from functools import partial
from copy import deepcopy
import logging
import os
import time


# Try to import MPI for parallel computation, but make it optional
try:
    from mpi4py import MPI
    hasmpi = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    hasmpi = False  # pragma: no cover
    MPI = None  # pragma: no cover

import torch

from . import distances
from . import downsampling
from . import utils
from . import cpd
from . import transforms
from .dataset import zRegPointCloud
from .validation import _validate_tensors
from .distances import DistanceMetric
from .types import StoredTransform, PairwiseResult


log = logging.getLogger(__name__)


__all__ = [
    "create_pairwise_distance_matrix",
    "create_pairwise_distance_matrix_given_rigid_rot",
    "StoredTransform",
    "PairwiseResult",
]


def create_pairwise_distance_matrix(
    x: dict[int, zRegPointCloud],
    y: dict[int, zRegPointCloud],
    window: int | None = None,
    normalize: bool = True,
    distance_metric: list[str | DistanceMetric] | str | DistanceMetric = "swd",
    distance_kwargs: list[dict] | dict | None = None,
    downsample_method: str | None = None,
    cpd_type: str | None = None,
    mpi_distribute: bool = False,
) -> PairwiseResult:
    """Create a pairwise distance matrix using the given parameters.

    Parameters
    ----------
    x : dict[int, zRegPointCloud]
        A dictionary containing the first set of point cloud data. The keys are integer indices, and the values
        are zRegPointCloud instances containing point cloud data (e.g., 'pos' for positions).
    y : dict[int, zRegPointCloud]
        A dictionary containing the second set of point cloud data. The keys are integer indices, and the values
        are zRegPointCloud instances containing point cloud data (e.g., 'pos' for positions).
    window : int | None, optional
        The window size to use for the distance matrix calculation. If None, no windowing is used (full matrix).
        By default, None.
    normalize : bool, optional
        Whether to normalize the point clouds before calculating the distance.
        By default, True.
    distance_metric : list[str | DistanceMetric] | str | DistanceMetric, optional
        The distance metric(s) to use for the calculation. Accepts string identifiers
        ('swd', 'euclidean', 'manhattan', 'minkowski', 'cpd', 'aswd', 'oswd', 'gswd', 'pswd')
        or any callable conforming to the `DistanceMetric` protocol
        (``zreg.distances.DistanceMetric``). Can be a single value or a list of values.
        By default, "swd".
    distance_kwargs : list[dict] | dict | None, optional
        Keyword arguments to pass to the distance function(s). If `distance_metric` is a list, this should be a list of
        dictionaries of the same length.
        By default, None.
    downsample_method : str | None, optional
        The downsampling method to use. If None, no downsampling is performed.
        By default, None.
    cpd_type : str | None, optional
        The type of Coherent Point Drift registration to perform. If None, no CPD is used.
        By default, None.
    mpi_distribute : bool, optional
        Whether to distribute the computation across multiple MPI processes. Requires `mpi4py`.
        By default, False.

    Returns
    -------
    PairwiseResult
        A dataclass containing:
            - cost_matrix (torch.Tensor): The pairwise distance matrix.
            - rotations (torch.Tensor | None): The rotations from rigid CPD registration, or None.
            - stored_transforms (dict[tuple[int, int], StoredTransform]): Stored CPD transforms
              keyed by (i, j) pair indices; empty dict when cpd_type is None.
    """
    _validate_tensors(x[0]["pos"], y[0]["pos"], names=["x[0]['pos']", "y[0]['pos']"])
    rank, size = 0, 1
    if mpi_distribute and hasmpi:
        comm_world = MPI.COMM_WORLD
        rank, size = comm_world.rank, comm_world.size

    # Sanitize the inputs and get the distance and downsampling functions
    distance_fns, downsample_method, downsample_fn = _sanitize_pairwise_distance_matrix(
        distance_kwargs=distance_kwargs,
        distance_metrics=distance_metric,
        downsample_method=downsample_method,
        x=x,
        y=y,
    )
    # distance_fn: list  # this is a list of callables / None (for cpd)

    # Get the number of samples in each set of data
    x_samples, y_samples = max(x), max(y)

    shape = (len(distance_fns), x_samples + 1, y_samples + 1)

    distance_matrix = torch.full(shape, torch.inf, dtype=x[0]["pos"].dtype, device=x[0]["pos"].device)

    # Calculate the number of distance elements to compute
    if window is not None:
        k = 1 + 2 * window
        n = x_samples
        num_dist_elems = int(n * k - (k * (k - 1)) / 2)
    else:
        num_dist_elems = int(x_samples * y_samples)

    # Set the logging frequency and intervals
    log_freq = 0.10
    log_intervals = torch.linspace(0, num_dist_elems, steps=int(1 / log_freq) + 1, dtype=torch.int)[1:]
    rots = []
    stored_transforms: dict[tuple[int, int], StoredTransform] = {}

    # Initialize the loop counter and timing dictionary
    full_counter = 0
    times = {
        "copy": [],
        "norm": [],
        "downsample": [],
        "cpd": [],
        "distance": [],
        "total": [],
    }

    for i in range(x_samples + 1):
        # Calculate the window boundaries
        if window is not None:
            window_min = i - window
            if window_min < 0:
                window_min = 0
            window_max = i + window
            if window_max > y_samples + 1:
                window_max = y_samples + 1
        else:
            window_min, window_max = 0, y_samples + 1

        # Iterate over the samples in the second set of data within the window
        for j in range(window_min, window_max):
            if full_counter % size != rank and mpi_distribute:
                distance_matrix[:, i, j] = 0.0
                full_counter += 1
                continue
            t0 = time.perf_counter()
            # copies to avoid overwriting...
            xi = deepcopy(x[i])
            yj = deepcopy(y[j])
            tc = time.perf_counter()
            times["copy"].append(tc - t0)

            # normalize the smaller point cloud to the largest
            src_min = src_max = tgt_min = tgt_max = None
            if normalize:
                # source = copy.deepcopy(pcs[0])
                downsampling.remove_outliers_knn(xi, inplace=True)
                downsampling.remove_outliers_knn(yj, inplace=True)

                # xi, yj = downsampling.random_down_sample(xi, yj)
                xi["pos"], (src_min, src_max) = utils.normalize_point_cloud(xi["pos"])
                yj["pos"], (tgt_min, tgt_max) = utils.normalize_point_cloud(yj["pos"])
                # xi["pos"], yj["pos"], _ = utils.normalize_to_pc_w_most_points(xi["pos"], yj["pos"])
            tn = time.perf_counter()
            times["norm"].append(tn - tc)

            # downsample the point could to be the same size
            xi, yj = downsample_fn(xi, yj)
            tdn = time.perf_counter()
            times["downsample"].append(tdn - tn)

            # do CPD registration to transform *yj*
            # this means that yj is the source and xi is the target
            cpd_metric = torch.inf
            if cpd_type is not None:
                tf_params = {"device": xi["pos"].device, "dtype": xi["pos"].dtype}
                if cpd_type == "nonrigid":
                    cpd_obj = cpd.NonRigidCPD(source=xi["pos"], use_color=False, log_freq=-1)
                elif cpd_type == "affine":
                    cpd_obj = cpd.AffineCPD(
                        source=xi["pos"], use_color=False, tf_init_params=tf_params, log_freq=-1
                    )
                else:  # "rigid"
                    cpd_obj = cpd.RigidCPD(
                        source=xi["pos"], use_color=False, tf_init_params=tf_params, log_freq=-1
                    )
                reg = cpd_obj.registration(yj["pos"], w=0.0, maxiter=1000, tol=1e-5)

                xi["pos"] = cpd_obj.transformation.transform(xi["pos"])
                if hasattr(reg.transformation, "rot"):
                    rots.append(reg.transformation.rot.unsqueeze(0))
                cpd_metric = reg.q
                # Store the transform and normalisation params for reuse in _build_aligned_cloud
                stored_transforms[(i, j)] = StoredTransform(
                    transform=reg.transformation,
                    src_min=src_min,
                    src_max=src_max,
                    tgt_min=tgt_min,
                    tgt_max=tgt_max,
                )
            tcpd = time.perf_counter()
            times["cpd"].append(tcpd - tdn)

            # distance calculations
            dists = []
            for fn in distance_fns:
                if fn is None:
                    dists.append(cpd_metric)
                    continue
                if hasattr(fn, "projs_history"):
                    # cleanup ASWD projection history file
                    fn.remove_history()
                dist = fn(xi["pos"], yj["pos"])
                if dist.numel() > 1:
                    dist = dist.mean()
                dists.append(dist)
            tdist = time.perf_counter()
            times["distance"].append(tdist - tcpd)

            for di in range(len(distance_fns)):
                distance_matrix[di, i, j] = dists[di]

            full_counter += 1  # noqa: E741
            tf = time.perf_counter()
            times["total"].append(tf - t0)

            if full_counter in log_intervals:
                tc = sum(times["copy"]) / float(len(times["copy"]))
                tn = sum(times["norm"]) / float(len(times["norm"]))
                tdn = sum(times["downsample"]) / float(len(times["downsample"]))
                tcpd = sum(times["cpd"]) / float(len(times["cpd"]))
                tdi = sum(times["distance"]) / float(len(times["distance"]))
                tt = sum(times["total"]) / float(len(times["total"]))
                log.info(
                    f"iteration {full_counter + 1}/{num_dist_elems + 1}: time: full: {tt:.4f}, copy: {tc:.4f}, "
                    f"norm: {tn:.4f}, downsample: {tdn:.4f}, cpd: {tcpd:.4f}, distance: {tdi:.4f}"
                )
            if full_counter == 1:
                log.debug("end of first iteration")

        # if l in log_intervals:
        if len(times["copy"]) == 0:
            continue
        tc = sum(times["copy"]) / float(len(times["copy"]))
        tn = sum(times["norm"]) / float(len(times["norm"]))
        tdn = sum(times["downsample"]) / float(len(times["downsample"]))
        tcpd = sum(times["cpd"][2:]) / float(len(times["cpd"][2:])) if len(times["cpd"]) > 2 else 0.0
        tdi = sum(times["distance"]) / float(len(times["distance"]))
        tt = sum(times["total"]) / float(len(times["total"]))

        # logging at the end of every row, can be removed without issue
        log.info(
            f"iteration {i + 1}/{x_samples + 1}: time: full: {tt:.4f}, copy: {tc:.4f}, "
            f"norm: {tn:.4f}, downsample: {tdn:.4f}, cpd: {tcpd:.4f}, distance: {tdi:.4f}"
        )
        # reset time counters
        times["copy"] = []
        times["norm"] = []
        times["downsample"] = []
        times["cpd"] = []
        times["distance"] = []
        times["total"] = []

        # sync up mpi things
        if mpi_distribute and hasmpi:
            tcomm = time.perf_counter()
            row = distance_matrix[:, i].cpu().numpy()
            gathered = comm_world.allgather(row)
            combined = sum(gathered)
            distance_matrix[:, i] = torch.tensor(combined, device=distance_matrix.device, dtype=distance_matrix.dtype)
            if rank == 0:
                log.debug("MPI allgather row %d: %.4f s", i, time.perf_counter() - tcomm)
    rotations = torch.cat(rots, dim=0) if len(rots) > 0 else None
    return PairwiseResult(
        cost_matrix=distance_matrix,
        rotations=rotations,
        stored_transforms=stored_transforms,
    )


def create_pairwise_distance_matrix_given_rigid_rot(
    x: dict[int, zRegPointCloud],
    y: dict[int, zRegPointCloud],
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: float = 1.0,
    window: int | None = None,
    normalize: bool = True,
    distance_metric: list[str] | str = "swd",
    distance_kwargs: list[dict] | dict | None = None,
    downsample_method: str | None = None,
    mpi_distribute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This function follows the normal pairwise distance matrix function closely, but uses a fixed rotation
    # the other functionality is the same.

    rank, size = 0, 1
    if mpi_distribute and hasmpi:
        comm_world = MPI.COMM_WORLD
        rank, size = comm_world.rank, comm_world.size

    # Sanitize the inputs and get the distance and downsampling functions
    distance_fns, downsample_method, downsample_fn = _sanitize_pairwise_distance_matrix(
        distance_kwargs=distance_kwargs,
        distance_metrics=distance_metric,
        downsample_method=downsample_method,
        x=x,
        y=y,
    )
    # distance_fns: list  # this is a list of callables / None (for cpd)

    # Get the number of samples in each set of data
    x_samples, y_samples = max(x), max(y)

    shape = (len(distance_fns), x_samples + 1, y_samples + 1)

    distance_matrix = torch.full(shape, torch.inf, dtype=x[0]["pos"].dtype, device=x[0]["pos"].device)

    # Calculate the number of distance elements to compute
    if window is not None:
        k = 1 + 2 * window
        n = x_samples
        num_dist_elems = int(n * k - (k * (k - 1)) / 2)
    else:
        num_dist_elems = int(x_samples * y_samples)

    # Set the logging frequency and intervals
    log_freq = 0.10
    log_intervals = torch.linspace(0, num_dist_elems, steps=int(1 / log_freq) + 1, dtype=torch.int)[1:]

    # Initialize the loop counter and timing dictionary
    full_counter = 0
    times = {
        "copy": [],
        "norm": [],
        "downsample": [],
        "rot": [],
        "distance": [],
        "total": [],
    }

    trans = transforms.RigidTransformation(
        rot=rotation, t=translation, scale=scale, dtype=x[0]["pos"].dtype, device=x[0]["pos"].device
    )

    for i in range(x_samples + 1):
        # Calculate the window boundaries
        if window is not None:
            window_min = i - window
            if window_min < 0:
                window_min = 0
            window_max = i + window
            if window_max > y_samples + 1:
                window_max = y_samples + 1
        else:
            window_min, window_max = 0, y_samples + 1

        # Iterate over the samples in the second set of data within the window
        for j in range(window_min, window_max):
            if full_counter % size != rank and mpi_distribute:
                distance_matrix[:, i, j] = 0.0
                full_counter += 1
                continue
            t0 = time.perf_counter()
            # copies to avoid overwriting...
            xi = deepcopy(x[i])
            yj = deepcopy(y[j])
            tc = time.perf_counter()
            # normalize the smaller point cloud to the largest
            if normalize:
                # source = copy.deepcopy(pcs[0])
                downsampling.remove_outliers_knn(xi, inplace=True)
                downsampling.remove_outliers_knn(yj, inplace=True)

                # xi, yj = downsampling.random_down_sample(xi, yj)
                xi["pos"], _ = utils.normalize_point_cloud(xi["pos"])
                yj["pos"], _ = utils.normalize_point_cloud(yj["pos"])
                # xi["pos"], yj["pos"], _ = utils.normalize_to_pc_w_most_points(xi["pos"], yj["pos"])
            tn = time.perf_counter()
            # downsample the point could to be the same size
            xi, yj = downsample_fn(xi, yj)
            tdn = time.perf_counter()

            # CHANGE FROM OTHER THINGS ----------
            # do rotation here
            xi["pos"] = trans.transform(xi["pos"])
            trot = time.perf_counter()

            # distance calculations
            dists = []
            for fn in distance_fns:
                if fn is None:
                    continue
                if hasattr(fn, "projs_history"):
                    # cleanup ASWD projection history file
                    fn.remove_history()
                dist = fn(xi["pos"], yj["pos"])
                if dist.numel() > 1:
                    dist = dist.mean()
                dists.append(dist)
            tdist = time.perf_counter()

            for di in range(len(distance_fns)):
                distance_matrix[di, i, j] = dists[di]

            full_counter += 1  # noqa: E741
            tf = time.perf_counter()
            times["copy"].append(tc - t0)
            times["norm"].append(tn - tc)
            times["downsample"].append(tdn - tn)
            times["rot"].append(trot - tdn)
            times["distance"].append(tdist - trot)
            times["total"].append(tf - t0)

            if full_counter in log_intervals:
                tc = sum(times["copy"]) / float(len(times["copy"]))
                tn = sum(times["norm"]) / float(len(times["norm"]))
                tdn = sum(times["downsample"]) / float(len(times["downsample"]))
                trt = sum(times["rot"]) / float(len(times["rot"]))
                tdi = sum(times["distance"]) / float(len(times["distance"]))
                tt = sum(times["total"]) / float(len(times["total"]))
                log.info(
                    f"iteration {full_counter + 1}/{num_dist_elems + 1}: time: full: {tt:.4f}, copy: {tc:.4f}, "
                    f"norm: {tn:.4f}, downsample: {tdn:.4f}, rot: {trt:.4f}, distance: {tdi:.4f}"
                )
            if full_counter == 1:
                log.debug("end of first iteration")

        # if l in log_intervals:
        if len(times["copy"]) == 0:
            continue
        tc = sum(times["copy"]) / float(len(times["copy"]))
        tn = sum(times["norm"]) / float(len(times["norm"]))
        tdn = sum(times["downsample"]) / float(len(times["downsample"]))
        trt = sum(times["rot"][2:]) / float(len(times["rot"][2:])) if len(times["rot"]) > 2 else 0.0
        tdi = sum(times["distance"]) / float(len(times["distance"]))
        tt = sum(times["total"]) / float(len(times["total"]))
        log.info(
            f"iteration {i + 1}/{x_samples + 1}: time: full: {tt:.4f}, copy: {tc:.4f}, "
            f"norm: {tn:.4f}, downsample: {tdn:.4f}, rot: {trt:.4f}, distance: {tdi:.4f}"
        )
        # reset time counters
        times["copy"] = []
        times["norm"] = []
        times["downsample"] = []
        times["rot"] = []
        times["distance"] = []
        times["total"] = []

        # sync up mpi things
        if mpi_distribute and hasmpi:
            tcomm = time.perf_counter()
            row = distance_matrix[:, i].cpu().numpy()
            gathered = comm_world.allgather(row)
            combined = sum(gathered)
            distance_matrix[:, i] = torch.tensor(combined, device=distance_matrix.device, dtype=distance_matrix.dtype)
            if rank == 0:
                log.debug("MPI allgather row %d: %.4f s", i, time.perf_counter() - tcomm)
            # print(distance_matrix)
    # if len(rots) > 0:
    #     rots = torch.cat(rots, dim=0)
    return distance_matrix


def _sanitize_pairwise_distance_matrix(distance_kwargs, distance_metrics, downsample_method, x, y):
    """Resolve distance metric strings to callable objects.

    This function implements string-to-callable dispatch for distance metrics.
    The returned callables satisfy the DistanceMetric protocol interface:

        callable(x: Tensor, y: Tensor, **kwargs) -> Tensor

    Supported Metrics
    -----------------
    String      | Returns                          | Notes
    ------------|----------------------------------|----------------------------------
    "swd"       | SlicedWassersteinDistance        | Requires downsampling
    "aswd"      | AdaptiveSlicedWassersteinDistance| Requires downsampling
    "oswd"      | OrthogonalSlicedWassersteinDistance| Requires downsampling
    "gswd"      | GeneralisedSlicedWassersteinDistance| Requires downsampling
    "pswd"      | ProjectedWassersteinDistance     | Requires downsampling
    "euclidean" | partial(euclidean_distance, ...) | No downsampling required
    "manhattan" | partial(manhattan_distance, ...) | No downsampling required
    "minkowski" | partial(minkowski_distance, ...) | No downsampling required
    "cpd"       | None (uses CPD q metric)         | Special case

    Parameters
    ----------
    distance_kwargs : list[dict] | dict | None
        Keyword arguments for each distance metric.
    distance_metrics : list[str] | str
        String identifiers for distance metrics.
    downsample_method : str | None
        Downsampling method identifier.
    x, y : dict[int, zRegPointCloud]
        Point cloud dictionaries (used for device inference).

    Returns
    -------
    distance_fns : list[Callable | None]
        List of distance callables (None for cpd).
    downsample_method : str | None
        Resolved downsample method.
    downsample_fn : Callable
        Downsampling function.

    Raises
    ------
    ValueError
        If invalid distance metric string provided.
    RuntimeError
        If SWD metric used without downsampling method.
    """
    if not isinstance(distance_metrics, list):
        distance_metrics = [
            distance_metrics,
        ]  # noqa
    else:
        # need to copy the list to make sure we can run this iteratively without crashes
        distance_metrics = deepcopy(distance_metrics)
    if not isinstance(distance_kwargs, list):
        distance_kwargs = [
            distance_kwargs,
        ]  # noqa
    else:
        # need to copy the list to make sure we can run this iteratively without crashes
        distance_kwargs = deepcopy(distance_kwargs)

    if len(distance_kwargs) != len(distance_metrics) and distance_kwargs[0] is not None:
        raise RuntimeError(f"len distance kwargs != len distance metrics!: {distance_kwargs} v {distance_metrics}")

    if distance_kwargs[0] is None:
        distance_kwargs = [
            None,
        ] * len(distance_metrics)  # noqa

    for c, dist in enumerate(distance_metrics):
        dist_kwargs = distance_kwargs[c]
        if dist_kwargs is None:
            dist_kwargs = {}

        if callable(dist):
            # Caller supplied a conforming callable (function, bound method, or nn.Module).
            # Pass it through directly without string dispatch or downsampling requirement.
            distance_fn = dist
            distance_metrics[c] = distance_fn
            continue

        if dist == "swd":
            log.info("Using Sliced Wasserstein Distance for distance metric")
            # set default kwargs
            defaults = [
                ["device", x[0]["pos"].device],
                ["num_projs", 50],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            distance_fn = distances.SlicedWassersteinDistance(**dist_kwargs)
        elif dist == "aswd":
            log.info("Using Adaptive Sliced Wasserstein Distance for distance metric")
            defaults = [
                ["device", x[0]["pos"].device],
                ["max_slices", 100],
                ["init_projs", 50],
                ["step_projs", 25],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            if Path("projs_history.txt").exists():
                # remove the proj history...need to do this after every distance
                try:
                    os.remove("projs_history.txt")
                except FileNotFoundError:
                    pass
            distance_fn = distances.AdaptiveSlicedWassersteinDistance(**dist_kwargs)
        elif dist == "oswd":
            log.info("Using Orthogonal Wasserstein Distance for distance metric")
            # set default kwargs
            defaults = [
                ["device", x[0]["pos"].device],
                ["num_projs", 50],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            distance_fn = distances.OrthogonalSlicedWassersteinDistance(**dist_kwargs)
        elif dist == "gswd":
            log.info("Using Generalised Sliced Wasserstein Distance for distance metric")
            # set default kwargs
            defaults = [
                ["device", x[0]["pos"].device],
                ["num_projs", 50],
                ["degree", 2.0],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            distance_fn = distances.GeneralisedSlicedWassersteinDistance(**dist_kwargs)
        elif dist == "pswd":
            log.info("Using Projected Wasserstein Distance for distance metric")
            # set default kwargs
            defaults = [
                ["device", x[0]["pos"].device],
                ["num_projs", 50],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            distance_fn = distances.ProjectedWassersteinDistance(**dist_kwargs)
        elif dist == "euclidean":
            log.info("Using Euclidean Distance for distance metric")
            distance_fn = partial(distances.euclidean_distance, **dist_kwargs)
        elif dist == "manhattan":
            log.info("Using Manhatten Distance for distance metric")
            distance_fn = partial(distances.manhattan_distance, **dist_kwargs)
        elif dist == "minkowski":
            defaults = [
                ["p", 3],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            log.info(f"Using Minkowski Distance for distance metric with p={dist_kwargs['p']}")
            distance_fn = partial(distances.minkowski_distance, **dist_kwargs)
        elif dist == "cpd":
            distance_fn = None
        else:
            raise ValueError(f"Invalid distance function: {dist}")
        distance_metrics[c] = distance_fn
        if dist not in ["euclidean", "manhattan", "minkowski", "cpd"] and downsample_method is None:
            raise RuntimeError("with SWD methods, need to use a downsampling method")

    log.info(f"Using Downsampling method: {downsample_method}")
    if downsample_method is None:
        downsample_fn = lambda l1, l2: (l1, l2)  # noqa: E731
    elif downsample_method == "random":
        downsample_fn = downsampling.random_down_sample
    elif downsample_method == "uniform":
        downsample_fn = downsampling.uniform_down_sample
    elif downsample_method.startswith("farthest"):
        # TODO: add partial for kwargs?

        downsample_fn = partial(downsampling.farthest_point_down_sample, points=None, use_precomputed_indexes=False)
    else:
        raise ValueError("Invalid downsampling function")

    return distance_metrics, downsample_method, downsample_fn
