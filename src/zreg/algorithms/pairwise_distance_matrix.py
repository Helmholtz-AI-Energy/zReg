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

from .. import distance_metrics as distances
from ..preprocessing import downsampling
from .. import utils
from . import cpd
from ..core import transforms
from ..core.dataset import zRegPointCloud
from ..utils.validation import _validate_tensors
from ..distance_metrics import DistanceMetric
from ..core.types import StoredTransform, PairwiseResult


log = logging.getLogger(__name__)


# CPD variants usable as a DTW penalty. Constrained non-rigid CPD needs per-pair
# correspondence indices (idx_source/idx_target) and is therefore not supported here.
_VALID_CPD_TYPES = (None, "rigid", "affine", "nonrigid")


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
        (``zreg.distance_metrics.DistanceMetric``). Can be a single value or a list of values.
        By default, "swd".
    distance_kwargs : list[dict] | dict | None, optional
        Keyword arguments to pass to the distance function(s). If `distance_metric` is a list, this should be a list of
        dictionaries of the same length.
        By default, None.
    downsample_method : str | None, optional
        The downsampling method to use. If None, no downsampling is performed.
        By default, None.
    cpd_type : str | None, optional
        The type of Coherent Point Drift registration to perform: one of None, "rigid",
        "affine" or "nonrigid". If None, no CPD is used. With ``distance_metric="cpd"`` the
        cost of a pair is the converged CPD sigma2.
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
              keyed by (i, j) pair indices; empty dict when cpd_type is None. Entries are also
              never stored when cpd_type == "nonrigid" (unbounded-memory guard — see inline
              comment at the storage site) — callers needing a nonrigid transform for a specific
              pair must recompute it themselves.

    Raises
    ------
    ValueError
        If ``cpd_type`` is not one of None, "rigid", "affine", "nonrigid" (raised before
        any registration runs; constrained non-rigid CPD is not supported in DTW sweeps),
        or if ``distance_metric`` includes "cpd" while ``cpd_type`` is None.
    """
    # Validate cpd_type before any work. Deterministic on the arguments, so every MPI rank
    # raises identically (no rank divergence).
    if cpd_type not in _VALID_CPD_TYPES:
        if cpd_type in ("nonrigid_constrained", "constrained_nonrigid"):
            raise ValueError(
                f"cpd_type={cpd_type!r} is not supported: constrained non-rigid CPD needs "
                "correspondence indices (idx_source/idx_target) and cannot be used as a DTW "
                f"cpd_type. Valid values: {_VALID_CPD_TYPES}."
            )
        raise ValueError(f"cpd_type must be one of {_VALID_CPD_TYPES}; got {cpd_type!r}")

    # Use first available frame from each dict so callers with non-zero-based keys don't crash
    # (WR-01: x[0] / y[0] raised KeyError when keys did not include 0).
    _x0 = next(iter(x.values()))
    _y0 = next(iter(y.values()))
    _validate_tensors(_x0["pos"], _y0["pos"], names=["x[first]['pos']", "y[first]['pos']"])
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

    # Guard: distance_metric='cpd' uses the converged CPD sigma2 as the distance value (D-02).
    # If cpd_type=None, no CPD runs and cpd_metric stays torch.inf, making the entire cost
    # matrix inf and causing DTW backtracing to fail with a misleading "window too tight" error.
    if cpd_type is None and any(fn is None for fn in distance_fns):
        raise ValueError(
            "distance_metric='cpd' requires cpd_type to be set ('rigid', 'affine', or "
            "'nonrigid'), but cpd_type=None. Without CPD registration the cpd distance "
            "is always inf, producing an all-inf cost matrix and making DTW backtracing "
            "impossible. Set cpd_type to match the desired CPD registration type."
        )

    # Sweep frames by sorted key position (DIST-03): matrix indices and stored_transforms
    # keys are positional (i, j), so non-zero-based or gapped key layouts work and match
    # the DTW warping path. n_x / n_y are the last positions.
    x_keys, y_keys = sorted(x), sorted(y)
    n_x, n_y = len(x_keys) - 1, len(y_keys) - 1

    shape = (len(distance_fns), n_x + 1, n_y + 1)

    distance_matrix = torch.full(shape, torch.inf, dtype=_x0["pos"].dtype, device=_x0["pos"].device)

    # Calculate the number of distance elements to compute
    if window is not None:
        k = 1 + 2 * window
        n = n_x
        num_dist_elems = int(n * k - (k * (k - 1)) / 2)
    else:
        num_dist_elems = int(n_x * n_y)

    # Set the logging frequency and intervals
    log_freq = 0.10
    log_intervals = torch.linspace(0, num_dist_elems, steps=int(1 / log_freq) + 1, dtype=torch.int)[1:]
    rots = []
    stored_transforms: dict[tuple[int, int], StoredTransform] = {}

    # Initialize the loop counter and timing dictionary
    full_counter = 0
    # First local per-pair failure under MPI (WR-01); raised on every rank after the row gather
    pair_error: str | None = None
    pair_exc: Exception | None = None
    times = {
        "copy": [],
        "norm": [],
        "downsample": [],
        "cpd": [],
        "distance": [],
        "total": [],
    }

    for i in range(n_x + 1):
        # Calculate the window boundaries
        if window is not None:
            window_min = i - window
            if window_min < 0:
                window_min = 0
            # +1 so that range(window_min, window_max) includes j = i + window
            window_max = min(i + window + 1, n_y + 1)
        else:
            window_min, window_max = 0, n_y + 1

        # Iterate over the samples in the second set of data within the window
        for j in range(window_min, window_max):
            owner = full_counter % size
            full_counter += 1
            # Pairs owned by another rank, and the rest of a row after a local pair
            # failure, contribute 0.0 to the row sum-combine.
            if (mpi_distribute and owner != rank) or pair_error is not None:
                distance_matrix[:, i, j] = 0.0
                continue
            try:
                t0 = time.perf_counter()
                # copies to avoid overwriting...
                xi = deepcopy(x[x_keys[i]])
                yj = deepcopy(y[y_keys[j]])
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
                    elif cpd_type == "rigid":
                        cpd_obj = cpd.RigidCPD(
                            source=xi["pos"], use_color=False, tf_init_params=tf_params, log_freq=-1
                        )
                    reg = cpd_obj.registration(yj["pos"], w=0.0, maxiter=1000, tol=1e-5)

                    xi["pos"] = cpd_obj.transformation.transform(xi["pos"])
                    if hasattr(reg.transformation, "rot"):
                        rots.append(reg.transformation.rot.unsqueeze(0))
                    cpd_metric = _cpd_dtw_cost(reg)
                    # Store the transform and normalisation params for reuse in _build_aligned_cloud.
                    # Only store when normalize=True: when normalize=False, src_min/src_max/tgt_min/
                    # tgt_max remain None and the reuse path in _build_aligned_cloud would apply
                    # normalisation that was never done in Step 1, corrupting the output (CR-02).
                    #
                    # cpd_type == "nonrigid" is excluded from caching: NonRigidTransformation
                    # retains a dense (n_points, n_points) RBF kernel matrix (see
                    # zreg.core.transforms.nonrigid.NonRigidTransformation.g). With real, full-resolution
                    # point clouds (tens of thousands of points/frame) and a windowed sweep touching
                    # thousands of (i, j) pairs, retaining one of these per pair grows this dict
                    # unboundedly into the hundreds of GB, exhausting memory/swap well before the
                    # sweep completes. _build_aligned_cloud already has a tested, correctness-
                    # preserving fallback for missing cache entries (D-10: fresh CPD from raw data,
                    # eval/stages/alignment.py) — losing the nonrigid cache only means that fallback
                    # runs for the (bounded, ~len(target)) frames actually selected by the DTW warp
                    # path, instead of reusing a precomputed transform.
                    if normalize and cpd_type != "nonrigid":
                        stored_transforms[(i, j)] = StoredTransform(
                            transform=reg.transformation,
                            src_min=src_min,
                            src_max=src_max,
                            tgt_min=tgt_min,
                            tgt_max=tgt_max,
                        )
                    del cpd_obj, reg  # release GPU tensors held by CPD internals
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
            except Exception as exc:  # noqa: BLE001 - re-raised on every rank after the row gather
                if not (mpi_distribute and hasmpi):
                    raise
                # Raising here would leave the other ranks blocked in this row's allgather
                # (WR-01). Record the failure; _allgather_row shares it with every rank.
                pair_error, pair_exc = f"pair ({i}, {j}): {type(exc).__name__}: {exc}", exc
                distance_matrix[:, i, j] = 0.0

        # Per-row timing log, only when this rank computed pairs in the row. No `continue`
        # here: the per-row allgather below must be reached by every rank (DIST-01).
        if times["total"]:  # "total" is appended only by a completed pair (WR-01)
            tc = sum(times["copy"]) / float(len(times["copy"]))
            tn = sum(times["norm"]) / float(len(times["norm"]))
            tdn = sum(times["downsample"]) / float(len(times["downsample"]))
            tcpd = sum(times["cpd"][2:]) / float(len(times["cpd"][2:])) if len(times["cpd"]) > 2 else 0.0
            tdi = sum(times["distance"]) / float(len(times["distance"]))
            tt = sum(times["total"]) / float(len(times["total"]))

            # logging at the end of every row, can be removed without issue
            log.info(
                f"iteration {i + 1}/{n_x + 1}: time: full: {tt:.4f}, copy: {tc:.4f}, "
                f"norm: {tn:.4f}, downsample: {tdn:.4f}, cpd: {tcpd:.4f}, distance: {tdi:.4f}"
            )
            # reset time counters
            times["copy"] = []
            times["norm"] = []
            times["downsample"] = []
            times["cpd"] = []
            times["distance"] = []
            times["total"] = []

        # Release CUDA allocator cache after each row so del'd CPD tensors are freed promptly
        if cpd_type is not None and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()

        # sync up mpi things: every rank joins the row collective, even without local pairs
        if mpi_distribute and hasmpi:
            failures = _allgather_row(comm_world, distance_matrix, i, rank, error=pair_error)
            _raise_pair_failures(failures, pair_exc)
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

    # Guard (mirrors create_pairwise_distance_matrix): this function runs no CPD, so a
    # "cpd" metric has no value to report (previously an IndexError mid-sweep, DIST-03).
    if any(fn is None for fn in distance_fns):
        raise ValueError(
            "distance_metric='cpd' is not supported by create_pairwise_distance_matrix_given_rigid_rot "
            "(it runs no CPD); use create_pairwise_distance_matrix with cpd_type set"
        )

    # Sweep frames by sorted key position (DIST-03): matrix indices and stored_transforms
    # keys are positional (i, j), so non-zero-based or gapped key layouts work and match
    # the DTW warping path. n_x / n_y are the last positions.
    x_keys, y_keys = sorted(x), sorted(y)
    n_x, n_y = len(x_keys) - 1, len(y_keys) - 1

    shape = (len(distance_fns), n_x + 1, n_y + 1)

    # Use first available frame for dtype/device (WR-01: x[0] crashes on non-zero-based keys).
    _x0_rigid = next(iter(x.values()))
    distance_matrix = torch.full(shape, torch.inf, dtype=_x0_rigid["pos"].dtype, device=_x0_rigid["pos"].device)

    # Calculate the number of distance elements to compute
    if window is not None:
        k = 1 + 2 * window
        n = n_x
        num_dist_elems = int(n * k - (k * (k - 1)) / 2)
    else:
        num_dist_elems = int(n_x * n_y)

    # Set the logging frequency and intervals
    log_freq = 0.10
    log_intervals = torch.linspace(0, num_dist_elems, steps=int(1 / log_freq) + 1, dtype=torch.int)[1:]

    # Initialize the loop counter and timing dictionary
    full_counter = 0
    # First local per-pair failure under MPI (WR-01); raised on every rank after the row gather
    pair_error: str | None = None
    pair_exc: Exception | None = None
    times = {
        "copy": [],
        "norm": [],
        "downsample": [],
        "rot": [],
        "distance": [],
        "total": [],
    }

    trans = transforms.RigidTransformation(
        rot=rotation, t=translation, scale=scale, dtype=_x0_rigid["pos"].dtype, device=_x0_rigid["pos"].device
    )

    for i in range(n_x + 1):
        # Calculate the window boundaries
        if window is not None:
            window_min = i - window
            if window_min < 0:
                window_min = 0
            # +1 so that range(window_min, window_max) includes j = i + window
            window_max = min(i + window + 1, n_y + 1)
        else:
            window_min, window_max = 0, n_y + 1

        # Iterate over the samples in the second set of data within the window
        for j in range(window_min, window_max):
            owner = full_counter % size
            full_counter += 1
            # Pairs owned by another rank, and the rest of a row after a local pair
            # failure, contribute 0.0 to the row sum-combine.
            if (mpi_distribute and owner != rank) or pair_error is not None:
                distance_matrix[:, i, j] = 0.0
                continue
            try:
                t0 = time.perf_counter()
                # copies to avoid overwriting...
                xi = deepcopy(x[x_keys[i]])
                yj = deepcopy(y[y_keys[j]])
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
            except Exception as exc:  # noqa: BLE001 - re-raised on every rank after the row gather
                if not (mpi_distribute and hasmpi):
                    raise
                # Raising here would leave the other ranks blocked in this row's allgather
                # (WR-01). Record the failure; _allgather_row shares it with every rank.
                pair_error, pair_exc = f"pair ({i}, {j}): {type(exc).__name__}: {exc}", exc
                distance_matrix[:, i, j] = 0.0

        # Per-row timing log, only when this rank computed pairs in the row. No `continue`
        # here: the per-row allgather below must be reached by every rank (DIST-01).
        if times["copy"]:
            tc = sum(times["copy"]) / float(len(times["copy"]))
            tn = sum(times["norm"]) / float(len(times["norm"]))
            tdn = sum(times["downsample"]) / float(len(times["downsample"]))
            trt = sum(times["rot"][2:]) / float(len(times["rot"][2:])) if len(times["rot"]) > 2 else 0.0
            tdi = sum(times["distance"]) / float(len(times["distance"]))
            tt = sum(times["total"]) / float(len(times["total"]))
            log.info(
                f"iteration {i + 1}/{n_x + 1}: time: full: {tt:.4f}, copy: {tc:.4f}, "
                f"norm: {tn:.4f}, downsample: {tdn:.4f}, rot: {trt:.4f}, distance: {tdi:.4f}"
            )
            # reset time counters
            times["copy"] = []
            times["norm"] = []
            times["downsample"] = []
            times["rot"] = []
            times["distance"] = []
            times["total"] = []

        # sync up mpi things: every rank joins the row collective, even without local pairs
        if mpi_distribute and hasmpi:
            failures = _allgather_row(comm_world, distance_matrix, i, rank, error=pair_error)
            _raise_pair_failures(failures, pair_exc)
    # if len(rots) > 0:
    #     rots = torch.cat(rots, dim=0)
    return distance_matrix


def _allgather_row(
    comm, distance_matrix: torch.Tensor, i: int, rank: int, error: str | None = None
) -> list[tuple[int, str]]:
    """Combine row ``i`` of an MPI-distributed cost matrix across all ranks, in place.

    ``allgather`` is a collective and collectives match by call order, so every rank
    must call this helper for every row, in the same order -- also for rows in which it
    computed no pair. Skipping it on one rank pairs that rank's next call with another
    rank's call for a different row (silently wrong matrix) or blocks forever (DIST-01).

    The sum-combine is exact: each in-window entry is computed by exactly one rank and
    is 0.0 on every other rank, and out-of-window entries are inf on all ranks.

    Each rank also contributes its first per-pair failure (or None), so every rank learns
    of a failure on any rank in the same collective and can raise in lock-step instead
    of leaving the other ranks blocked in the next row's allgather (WR-01).

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator (``MPI.COMM_WORLD``).
    distance_matrix : torch.Tensor
        Cost matrix of shape ``(n_metrics, n_x, n_y)``; row ``i`` is overwritten with
        the combined row.
    i : int
        Row (x frame position) to combine.
    rank : int
        Rank of the caller (used only for the debug timing log).
    error : str | None, optional
        Description of this rank's first failed pair, or None. By default, None.

    Returns
    -------
    list[tuple[int, str]]
        ``(rank, error)`` for every rank that reported a failure; empty if none did.
        Identical on every rank.
    """
    tcomm = time.perf_counter()
    row = distance_matrix[:, i].cpu().numpy()
    gathered = comm.allgather((row, error))
    combined = sum(r for r, _ in gathered)
    distance_matrix[:, i] = torch.tensor(combined, device=distance_matrix.device, dtype=distance_matrix.dtype)
    if rank == 0:
        log.debug("MPI allgather row %d: %.4f s", i, time.perf_counter() - tcomm)
    return [(r, err) for r, (_, err) in enumerate(gathered) if err is not None]


def _raise_pair_failures(failures: list[tuple[int, str]], local_exc: Exception | None) -> None:
    """Raise on every rank if any rank reported a failed pair (WR-01).

    Parameters
    ----------
    failures : list[tuple[int, str]]
        ``(rank, error)`` pairs returned by :func:`_allgather_row`.
    local_exc : Exception | None
        This rank's own exception, chained as ``__cause__`` when present.

    Raises
    ------
    RuntimeError
        If ``failures`` is non-empty.
    """
    if not failures:
        return
    detail = "; ".join(f"rank {r}: {err}" for r, err in failures)
    raise RuntimeError(f"MPI-distributed pairwise sweep failed on {len(failures)} rank(s): {detail}") from local_exc


def _cpd_dtw_cost(reg: cpd.MstepResult) -> torch.Tensor:
    """Return the DTW cost of one CPD registration: the converged sigma2.

    sigma2 is the P-weighted mean squared residual per point and coordinate of the
    converged registration. It is >= 0, ~0 at a perfect fit, monotone in fit quality and
    independent of the point count, which makes it a well-behaved DTW local cost.

    ``reg.q`` is deliberately NOT used: the correct Myronenko & Song objective
    ``q = N_P*D/2*(1 + log sigma2)`` is negative for sigma2 < 1/e and extensive in N, and
    negative or offset local costs bias the DTW dynamic programme towards long paths (D-02).

    A non-finite sigma2 is passed through unchanged rather than raised here. Exceptions
    raised while computing a pair (CPD, normalisation, distance) are propagated in an
    MPI-distributed sweep by the per-row gather: every rank raises a ``RuntimeError``
    naming the failed pair (see ``_allgather_row`` / ``_raise_pair_failures``).

    Parameters
    ----------
    reg : cpd.MstepResult
        Result returned by ``registration()`` of a rigid, affine or non-rigid CPD.

    Returns
    -------
    torch.Tensor
        Scalar tensor ``max(reg.sigma2, 0)``.
    """
    return torch.as_tensor(reg.sigma2).clamp_min(0.0)


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
    "maxswd"    | MaxSlicedWassersteinDistance     | Requires downsampling
    "euclidean" | partial(euclidean_distance, ...) | No downsampling required
    "manhattan" | partial(manhattan_distance, ...) | No downsampling required
    "minkowski" | partial(minkowski_distance, ...) | No downsampling required
    "cpd"       | None (uses CPD sigma2)           | Special case

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
    # Use first available frame for device references (WR-01: x[0] crashes on non-zero-based keys).
    _x0_san = next(iter(x.values()))
    if not isinstance(distance_metrics, list):
        distance_metrics = [
            distance_metrics,
        ]  # noqa
    else:
        # need to copy the list to make sure we can run this iteratively without crashes
        distance_metrics = deepcopy(distance_metrics)
    # Normalise distance_kwargs to one entry per metric (DIST-03). None and [None] are the
    # only broadcast forms; dicts are copied so the defaults filled in below never mutate
    # the caller's kwargs; a leading None no longer discards later entries or disables the
    # length check.
    if distance_kwargs is None or (isinstance(distance_kwargs, list) and distance_kwargs == [None]):
        distance_kwargs = [None] * len(distance_metrics)
    elif isinstance(distance_kwargs, dict):
        distance_kwargs = [dict(distance_kwargs)]
    elif isinstance(distance_kwargs, list):
        distance_kwargs = [None if kw is None else dict(kw) for kw in distance_kwargs]
    else:
        distance_kwargs = [distance_kwargs]

    if len(distance_kwargs) != len(distance_metrics):
        raise RuntimeError(f"len distance kwargs != len distance metrics!: {distance_kwargs} v {distance_metrics}")

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
                ["device", _x0_san["pos"].device],
                ["num_projs", 50],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            distance_fn = distances.SlicedWassersteinDistance(**dist_kwargs)
        elif dist == "aswd":
            log.info("Using Adaptive Sliced Wasserstein Distance for distance metric")
            defaults = [
                ["device", _x0_san["pos"].device],
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
                ["device", _x0_san["pos"].device],
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
                ["device", _x0_san["pos"].device],
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
                ["device", _x0_san["pos"].device],
                ["num_projs", 50],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            distance_fn = distances.ProjectedWassersteinDistance(**dist_kwargs)
        elif dist == "maxswd":
            log.info("Using Max Sliced Wasserstein Distance for distance metric")
            # set default kwargs
            defaults = [
                ["device", _x0_san["pos"].device],
                ["max_sw_num_iters", 100],
            ]
            for kw, val in defaults:
                if kw not in dist_kwargs:
                    dist_kwargs[kw] = val
            distance_fn = distances.MaxSlicedWassersteinDistance(**dist_kwargs)
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
