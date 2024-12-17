from pathlib import Path
from functools import partial
from copy import deepcopy
import logging
import os
import time

from typing import Union

from mpi4py import MPI

import torch

from . import distances
from . import downsampling
from . import utils
from . import cpd


log = logging.getLogger(__name__)


__all__ = ["create_dtw_matrix"]


def create_dtw_matrix(
    x: dict,
    y: dict,
    window: int = None,
    normalize: bool = True,
    distance_metric: Union[list, str] = "swd",
    distance_kwargs: Union[list, dict] = None,
    downsample_method: str = None,
    cpd_type: str = None,
    mpi_distribute: bool = False,
    save_filename: str = None,
) -> torch.Tensor:
    """Create a DTW matrix using the given parameters.

    Parameters
    ----------
    x : dict
        A dictionary containing the first set of data.
    y : dict
        A dictionary containing the second set of data.
    window : int, optional
        The window size to use for the DTW calculation, by default None
    normalize : bool, optional
        Whether to normalize the data before calculating the DTW distance, by default True
    distance_metric : str, optional
        The distance metric to use for the DTW calculation, by default "swd"
    distance_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the distance function, by default None
    downsample_method : str, optional
        The downsampling method to use, by default None

    Returns
    -------
    torch.Tensor
        The DTW matrix.
    """
    rank, size = 0, 1
    if mpi_distribute:
        comm_world = MPI.COMM_WORLD
        rank, size = comm_world.rank, comm_world.size

    # Sanitize the inputs and get the distance and downsampling functions
    distance_fns, downsample_method, downsample_fn = _sanitize_dtw_matrix(
        distance_kwargs=distance_kwargs,
        distance_metrics=distance_metric,
        downsample_method=downsample_method,
        x=x,
        y=y,
    )
    # distance_fn: list  # this is a list of callables / None (for cpd)

    # # Precompute the farthest point downsampling if necessary
    # if downsample_method.startswith("farthest"):
    #     log.debug("Starting precompute for farthest points for downsampling for all time series points")
    #     t0 = time.perf_counter()
    #     for k in x:
    #         if "fps-idx" not in x[k]:
    #             x[k] = downsampling.precompute_fps(x[k])
    #     for k in y:
    #         if "fps-idx" not in y[k]:
    #             y[k] = downsampling.precompute_fps(y[k])
    #     log.debug(f"Precomute time required: {time.perf_counter() - t0}")

    # Get the number of samples in each set of data
    x_samples, y_samples = max(x), max(y)

    shape = (len(distance_fns), x_samples + 1, y_samples + 1)

    dtw_matrix = torch.full(shape, torch.inf, dtype=x[0]["pos"].dtype, device=x[0]["pos"].device)

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
    full_counter = 0  # noqa: E741
    times = {
        "copy": [],
        "norm": [],
        "downsample": [],
        "cpd": [],
        "distance": [],
        "total": [],
    }
    if cpd_type is not None:
        # set source to use for all in row here
        cpd_obj = cpd.RigidCPD(
            source=x[0]["pos"],
            use_color=False,
            # rot=None,
            # t=None,
            # scale=None,
            tf_init_params={"device": x[0]["pos"].device, "dtype": x[0]["pos"].dtype},
            log_freq=-1,
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
        # TODO: make comms communicate only the row that was calculated instead of the whole matrix (bandaid for now)
        for j in range(window_min, window_max):
            if full_counter % size != rank and mpi_distribute:
                dtw_matrix[:, i, j] = 0.0
                full_counter += 1
                continue
            t0 = time.perf_counter()
            # copies to avoid overwriting...
            xi = deepcopy(x[i])
            yj = deepcopy(y[j])
            tc = time.perf_counter()
            # normalize the smaller point cloud to the largest
            if normalize:
                xi["pos"], yj["pos"], _ = utils.normalize_to_larger_pc(xi["pos"], yj["pos"])
            tn = time.perf_counter()
            # downsample the point could to be the same size
            xi, yj = downsample_fn(xi, yj)
            tdn = time.perf_counter()

            # do CPD registration to transform *yj*
            # this means that yj is the source and xi is the target
            cpd_metric = torch.inf
            if cpd_type is not None:
                # need to set the source as the normalized xi
                cpd_obj.set_source(xi["pos"])
                cpd_obj.reset_transform()
                reg = cpd_obj.registration(yj["pos"], w=0.0, maxiter=1000, tol=1e-5)
                xi["pos"] = cpd_obj.transformation.transform(xi["pos"])
                cpd_metric = reg.q
            tcpd = time.perf_counter()

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

            for di in range(len(distance_fns)):
                dtw_matrix[di, i, j] = dists[di]

            # _save_value_to_array(filename=save_filename, value=dist.item(), matrix=dtw_matrix, indexi=i, indexj=j, lock_filename=lockfile)

            full_counter += 1  # noqa: E741
            tf = time.perf_counter()
            times["copy"].append(tc - t0)
            times["norm"].append(tn - tc)
            times["downsample"].append(tdn - tn)
            times["cpd"].append(tcpd - tdn)
            times["distance"].append(tdist - tcpd)
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

        # if l in log_intervals:
        tc = sum(times["copy"]) / float(len(times["copy"]))
        tn = sum(times["norm"]) / float(len(times["norm"]))
        tdn = sum(times["downsample"]) / float(len(times["downsample"]))
        tcpd = sum(times["cpd"]) / float(len(times["cpd"]))
        tdi = sum(times["distance"]) / float(len(times["distance"]))
        tt = sum(times["total"]) / float(len(times["total"]))
        # log.info(
        #     f"iteration {l + 1}/{num_dist_elems + 1}: time: full: {tt:.4f}, copy: {tc:.4f}, "
        #     f"norm: {tn:.4f}, downsample: {tdn:.4f}, cpd: {tcpd:.4f}, distance: {tdi:.4f}"
        # )
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
        if mpi_distribute:
            tcomm = time.perf_counter()
            row = dtw_matrix[:, i].cpu().numpy()
            row = comm_world.allreduce(row)
            dtw_matrix[:, i] = torch.tensor(row, device=dtw_matrix.device, dtype=dtw_matrix.dtype)
            if rank == 0:
                print(f"Allreduce time required: {time.perf_counter() - tcomm}")
            # print(dtw_matrix)

    return dtw_matrix


def _sanitize_dtw_matrix(distance_kwargs, distance_metrics, downsample_method, x, y):
    if not isinstance(distance_metrics, list):
        distance_metrics = [
            distance_metrics,
        ]
    if not isinstance(distance_kwargs, list):
        distance_kwargs = [
            distance_kwargs,
        ]
    if len(distance_kwargs) != len(distance_metrics) and distance_kwargs[0] is not None:
        raise RuntimeError(f"len distance kwargs != len distance metrics!: {distance_kwargs} v {distance_metrics}")

    if distance_kwargs[0] is None:
        distance_kwargs = [
            None,
        ] * len(distance_metrics)

    for c, dist in enumerate(distance_metrics):
        dist_kwargs = distance_kwargs[c]
        if dist_kwargs is None:
            dist_kwargs = {}

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
        if dist not in ["euclidean", "manhatten", "minkowski"] and downsample_method is None:
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
