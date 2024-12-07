from pathlib import Path
from functools import partial
from copy import deepcopy
import logging
import os
import time

from mpi4py import MPI

import torch
import numpy as np

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
    distance_metric: str = "swd",
    distance_kwargs: dict = None,
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
    distance_kwargs, distance_metric, downsample_method, distance_fn, downsample_fn = _sanitize_dtw_matrix(
        distance_kwargs=distance_kwargs,
        distance_metric=distance_metric,
        downsample_method=downsample_method,
        x=x,
        y=y,
    )

    # Precompute the farthest point downsampling if necessary
    if downsample_method.startswith("farthest"):
        log.debug("Starting precompute for farthest points for downsampling for all time series points")
        t0 = time.perf_counter()
        for k in x:
            if "fps-idx" not in x[k]:
                x[k] = downsampling.precompute_fps(x[k])
            if "fps-idx" not in y[k]:
                y[k] = downsampling.precompute_fps(y[k])
        log.debug(f"Precomute time required: {time.perf_counter() - t0}")

    # Get the number of samples in each set of data
    x_samples = max(x)
    y_samples = max(y)

    if save_filename is None:
        dtw_matrix = torch.full(
            (x_samples + 1, y_samples + 1), torch.inf, dtype=x[0]["pos"].dtype, device=x[0]["pos"].device
        )
    else:
        np_dtype = x[0]["pos"][:2, :2].cpu().numpy().dtype
        dtw_matrix = np.full((x_samples + 1, y_samples + 1), np.inf, dtype=np_dtype)
        lockfile = "lockfile-to-be-removed-later"
        # want to stager the processes to avoid errors
        time.sleep(rank * 0.01)

        try:
            if os.path.exists(lockfile):
                os.remove(lockfile)
        except Exception as e:
            print(f"Exception when deleting file!: {e}")

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
                dtw_matrix[i, j] = 0.0
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

            if cpd_type is not None:
                # need to set the source as the normalized xi
                cpd_obj.set_source(xi["pos"])
                cpd_obj.reset_transform()
                cpd_obj.registration(yj["pos"], w=0.0, maxiter=200, tol=1e-5)
                xi["pos"] = cpd_obj.transformation.transform(xi["pos"])
            tcpd = time.perf_counter()

            if distance_metric == "aswd":
                file = Path(distance_fn.projs_history)
                if file.exists():
                    # remove the proj history...need to do this after every distance
                    try:
                        os.remove("projs_history.txt")
                    except FileNotFoundError:
                        # preventing race condition when running in parallel
                        pass
            dist = distance_fn(xi["pos"], yj["pos"])
            tdist = time.perf_counter()

            if dist.numel() > 1:
                dist = dist.mean()

            if isinstance(dtw_matrix, torch.Tensor):
                # create matrix locally
                dtw_matrix[i, j] = dist
            # else:
            # save the file to the
            # print(f"saving {i, j}")
            # _save_value_to_array(filename=save_filename, value=dist.item(), matrix=dtw_matrix, indexi=i, indexj=j, lock_filename=lockfile)
            elif isinstance(dtw_matrix, np.ndarray):
                # print(i, j, dist.item())
                dtw_matrix[i, j] = dist.item()
                # print(i, j, dist.item(), dtw_matrix[i, j])
            else:
                raise TypeError(f"dtw matrix is type {type(dtw_matrix)}")
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
            row = dtw_matrix[i].copy()
            row = comm_world.allreduce(row)
            dtw_matrix[i] = row
            if rank == 0:
                print(f"Allreduce time required: {time.perf_counter() - tcomm}")
            # print(dtw_matrix)

    return dtw_matrix


def _sanitize_dtw_matrix(distance_kwargs, distance_metric, downsample_method, x, y):
    if distance_kwargs is None:
        distance_kwargs = {}

    if distance_metric == "swd":
        log.info("Using Sliced Wasserstein Distance for distance metric")
        # set default kwargs
        defaults = [
            ["device", x[0]["pos"].device],
            ["num_projs", 50],
        ]
        for kw, val in defaults:
            if kw not in distance_kwargs:
                distance_kwargs[kw] = val
        distance_fn = distances.SlicedWassersteinDistance(**distance_kwargs)
    elif distance_metric == "aswd":
        log.info("Using Adaptive Sliced Wasserstein Distance for distance metric")
        defaults = [
            ["device", x[0]["pos"].device],
            ["max_slices", 100],
            ["init_projs", 50],
            ["step_projs", 25],
        ]
        for kw, val in defaults:
            if kw not in distance_kwargs:
                distance_kwargs[kw] = val
        if Path("projs_history.txt").exists():
            # remove the proj history...need to do this after every distance
            try:
                os.remove("projs_history.txt")
            except FileNotFoundError:
                pass
        distance_fn = distances.AdaptiveSlicedWassersteinDistance(**distance_kwargs)
    elif distance_metric == "oswd":
        log.info("Using Orthogonal Wasserstein Distance for distance metric")
        # set default kwargs
        defaults = [
            ["device", x[0]["pos"].device],
            ["num_projs", 50],
        ]
        for kw, val in defaults:
            if kw not in distance_kwargs:
                distance_kwargs[kw] = val
        distance_fn = distances.OrthogonalSlicedWassersteinDistance(**distance_kwargs)
    elif distance_metric == "gswd":
        log.info("Using Generalised Sliced Wasserstein Distance for distance metric")
        # set default kwargs
        defaults = [
            ["device", x[0]["pos"].device],
            ["num_projs", 50],
        ]
        for kw, val in defaults:
            if kw not in distance_kwargs:
                distance_kwargs[kw] = val
        distance_fn = distances.GeneralisedSlicedWassersteinDistance(**distance_kwargs)
    elif distance_metric == "pswd":
        log.info("Using Projected Wasserstein Distance for distance metric")
        # set default kwargs
        defaults = [
            ["device", x[0]["pos"].device],
            ["num_projs", 50],
        ]
        for kw, val in defaults:
            if kw not in distance_kwargs:
                distance_kwargs[kw] = val
        distance_fn = distances.ProjectedWassersteinDistance(**distance_kwargs)
    elif distance_metric == "euclidean":
        log.info("Using Euclidean Distance for distance metric")
        distance_fn = partial(distances.euclidean_distance, **distance_kwargs)
    elif distance_metric == "manhattan":
        log.info("Using Manhatten Distance for distance metric")
        distance_fn = partial(distances.manhattan_distance, **distance_kwargs)
    elif distance_metric == "minkowski":
        defaults = [
            ["p", 3],
        ]
        for kw, val in defaults:
            if kw not in distance_kwargs:
                distance_kwargs[kw] = val
        log.info(f"Using Minkowski Distance for distance metric with p={distance_kwargs['p']}")
        distance_fn = partial(distances.minkowski_distance, **distance_kwargs)
    else:
        raise ValueError(f"Invalid distance function: {distance_metric}")

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

    if distance_metric not in ["euclidean", "manhatten", "minkowski"] and downsample_method is None:
        raise RuntimeError("with SWD methods, need to use a downsampling method")

    return distance_kwargs, distance_metric, downsample_method, distance_fn, downsample_fn


def _save_value_to_array(filename, value, lock_filename, matrix, indexi, indexj):
    """
    Iteratively saves a value to a numpy array and stores it in a file.

    Uses a lock file to prevent data corruption from multiple processes.

    Args:
        filename: The name of the file to save the array to.
        value: The value to append to the array.
        lock_filename: The name of the lock file.
    """
    while os.path.exists(lock_filename):
        time.sleep(0.01)
    with open(lock_filename, "w") as lock_file:
        lock_file.write("locked")

        # Load existing array or create a new one
        if os.path.exists(filename):
            matrix = np.load(filename)

        # Append the new value
        matrix[indexi, indexj] = value
        # print(matrix.shape)

        # Save the updated array
        np.save(filename, matrix)

    # Release the lock by deleting the lock file
    os.remove(lock_filename)

    # while True:
    #     try:
    #         # if
    #         time.sleep(MPI.COMM_WORLD.rank * 0.1)
    #         # Attempt to acquire the lock
    #         with open(lock_filename, 'w') as lock_file:
    #             lock_file.write('locked')

    #             # Load existing array or create a new one
    #             if os.path.exists(filename):
    #                 matrix = np.load(filename)

    #             # Append the new value
    #             matrix[indexi, indexj] = value
    #             # print(matrix.shape)

    #             # Save the updated array
    #             np.save(filename, matrix)

    #         # Release the lock by deleting the lock file
    #         os.remove(lock_filename)
    #         break  # Exit the loop after successful save
    #     except Exception as e:
    #         # If lock acquisition fails, wait and retry
    #         if "No such file or directory" in e:
    #             time.sleep(0.05)
    #         else:
    #             raise e
    #         # print(f"Error saving value: {e}")
    #         continue
