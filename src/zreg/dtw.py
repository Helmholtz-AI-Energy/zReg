from copy import deepcopy
import torch
from . import utils
from . import downsampling
from . import distances

__all__ = ["create_dtw_matrix"]


def create_dtw_matrix(
    x: dict, y: dict, window=None, distance_metric="swd", normalize=True, distance_kwargs: dict = None
):
    if distance_metric == "swd":
        if distance_kwargs is None:
            distance_kwargs = {}
        # TODO: automate kwarg defaults?
        distance_fn = distances.SlicedWassersteinDistance(num_projs=200, device=x[0]["pos"].device, nobatchdim=True)
    elif distance_metric == "euclidean":
        # TODO: update with partial for kwargs
        distance_fn = distances.euclidean_distance

    x_samples = max(x)  # gets the max key (all ints...)
    y_samples = max(y)  # gets the max key (all ints...)

    dtw_matrix = torch.full(
        (x_samples + 1, y_samples + 1), torch.inf, dtype=x[0]["pos"].dtype, device=x[0]["pos"].device
    )
    # xpos, ypos = x['pos'], y['pos']

    # TODO: add raise for window not being int and if its not positive and less than 0

    for i in range(x_samples + 1):
        if window is not None:
            window_min = i - window
            if window_min < 0:
                window_min = 0
            window_max = i + window
            if window_max > y_samples + 1:
                window_max = y_samples + 1
        else:
            window_min, window_max = 0, y_samples + 1

        # compare the sample with other samples in window
        for j in range(window_min, window_max):
            # t0 = time.perf_counter()
            xi = deepcopy(x[i])
            yj = deepcopy(y[j])
            # tc = time.perf_counter()
            if normalize:
                xi["pos"], yj["pos"], _ = utils.normalize_to_larger_pc(xi["pos"], yj["pos"])
            # tn = time.perf_counter()
            # TODO: add option for which downsampling to use
            xi, yj = downsampling.farthest_point_down_sample(xi, yj, preserve_labels=False)
            # xi, yj = downsampling.random_down_sample(xi, yj)
            # xi, yj = downsampling.uniform_down_sample(xi, yj)

            dist = distance_fn(xi["pos"], yj["pos"])
            if dist.numel() > 1:
                dist = dist.mean()
            dtw_matrix[i, j] = dist

    return dtw_matrix
