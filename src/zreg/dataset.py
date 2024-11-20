import pandas as pd
import scipy.io as sio
import numpy as np

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


__all__ = [
    "load_data_from_tracklets",
]


def load_data_from_tracklets(
    filepath: str,
    return_pandas: bool = False,
    return_torch: bool = True,
    device: str = "cpu",
) -> tuple:
    """Load data from tracklets.

    Loads data from a MATLAB file containing tracklet data and returns a tuple containing
    a dictionary of point clouds and the raw tracklet data.

    Parameters
    ----------
    filepath : str
        Path to the MATLAB file containing tracklet data.
    return_pandas : bool, optional
        Whether to return point clouds as pandas DataFrames, by default False.
    return_torch : bool, optional
        Whether to return point clouds as PyTorch tensors, by default True.
    device : str, optional
        Device to place PyTorch tensors on, by default 'cpu'.

    Returns
    -------
    tuple
        A tuple containing:
            - A dictionary where keys are time points and values are point clouds at each
              time point. Point clouds can be NumPy arrays, pandas DataFrames, or PyTorch
              tensors depending on the `return_pandas` and `return_torch` parameters.
            - The raw tracklet data as loaded from the MATLAB file.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the file does not contain valid tracklet data or if both `return_pandas`
        and `return_torch` are True.

    Notes
    -----
    The MATLAB file should contain a variable named 'tracklets' which is a cell array
    of tracklet structures. Each tracklet structure should have the following fields:

        * startTime: The starting time point of the tracklet.
        * endTime: The ending time point of the tracklet.
        * pos: An array of positions (x, y, z) for each time point in the tracklet.
        * id: The ID of the tracklet.

    Examples
    --------
    >>> pc, tracklets = load_data_from_tracklets('tracklets.mat')
    >>> pc[0]  # Get the point cloud at the first time point
    array([[1.0, 2.0, 3.0, 1],
           [4.0, 5.0, 6.0, 2]])
    >>> tracklets[0]['id']  # Access the ID of the first tracklet
    1
    >>> pc, tracklets = load_data_from_tracklets('tracklets.mat', return_pandas=True)
    >>> pc[0]  # Get the point cloud at the first time point as a pandas DataFrame
            0    1    2
    1  1.0  2.0  3.0
    2  4.0  5.0  6.0
    >>> pc, tracklets = load_data_from_tracklets('tracklets.mat', return_torch=True, device='cuda')
    >>> pc[0]  # Get the point cloud at the first time point as a PyTorch tensor on the GPU
    tensor([[1.0, 2.0, 3.0, 1],
            [4.0, 5.0, 6.0, 2]], device='cuda:0')
    """
    if return_torch and not has_torch:
        raise RuntimeError("torch is not available. install it to use torch/gpus")
    if return_pandas and return_torch:
        raise ValueError("Cannot return both pandas DataFrames and PyTorch tensors.")

    data = sio.loadmat(filepath, simplify_cells=True, squeeze_me=True)
    pc = {i: [] for i in range(len(data["trackletsPerTimePoint"]))}
    # timestep, positions (x, y, z, tracklet_num)

    for idx in range(len(data["tracklets"])):
        tracklet = data["tracklets"][idx]

        # add tracklet to all point cloud entries
        for c, j in enumerate(range(tracklet["startTime"] - 1, tracklet["endTime"])):
            # off by one makes this from -1 -> normal (its included in the endTime number but that is 1 too high)
            # append a list to create a list of lists (extend() extends the list)
            pos_id = tracklet["pos"][c].tolist()
            pos_id.append(tracklet["id"])
            pc[j].append(pos_id)

    for i in pc:
        if return_pandas:
            pc[i] = pd.DataFrame(np.array(pc[i])[:, :3], index=np.array(pc[i])[:, 3])
        elif return_torch:
            pc[i] = torch.tensor(pc[i], device=device)
        else:
            pc[i] = np.array(pc[i])
    return pc, data["tracklets"]
