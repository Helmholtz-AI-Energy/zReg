import pandas as pd
import scipy.io as sio
import numpy as np

__all__ = [
    "load_data_from_tracklets",
]


def load_data_from_tracklets(filepath: str, return_pandas: bool = False) -> tuple:
    """Load data from tracklets.

    Loads data from a MATLAB file containing tracklet data and returns a tuple containing
    a dictionary of point clouds and the raw tracklet data.

    Parameters
    ----------
    filepath : str
        Path to the MATLAB file containing tracklet data.
    return_pandas : bool, optional
        Whether to return point clouds as pandas DataFrames, by default False.

    Returns
    -------
    tuple
        A tuple containing:
            - A dictionary where keys are time points and values are point clouds at each
              time point. Point clouds are either NumPy arrays or pandas DataFrames
              depending on the `return_pandas` parameter.
            - The raw tracklet data as loaded from the MATLAB file.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the file does not contain valid tracklet data.

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
    """

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
        else:
            pc[i] = np.array(pc[i])
    return pc, data["tracklets"]
