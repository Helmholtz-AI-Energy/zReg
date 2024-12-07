import scipy.io as sio
import logging
import time
from typing import Dict, Tuple

import open3d.t.geometry as o3dtgeo
import open3d.core as o3c

# from torch_geometric.data import Data

import torch


log = logging.getLogger(__name__)


__all__ = ["load_data_from_tracklets", "open3d_to_torch", "torch_to_open3d"]


def load_data_from_tracklets(
    filepath: str,
    device: str = "cpu",
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict]]:
    """Load data from tracklets.

    Loads data from a MATLAB file containing tracklet data and returns a tuple containing
    a dictionary of point clouds and the raw tracklet data.

    Parameters
    ----------
    filepath : str
        Path to the MATLAB file containing tracklet data.
    device : str, optional
        Device to place PyTorch tensors on, by default 'cpu'.

    Returns
    -------
    tuple
        A tuple containing:
            - A dictionary where keys are time points and values are point clouds at each
              time point. Point clouds are PyTorch tensors.
              Dictionary structure:
                  pc[i]['pos'] -> x, y, z position
                  pc[i]['color'] -> color
                  pc[i]['id'] -> cell id
            - The raw tracklet data as loaded from the MATLAB file.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the file does not contain valid tracklet data

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
    if device is not None and device != "cpu" and not torch.cuda.is_available():
        log.info("CUDA/GPUs not available, defaulting to cpu loading")
        device = "cpu"

    log.info(f"Loading data from {filepath}")
    t0 = time.perf_counter()

    data = sio.loadmat(filepath, simplify_cells=True, squeeze_me=True)
    pc = {i: {"pos": [], "color": [], "id": []} for i in range(len(data["trackletsPerTimePoint"]))}
    # timestep, positions (x, y, z, tracklet_num)

    for idx in range(len(data["tracklets"])):
        tracklet = data["tracklets"][idx]
        col = tracklet["color"].tolist()
        cellid = tracklet["id"]

        # add tracklet to all point cloud entries
        for c, j in enumerate(range(tracklet["startTime"] - 1, tracklet["endTime"])):
            # off by one makes this from -1 -> normal (its included in the endTime number but that is 1 too high)
            # append a list to create a list of lists (extend() extends the list)
            # print()
            pos = tracklet["pos"][c].tolist()
            pc[j]["pos"].append(pos)
            pc[j]["color"].append(col)
            pc[j]["id"].append(cellid)

    for i in pc:
        pc[i]["pos"] = torch.tensor(pc[i]["pos"], device=device)
        pc[i]["color"] = torch.tensor(pc[i]["color"], device=device)
        pc[i]["id"] = torch.tensor(pc[i]["id"], device=device)

    t1 = time.perf_counter() - t0
    log.info(f"Finished loading. Time required: {t1}")
    return pc, data["tracklets"]


def torch_to_open3d(pc: Dict[str, torch.Tensor]) -> o3dtgeo.PointCloud:
    """Converts a point cloud from a PyTorch dictionary to an Open3D point cloud.

    This function takes a dictionary representing a point cloud, where the keys are
    'pos', 'color', and 'id', and the values are either PyTorch tensors or Open3D
    tensors. It converts the dictionary to an Open3D point cloud object.

    Args:
        pc: A dictionary representing the point cloud. The keys should be 'pos',
            'color', and 'id', and the values should be either PyTorch tensors or
            Open3D tensors.

    Returns:
        An Open3D point cloud object.

    Raises:
        KeyError: If the input dictionary does not contain the keys 'pos', 'color',
                  and 'id'.
    """

    # Check if the input is a PyTorch tensor
    from_torch = isinstance(pc["pos"], torch.Tensor)

    # Create a dictionary to store the Open3D tensors
    map_to_tensors = {}

    # Convert the tensors to Open3D tensors
    if from_torch:
        map_to_tensors["positions"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["pos"]))
        map_to_tensors["colors"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["color"]))
        map_to_tensors["labels"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["id"]))
    else:
        # If the input is already an Open3D tensor, no conversion is needed
        map_to_tensors["positions"] = pc["pos"]
        map_to_tensors["colors"] = pc["color"]
        map_to_tensors["labels"] = pc["id"]

    # Create and return the Open3D point cloud
    return o3dtgeo.PointCloud(map_to_tensors)


def open3d_to_torch(
    pc: o3dtgeo.PointCloud, device: torch.device = None, to_torch: bool = True
) -> Dict[str, torch.Tensor]:
    """Converts an Open3D point cloud to a dictionary of Torch tensors or NumPy arrays.

    This function extracts the positions, colors, and labels from an Open3D
    point cloud and returns them as a dictionary. The values in the dictionary
    can be either Torch tensors (if `to_torch` is True) or NumPy arrays
    (if `to_torch` is False).

    Parameters
    ----------
    pc : open3d.t.geometry.PointCloud
        The Open3D point cloud to convert.
    device : torch.device, optional
        The device to which the Torch tensors should be moved.
        If None, the tensors will be created on the CPU.
    to_torch : bool, optional
        Whether to convert the data to Torch tensors.
        If False, the data will be returned as NumPy arrays.

    Returns
    -------
    dict
        A dictionary containing the point cloud data. The keys are
        "pos", "color", and "id", and the values are either Torch tensors
        or NumPy arrays.

    Examples
    --------
    >>> import open3d as o3d
    >>> import torch
    >>> pc = o3d.t.geometry.PointCloud()
    >>> # ... populate the point cloud ...
    >>> data = open3d_to_torch(pc, device=torch.device('cuda:0'))
    >>> print(data['pos'].device)
    cuda:0
    >>> data = open3d_to_torch(pc, to_torch=False)
    >>> print(type(data['pos']))
    <class 'numpy.ndarray'>
    """

    # Extract positions, colors, and labels as NumPy arrays
    if isinstance(pc, o3dtgeo.PointCloud):
        pos = pc.point.positions.cpu().numpy()
        col = pc.point.colors.cpu().numpy()
        try:
            ids = pc.point.labels.cpu().numpy()
        except KeyError:
            ids = None
    else:
        pos = pc.positions.cpu().numpy()
        col = pc.colors.cpu().numpy()
        try:
            ids = pc.labels.cpu().numpy()
        except KeyError:
            ids = None

    ret = {}
    if to_torch:
        # Convert to PyTorch tensors and move to the specified device
        ret["pos"] = torch.tensor(pos, device=device)
        ret["color"] = torch.tensor(col, device=device)
        ret["id"] = torch.tensor(ids, device=device) if ids is not None else None
    else:
        # Return NumPy arrays
        ret["pos"] = pos
        ret["color"] = col
        ret["id"] = ids
    return ret


def to(pc, device):
    if not isinstance(pc, dict):
        raise NotImplementedError("FIXME")
    else:
        pc["pos"] = pc["pos"].to(device=device)
        pc["color"] = pc["color"].to(device=device)
        pc["id"] = pc["id"].to(device=device) if pc["id"] is not None else None
        return pc


# def to_torch_geometric(pc: Dict[str, torch.Tensor]):
#     data = Data(pos=pc['pos'], id=pc['id'], color=pc['color'])
#     return data
