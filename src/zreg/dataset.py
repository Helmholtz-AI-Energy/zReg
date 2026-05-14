import scipy.io as sio
import logging
import time
import csv
from typing import Self

from pathlib import Path

import torch


def _import_open3d():
    try:
        import open3d.t.geometry as o3dtgeo
        import open3d.core as o3c
        return o3dtgeo, o3c, True
    except (ImportError, OSError):
        return None, None, False


log = logging.getLogger(__name__)


__all__ = ["load_data_from_tracklets", "open3d_to_zreg", "zreg_to_open3d", "zRegPointCloud", "load_shah_from_csv"]


class zRegPointCloud(dict):  # dict[str, torch.Tensor]
    def __init__(self, *args, **kwargs):
        super(zRegPointCloud, self).__init__(*args, **kwargs)

        # add default values to point cloud
        for key in ["pos", "color", "id", "fps-idx"]:
            self[key] = kwargs[key] if key in kwargs else None

    def to(self, device) -> Self:
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                self[k] = self[k].to(device=device)
        return self

    def get_open3d_pc(self):
        o3dtgeo, o3c, has_open3d = _import_open3d()
        if not has_open3d:
            raise RuntimeError("open3d is not available. Check your Python architecture and open3d installation.")
        map_to_tensors = {}
        map_to_tensors["positions"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(self["pos"]))
        return o3dtgeo.PointCloud(map_to_tensors)


def load_data_from_tracklets(
    filepath: str,
    device: str = "cpu",
) -> tuple[dict[int, zRegPointCloud], dict[int, dict]]:
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
        pc[i] = zRegPointCloud(
            pos=torch.tensor(pc[i]["pos"], device=device),
            color=torch.tensor(pc[i]["color"], device=device),
            id=torch.tensor(pc[i]["id"], device=device),
        )
        # pc[i]["pos"] =
        # pc[i]["color"] = torch.tensor(pc[i]["color"], device=device)
        # pc[i]["id"] = torch.tensor(pc[i]["id"], device=device)

    t1 = time.perf_counter() - t0
    log.info(f"Finished loading. Time required: {t1}")
    return pc, data["tracklets"]


def zreg_to_open3d(pc: zRegPointCloud) -> "o3dtgeo.PointCloud":
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
        RuntimeError: If open3d is not available.
    """
    o3dtgeo, o3c, has_open3d = _import_open3d()
    if not has_open3d:
        raise RuntimeError("open3d is not available. Check your Python architecture and open3d installation.")

    from_torch = isinstance(pc["pos"], torch.Tensor)
    map_to_tensors = {}
    if from_torch:
        map_to_tensors["positions"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["pos"]))
        map_to_tensors["colors"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["color"]))
        map_to_tensors["labels"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["id"]))
        if pc["fps-idx"] is not None:
            map_to_tensors["fps_idx"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["fps-idx"]))
    else:
        map_to_tensors["positions"] = pc["pos"]
        map_to_tensors["colors"] = pc["color"]
        map_to_tensors["labels"] = pc["id"]
        if pc["fps-idx"] is not None:
            map_to_tensors["fps_idx"] = pc["fps-idx"]
    return o3dtgeo.PointCloud(map_to_tensors)


def open3d_to_zreg(
    pc: "o3dtgeo.PointCloud", device: torch.device = None, to_torch: bool = True
) -> dict[str, torch.Tensor]:
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

    Raises:
        RuntimeError: If open3d is not available.
    """
    o3dtgeo, o3c, has_open3d = _import_open3d()
    if not has_open3d:
        raise RuntimeError("open3d is not available. Check your Python architecture and open3d installation.")

    # Extract positions, colors, and labels as NumPy arrays
    if isinstance(pc, o3dtgeo.PointCloud):
        pos = pc.point.positions.cpu().numpy()
        col = pc.point.colors.cpu().numpy()
        try:
            ids = pc.point.labels.cpu().numpy()
        except KeyError:
            ids = None
        try:
            fps_idx = pc.point.fps_idx.cpu().numpy()
        except KeyError:
            fps_idx = None
    else:
        pos = pc.positions.cpu().numpy()
        col = pc.colors.cpu().numpy()
        try:
            ids = pc.labels.cpu().numpy()
        except KeyError:
            ids = None
        try:
            fps_idx = pc.fps_idx.cpu().numpy()
        except KeyError:
            fps_idx = None

    ret = zRegPointCloud()
    if to_torch:
        # Convert to PyTorch tensors and move to the specified device
        ret["pos"] = torch.tensor(pos, device=device)
        ret["color"] = torch.tensor(col, device=device)
        ret["id"] = torch.tensor(ids, device=device) if ids is not None else None
        ret["fps-idx"] = torch.tensor(fps_idx, device=device) if fps_idx is not None else None
    else:
        # Return NumPy arrays
        ret["pos"] = pos
        ret["color"] = col
        ret["id"] = ids
        ret["fps-idx"] = fps_idx
    return ret


def load_shah_from_csv(filepath: str | Path, device: str | torch.device) -> dict[int, zRegPointCloud]:
    """
    Loads point cloud data from a CSV file in the format used by Shah

    This function assumes the CSV file has the following columns:

    - `x`, `y`, `z`: Coordinates of the point.
    - `t`: Time index.
    - `layer`: Layer index (used as color).
    - `id`: Point ID.

    The function reads the CSV file, extracts the data, and creates a dictionary of
    `zRegPointCloud` objects, where the keys are the time indices and the values
    are the corresponding point clouds.

    Args:
        filename (str or Path): The path to the CSV file.
        device (torch.device): The device to store the point cloud data on.

    Returns:
         dict[int, zRegPointCloud]: A dictionary of `zRegPointCloud` objects.
    """
    pcs = {}

    with open(filepath, "r") as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            # skip header
            if i == 0:
                i += 1
                continue

            # Extract data
            x, y, z, t, layer, ident = [float(x) for x in row]
            t, layer, ident = int(t) - 1, int(layer), int(ident)

            # add data to pcs dictionary
            if t in pcs:
                pcs[t]["pos"].append([x, y, z])
                pcs[t]["id"].append(ident)
                pcs[t]["labels"].append(layer)
            else:
                pcs[t] = {
                    "pos": [[x, y, z]],
                    "id": [ident],
                    "labels": [layer],
                }

    # Convert to zRegPointCloud objects
    for i in range(min(pcs), max(pcs) + 1):
        pcs[i] = zRegPointCloud(
            pos=torch.tensor(pcs[i]["pos"], device=device),
            color=torch.tensor(pcs[i]["labels"], device=device),
            id=torch.tensor(pcs[i]["id"], device=device),
        )
    return pcs
