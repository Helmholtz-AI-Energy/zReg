import scipy.io as sio
import logging
import time
import csv
from typing import Self

from pathlib import Path
import importlib.util

import torch

HAS_OPEN3D = importlib.util.find_spec("open3d") is not None


def _import_open3d():
    if not HAS_OPEN3D:
        return None, None, False
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
        for key in ["pos", "label", "id", "fps-idx"]:
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


def _canonical_colour(raw, context: str = "tracklet colour"):
    """Normalise one tracklet colour to a Python scalar or a tuple.

    ``scipy.io.loadmat(..., simplify_cells=True, squeeze_me=True)`` returns a
    scalar colour as a Python ``int``/``float`` and an RGB colour as a numpy
    array, so the raw value cannot be assumed to have ``tolist``.

    Parameters
    ----------
    raw : int | float | bool | numpy.ndarray | numpy.generic | torch.Tensor
        Raw colour value of one tracklet.
    context : str, optional
        Location used in error messages, e.g. ``"tracklet 7 (id 12) colour"``.

    Returns
    -------
    int | float | bool | tuple
        Python ``int``/``float``/``bool`` scalars pass through unchanged.
        Anything with a ``tolist`` method (numpy array, numpy scalar, torch
        tensor) is converted with ``tolist()``; a resulting list of length 1
        is unwrapped to its element (a scalar colour) and a list of length 3
        or 4 of scalars becomes a ``tuple`` (an RGB / RGBA colour).

    Raises
    ------
    ValueError
        For an empty colour (MATLAB ``[]``) or a sequence colour of any other
        length or with non-scalar entries (62-REVIEW WR-10): such a colour is
        malformed and must not be misreported later as "mixed scalar and RGB
        colours" or silently become an extra label class.
    """
    if isinstance(raw, (bool, int, float)):
        return raw
    if hasattr(raw, "tolist"):
        value = raw.tolist()
    else:
        value = raw
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{context} is empty; expected a scalar or an RGB(A) triple")
        if len(value) == 1:
            return value[0]
        if len(value) not in (3, 4) or not all(
            isinstance(v, (bool, int, float)) for v in value
        ):
            raise ValueError(
                f"{context} {value!r} is malformed; expected a scalar or an RGB(A) "
                "colour of 3 or 4 scalar entries"
            )
        return tuple(value)
    return value


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
                  pc[i]['label'] -> label
                  pc[i]['id'] -> cell id
            - The raw tracklet data as loaded from the MATLAB file.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the file does not contain valid tracklet data, or if it mixes
        scalar and RGB colours.

    Notes
    -----
    The MATLAB file should contain a variable named 'tracklets' which is a cell array
    of tracklet structures. Each tracklet structure should have the following fields:

        * startTime: The starting time point of the tracklet.
        * endTime: The ending time point of the tracklet.
        * pos: An array of positions (x, y, z) for each time point in the tracklet.
        * id: The ID of the tracklet.
        * color: A scalar class id or an RGB triple.

    Colours are normalised by ``_canonical_colour`` (Python scalar, or a
    tuple for RGB). Whether RGB colours are remapped to integer indices is
    decided over all labels of all non-empty frames, so leading empty frames
    do not disable the remap; every unique RGB triple maps to the same index
    in every frame (first-seen order over frames and points). A recording
    that mixes scalar and RGB colours raises ``ValueError``.

    Label dtype is one value for the whole recording: ``torch.long`` when the
    RGB remap ran, when every colour is an integer, or when every frame is
    empty; ``torch.float32`` when the colours are float scalars. Empty frames
    get a ``(0,)`` label tensor of that dtype.

    Examples
    --------
    Write a small two-tracklet file and load it:

    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> import scipy.io as sio
    >>> from zreg.core.dataset import load_data_from_tracklets
    >>> tracklets = np.empty(2, dtype=object)
    >>> tracklets[0] = {"startTime": 1, "endTime": 2, "id": 1, "color": 3,
    ...                 "pos": np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])}
    >>> tracklets[1] = {"startTime": 1, "endTime": 2, "id": 2, "color": 5,
    ...                 "pos": np.array([[4.0, 5.0, 6.0], [4.5, 5.5, 6.5]])}
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "tracklets.mat"
    ...     sio.savemat(path, {"tracklets": tracklets, "trackletsPerTimePoint": np.zeros(2)})
    ...     pc, raw = load_data_from_tracklets(str(path))
    >>> pc[0]["pos"]  # positions at the first time point
    tensor([[1., 2., 3.],
            [4., 5., 6.]])
    >>> pc[0]["label"], pc[0]["id"]
    (tensor([3, 5]), tensor([1, 2]))
    >>> pc[1]["pos"]
    tensor([[1.5000, 2.5000, 3.5000],
            [4.5000, 5.5000, 6.5000]])
    >>> int(raw[0]["id"])  # raw MATLAB tracklet data
    1
    """
    if device is not None and device != "cpu" and not torch.cuda.is_available():
        log.info("CUDA/GPUs not available, defaulting to cpu loading")
        device = "cpu"

    log.info(f"Loading data from {filepath}")
    t0 = time.perf_counter()

    data = sio.loadmat(filepath, simplify_cells=True, squeeze_me=True)
    pc = {i: {"pos": [], "label": [], "id": []} for i in range(len(data["trackletsPerTimePoint"]))}
    # timestep, positions (x, y, z, tracklet_num)

    for idx in range(len(data["tracklets"])):
        tracklet = data["tracklets"][idx]
        cellid = tracklet["id"]
        col = _canonical_colour(
            tracklet["color"], context=f"{filepath}: tracklet {idx} (id {cellid!r}) colour"
        )

        # add tracklet to all point cloud entries
        for c, j in enumerate(range(tracklet["startTime"] - 1, tracklet["endTime"])):
            # off by one makes this from -1 -> normal (its included in the endTime number but that is 1 too high)
            # append a list to create a list of lists (extend() extends the list)
            # print()
            pos = tracklet["pos"][c].tolist()
            pc[j]["pos"].append(pos)
            pc[j]["label"].append(col)
            pc[j]["id"].append(cellid)

    # MATLAB tracklet "color" is often an RGB triple [r, g, b].  When that is
    # the case, pc[j]["label"] ends up as a list of tuples and the resulting
    # tensor would be 2-D (n_points, 3), which breaks KNN-voting label
    # transfer that expects 1-D integer class indices.  Decide over ALL labels
    # of all non-empty frames (leading frames may be empty) and build a
    # globally consistent colour->index mapping so every unique RGB triple
    # maps to the same integer across all frames.
    first_rgb = None
    first_scalar = None
    for frame_data in pc.values():
        for c in frame_data["label"]:
            if isinstance(c, tuple):
                if first_rgb is None:
                    first_rgb = c
            elif first_scalar is None:
                first_scalar = c
        if first_rgb is not None and first_scalar is not None:
            raise ValueError(
                f"{filepath}: tracklets use mixed scalar and RGB colours "
                f"(first scalar {first_scalar!r}, first RGB {first_rgb!r}); "
                "no mapping between the two representations is defined"
            )

    if first_rgb is not None:
        color_to_idx: dict[tuple, int] = {}
        for frame_data in pc.values():
            for c in frame_data["label"]:
                if c not in color_to_idx:
                    color_to_idx[c] = len(color_to_idx)
        for frame_data in pc.values():
            frame_data["label"] = [color_to_idx[c] for c in frame_data["label"]]
        label_dtype = torch.long
    elif all(isinstance(c, (bool, int)) for frame_data in pc.values() for c in frame_data["label"]):
        # integer colours, or no non-empty frame at all
        label_dtype = torch.long
    else:
        label_dtype = torch.float32

    for i in pc:
        pc[i] = zRegPointCloud(
            pos=torch.tensor(pc[i]["pos"], device=device),
            label=torch.tensor(pc[i]["label"], dtype=label_dtype, device=device),
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
    'pos', 'label', and 'id', and the values are either PyTorch tensors or Open3D
    tensors. It converts the dictionary to an Open3D point cloud object.

    Args:
        pc: A dictionary representing the point cloud. The keys should be 'pos',
            'label', and 'id', and the values should be either PyTorch tensors or
            Open3D tensors.

    Returns:
        An Open3D point cloud object.

    Raises:
        KeyError: If the input dictionary does not contain the keys 'pos', 'label',
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
        map_to_tensors["colors"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["label"]))
        map_to_tensors["labels"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["id"]))
        if pc["fps-idx"] is not None:
            map_to_tensors["fps_idx"] = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc["fps-idx"]))
    else:
        map_to_tensors["positions"] = pc["pos"]
        map_to_tensors["colors"] = pc["label"]
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
        "pos", "label", and "id", and the values are either Torch tensors
        or NumPy arrays.

    Raises
    ------
    RuntimeError
        If open3d is not available.

    Examples
    --------
    This example needs the optional Open3D package, which is not installable
    on every platform (for example aarch64 JUPITER), so its lines are marked
    ``+SKIP``; ``tests/test_doctests.py`` runs it whenever Open3D is
    importable.

    >>> import open3d as o3d  # doctest: +SKIP
    >>> import torch  # doctest: +SKIP
    >>> from zreg.core.dataset import open3d_to_zreg  # doctest: +SKIP
    >>> pc = o3d.t.geometry.PointCloud()  # doctest: +SKIP
    >>> f32 = o3d.core.float32  # doctest: +SKIP
    >>> pc.point.positions = o3d.core.Tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=f32)  # doctest: +SKIP
    >>> pc.point.colors = o3d.core.Tensor([[1.0], [2.0]], dtype=f32)  # doctest: +SKIP
    >>> data = open3d_to_zreg(pc, device=torch.device("cpu"))  # doctest: +SKIP
    >>> data["pos"]  # doctest: +SKIP
    tensor([[0., 0., 0.],
            [1., 2., 3.]])
    >>> data["id"] is None  # doctest: +SKIP
    True
    >>> data = open3d_to_zreg(pc, to_torch=False)  # doctest: +SKIP
    >>> type(data["pos"])  # doctest: +SKIP
    <class 'numpy.ndarray'>
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
        ret["label"] = torch.tensor(col, device=device)
        ret["id"] = torch.tensor(ids, device=device) if ids is not None else None
        ret["fps-idx"] = torch.tensor(fps_idx, device=device) if fps_idx is not None else None
    else:
        # Return NumPy arrays
        ret["pos"] = pos
        ret["label"] = col
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
            label=torch.tensor(pcs[i]["labels"], device=device),
            id=torch.tensor(pcs[i]["id"], device=device),
        )
    return pcs
