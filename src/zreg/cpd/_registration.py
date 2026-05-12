"""CPD registration convenience functions.

This module provides high-level functions for CPD registration that abstract
away the class instantiation details.
"""

from typing import Any, Callable

import torch
try:
    import open3d as o3d
    HAS_OPEN3D = True
except (ImportError, OSError):  # pragma: no cover
    HAS_OPEN3D = False  # pragma: no cover
    o3d = None  # pragma: no cover

from ._types import MstepResult
from .rigid import RigidCPD
from .affine import AffineCPD
from .nonrigid import NonRigidCPD, ConstrainedNonRigidCPD
from ..dataset import zRegPointCloud, open3d_to_zreg
from .. import transforms as tf

__all__ = ["cpd_registration", "init_cpd_from_existing"]


def cpd_registration(
    source: "zRegPointCloud | o3d.t.geometry.PointCloud",
    target: "zRegPointCloud | o3d.t.geometry.PointCloud",
    tf_type_name: str = "rigid",
    w: float = 0.0,
    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: list[Callable] | None = None,
    use_color: bool = False,
    log_freq: int = 100,
    **kwargs: Any,
) -> MstepResult:
    """Perform CPD registration between two point clouds.

    This is a convenience function that instantiates the appropriate CPD
    class based on the transformation type and runs registration.

    Parameters
    ----------
    source : zRegPointCloud or o3d.t.geometry.PointCloud
        Source point cloud data.
    target : zRegPointCloud or o3d.t.geometry.PointCloud
        Target point cloud data.
    tf_type_name : str
        Transformation type: 'rigid', 'affine', 'nonrigid', or
        'nonrigid_constrained'.
    w : float
        Weight of uniform distribution for outlier handling (0 < w < 1).
    maxiter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    callbacks : list[Callable] | None
        Functions called after each iteration with the transformation.
    use_color : bool
        Use color information if available.
    log_freq : int
        Log frequency during registration (-1 to disable).
    **kwargs
        Additional arguments passed to the CPD class constructor:
        - update_scale (bool): For 'rigid', whether to optimize scale.
        - tf_init_params (dict): Initial transformation parameters.
        - beta (float): RBF kernel bandwidth for nonrigid.
        - lmd (float): Regularization for nonrigid.
        - alpha (float): Prior reliability for constrained nonrigid.
        - idx_source, idx_target (Tensor): Constraint indices.

    Returns
    -------
    MstepResult
        Registration result with transformation, sigma2, q, n_iters,
        and sigma2_history.

    Raises
    ------
    ValueError
        If tf_type_name is not a recognized transformation type.

    Examples
    --------
    >>> source_pc = {"pos": torch.randn(100, 3)}
    >>> target_pc = {"pos": torch.randn(100, 3)}
    >>> result = cpd_registration(source_pc, target_pc, tf_type_name="rigid")
    >>> transformed = result.transformation.transform(source_pc["pos"])
    """
    if callbacks is None:
        callbacks = []

    # Convert from Open3D if necessary
    if HAS_OPEN3D and isinstance(source, o3d.t.geometry.PointCloud):
        source = open3d_to_zreg(source)
    if HAS_OPEN3D and isinstance(target, o3d.t.geometry.PointCloud):
        target = open3d_to_zreg(target)

    # Prepare point data
    if use_color:
        sourcei = torch.cat([source["pos"], source["color"]], dim=1)
        targeti = torch.cat([target["pos"], target["color"]], dim=1)
    else:
        sourcei = source["pos"]
        targeti = target["pos"]

    # Instantiate appropriate CPD class
    if tf_type_name == "rigid":
        cpd = RigidCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    elif tf_type_name == "affine":
        cpd = AffineCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    elif tf_type_name == "nonrigid":
        cpd = NonRigidCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    elif tf_type_name == "nonrigid_constrained":
        cpd = ConstrainedNonRigidCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    else:
        raise ValueError(f"Unknown transformation type: {tf_type_name}")

    cpd.set_callbacks(callbacks)
    return cpd.registration(targeti, w, maxiter, tol)


def init_cpd_from_existing(
    transform: "tf.RigidTransformation | tf.AffineTransformation",
    source: "zRegPointCloud | o3d.t.geometry.PointCloud",
    target: "zRegPointCloud | o3d.t.geometry.PointCloud",
    w: float = 0.0,
    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: list[Callable] | None = None,
    use_color: bool = False,
    log_freq: int = 100,
) -> RigidCPD | AffineCPD:
    """Initialize a CPD object from an existing transformation.

    Useful for refining a previously computed transformation or continuing
    registration from a known starting point.

    Parameters
    ----------
    transform : RigidTransformation or AffineTransformation
        Existing transformation to use as initial state.
    source : zRegPointCloud or o3d.t.geometry.PointCloud
        Source point cloud data.
    target : zRegPointCloud or o3d.t.geometry.PointCloud
        Target point cloud data.
    w : float
        Weight of uniform distribution for outlier handling.
    maxiter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    callbacks : list[Callable] | None
        Functions called after each iteration.
    use_color : bool
        Use color information if available.
    log_freq : int
        Log frequency during registration.

    Returns
    -------
    RigidCPD or AffineCPD
        Initialized CPD object ready for registration.

    Raises
    ------
    TypeError
        If transform is not a RigidTransformation or AffineTransformation.

    Examples
    --------
    >>> tf = transforms.RigidTransformation(device="cpu", dtype=torch.float32)
    >>> cpd = init_cpd_from_existing(tf, source_pc, target_pc)
    >>> result = cpd.registration(target_pc["pos"])
    """
    if callbacks is None:
        callbacks = []

    # Convert from Open3D if necessary
    if HAS_OPEN3D and isinstance(source, o3d.t.geometry.PointCloud):
        source = open3d_to_zreg(source)
    if HAS_OPEN3D and isinstance(target, o3d.t.geometry.PointCloud):
        target = open3d_to_zreg(target)

    # Prepare point data
    if use_color:
        sourcei = torch.cat([source["pos"], source["color"]], dim=1)
    else:
        sourcei = source["pos"]

    # Create CPD object based on transform type
    if isinstance(transform, tf.RigidTransformation):
        tf_init_params = {
            "device": transform.rot.device,
            "dtype": transform.rot.dtype,
        }
        cpdobj = RigidCPD(
            sourcei,
            use_color=use_color,
            tf_init_params=tf_init_params,
            log_freq=log_freq,
        )
        cpdobj.transformation = transform
    elif isinstance(transform, tf.AffineTransformation):
        tf_init_params = {
            "device": transform.b.device,
            "dtype": transform.b.dtype,
        }
        cpdobj = AffineCPD(
            sourcei,
            use_color=use_color,
            tf_init_params=tf_init_params,
            log_freq=log_freq,
        )
        cpdobj.transformation = transform
    else:
        raise TypeError(f"transform type not known/not implemented: {type(transform)}")

    return cpdobj
