"""Immutable rigid and affine transform wrappers for synthetic trajectories.

Accepts a ``dict[int, zRegPointCloud]`` and returns a new dict with the
transformation applied to each frame's ``pos`` field.  The input dict is
never mutated (D-03 immutability contract: deep-copy before modifying).

4×4 homogeneous matrix construction follows the pattern established in
``zreg.evaluation.label_transfer._rigid_to_matrix`` /
``_affine_to_matrix``:

  Rigid:  M[:3, :3] = scale * rot;  M[:3, 3] = t;  M[3, 3] = 1.0
  Affine: M[:3, :3] = b;            M[:3, 3] = t;  M[3, 3] = 1.0

The w-coordinate defensive divide pattern from
``zreg.core.transforms.homogeneous`` is applied to guard against
near-zero w values (clamped to ``torch.finfo(dtype).eps``).
"""

import copy

import torch

from zreg.core.dataset import zRegPointCloud
from zreg.core.transforms import AffineTransformation, RigidTransformation

__all__ = ["apply_affine", "apply_rigid"]


def _rigid_to_matrix(tf: RigidTransformation) -> torch.Tensor:
    """Build a 4×4 homogeneous matrix from a RigidTransformation.

    Parameters
    ----------
    tf : RigidTransformation
        Rigid transformation with ``.rot`` (3×3), ``.t`` (3,), ``.scale``.

    Returns
    -------
    torch.Tensor
        4×4 homogeneous transformation matrix on ``tf.rot.device``.
    """
    device = tf.rot.device
    dtype = tf.rot.dtype
    M = torch.zeros(4, 4, device=device, dtype=dtype)
    M[:3, :3] = tf.scale * tf.rot
    M[:3, 3] = tf.t
    M[3, 3] = 1.0
    return M


def _affine_to_matrix(tf: AffineTransformation) -> torch.Tensor:
    """Build a 4×4 homogeneous matrix from an AffineTransformation.

    Parameters
    ----------
    tf : AffineTransformation
        Affine transformation with ``.b`` (3×3) and ``.t`` (3,).

    Returns
    -------
    torch.Tensor
        4×4 homogeneous transformation matrix on ``tf.b.device``.
    """
    device = tf.b.device
    dtype = tf.b.dtype
    M = torch.zeros(4, 4, device=device, dtype=dtype)
    M[:3, :3] = tf.b
    M[:3, 3] = tf.t
    M[3, 3] = 1.0
    return M


def _apply_matrix(
    trajectory: dict[int, zRegPointCloud],
    M: torch.Tensor,
) -> dict[int, zRegPointCloud]:
    """Apply a 4×4 homogeneous matrix to every frame's ``pos`` field.

    Returns a deep copy of *trajectory* with ``pos`` replaced by the
    transformed coordinates.  All other fields (``color``, ``id``,
    ``fps-idx``) are left unchanged.

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Input trajectory.  Never modified.
    M : torch.Tensor
        4×4 homogeneous transformation matrix.

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory with transformed ``pos`` tensors.
    """
    result = copy.deepcopy(trajectory)
    for pc in result.values():
        pos = pc["pos"]
        # Cast matrix to pos dtype and device
        M_cast = M.to(dtype=pos.dtype, device=pos.device)
        # Build homogeneous coordinates: (N, 4)
        ones = torch.ones((pos.shape[0], 1), dtype=pos.dtype, device=pos.device)
        homo = torch.hstack([pos, ones])
        # Apply transform: (N, 4) @ (4, 4).T  →  (N, 4)
        transformed_full = homo @ M_cast.T
        # Defensive w-divide: clamp magnitude while preserving sign to avoid
        # sign flip (e.g. clamping w=-1.0 to +eps would invert all coords).
        # For w near zero, default to +eps (same as existing behaviour for
        # the rigid/affine use-case where M[3,3]=1 so w is always 1.0).
        w = transformed_full[:, 3:]
        eps = torch.finfo(pos.dtype).eps
        w = torch.where(w.abs() < eps, torch.full_like(w, eps) * w.sign().clamp(min=1), w)
        pc["pos"] = transformed_full[:, :3] / w
    return result


def apply_rigid(
    trajectory: dict[int, zRegPointCloud],
    tf: RigidTransformation,
) -> dict[int, zRegPointCloud]:
    """Apply a rigid transformation to every frame of a trajectory.

    Returns a new ``dict[int, zRegPointCloud]`` with each frame's ``pos``
    updated by the transformation.  The input *trajectory* is never modified
    (D-03 immutability contract).

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Source trajectory.  Not modified.
    tf : RigidTransformation
        Rigid transformation (rotation, translation, scale).

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory with transformed ``pos``; all other fields preserved.

    Examples
    --------
    >>> import torch
    >>> from zreg.core.transforms import RigidTransformation
    >>> from zreg.data_generation import apply_rigid, generate_trajectory
    >>> traj = generate_trajectory(50, 3, seed=0)
    >>> result = apply_rigid(traj, RigidTransformation())  # identity
    >>> torch.allclose(result[0]["pos"], traj[0]["pos"])
    True
    """
    M = _rigid_to_matrix(tf)
    return _apply_matrix(trajectory, M)


def apply_affine(
    trajectory: dict[int, zRegPointCloud],
    tf: AffineTransformation,
) -> dict[int, zRegPointCloud]:
    """Apply an affine transformation to every frame of a trajectory.

    Returns a new ``dict[int, zRegPointCloud]`` with each frame's ``pos``
    updated by the transformation.  The input *trajectory* is never modified
    (D-03 immutability contract).

    Parameters
    ----------
    trajectory : dict[int, zRegPointCloud]
        Source trajectory.  Not modified.
    tf : AffineTransformation
        Affine transformation (linear matrix ``b``, translation ``t``).

    Returns
    -------
    dict[int, zRegPointCloud]
        New trajectory with transformed ``pos``; all other fields preserved.

    Examples
    --------
    >>> import torch
    >>> from zreg.core.transforms import AffineTransformation
    >>> from zreg.data_generation import apply_affine, generate_trajectory
    >>> traj = generate_trajectory(50, 3, seed=0)
    >>> tf = AffineTransformation(t=torch.zeros(3))  # identity (also the default)
    >>> result = apply_affine(traj, tf)
    >>> torch.allclose(result[0]["pos"], traj[0]["pos"])
    True
    """
    M = _affine_to_matrix(tf)
    return _apply_matrix(trajectory, M)
