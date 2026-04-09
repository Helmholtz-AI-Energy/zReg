from .. import utils
import torch
from ..validation import _validate_tensors

__all__ = ["euclidean_distance", "manhattan_distance", "minkowski_distance"]


def euclidean_distance(x: torch.Tensor, y: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Calculate the Euclidean distance between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor.
    y : torch.Tensor
        Second tensor.
    normalize : bool, optional
        Whether to normalize the tensors before calculating the distance, by default False.

    Returns
    -------
    torch.Tensor
        Euclidean distance between x and y.
    """
    return minkowski_distance(x, y, p=2, normalize=normalize)


def manhattan_distance(x: torch.Tensor, y: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Calculate the Manhattan distance between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor.
    y : torch.Tensor
        Second tensor.
    normalize : bool, optional
        Whether to normalize the tensors before calculating the distance, by default False.

    Returns
    -------
    torch.Tensor
        Manhattan distance between x and y.
    """
    return minkowski_distance(x, y, p=1, normalize=normalize)


def minkowski_distance(x: torch.Tensor, y: torch.Tensor, p: float = 2, normalize: bool = False) -> torch.Tensor:
    """
    Calculate the Minkowski distance between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        First tensor.
    y : torch.Tensor
        Second tensor.
    p : float, optional
        Order of the Minkowski distance, by default 2.
    normalize : bool, optional
        Whether to normalize the tensors before calculating the distance, by default False.

    Returns
    -------
    torch.Tensor
        Minkowski distance between x and y.

    Notes
    -----
    This function returns a tensor of shape (x.shape[0], y.shape[0]) where each element (i, j)
    is the Minkowski distance between x[i] and y[j].
    """
    _validate_tensors(x, y, names=["x", "y"])
    # TODO: add option to pass min/max to normalization function within this function
    if normalize:
        x, _ = utils.normalize_point_cloud(x)
        y, _ = utils.normalize_point_cloud(y)

    return torch.cdist(x, y, p=2)

    # if p == 1:
    #     return (x[None, :, :] - y[:, None, :]).abs().sum(dim=2)

    # if p // 2:  # even, no need for abs
    #     return (x[None, :, :] - y[:, None, :]).pow(p).sum(dim=2).pow(1 / float(p))
    # # else need to have abs and pows
    # return (x[None, :, :] - y[:, None, :]).abs().pow(p).sum(dim=2).pow(1 / float(p))
