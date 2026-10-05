"""Type definitions for CPD module."""

from collections import namedtuple

__all__ = ["EstepResult", "MstepResult"]

EstepResult = namedtuple("EstepResult", ["pt1", "p1", "px", "n_p", "pmat"])
EstepResult.__doc__ = """Result of Expectation step.

Attributes:
    pt1 (torch.Tensor): Sum of posterior probabilities over source points (shape: M).
    p1 (torch.Tensor): Sum of posterior probabilities over target points (shape: N).
    px (torch.Tensor): Weighted sum of target points (shape: N x D).
    n_p (torch.Tensor): Total probability mass (scalar).
    pmat (torch.Tensor): Full posterior probability matrix (shape: N x M).
"""

MstepResult = namedtuple(
    "MstepResult",
    ["transformation", "sigma2", "q", "n_iters", "sigma2_history"],
    defaults=(None, None),
)
MstepResult.__doc__ = """Result of Maximization step.

Attributes:
    transformation (tf.Transformation): Transformation from source to target.
    sigma2 (float): Variance of Gaussian distribution.
    q (float): Result of likelihood.
    n_iters (int): Number of EM iterations executed during registration.
    sigma2_history (list[float]): sigma2 value recorded after each M-step
        (length == n_iters). Only populated by the final return of
        registration(); intermediate MstepResult objects carry None.
"""
