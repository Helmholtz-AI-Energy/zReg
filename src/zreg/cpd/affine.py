"""Affine transformation CPD registration."""

import torch

from .base import CoherentPointDrift
from ._types import MstepResult, EstepResult
from .. import transforms as tf
from ..utils import squared_kernel_sum

__all__ = ["AffineCPD"]


class AffineCPD(CoherentPointDrift):
    """Coherent Point Drift for affine transformation.

    Aligns point clouds using an affine transformation (rotation, translation,
    scaling, and shearing).

    Parameters
    ----------
    source : torch.Tensor | None
        Source point cloud data with shape (N, D).
    tf_init_params : dict
        Parameters to initialize the affine transformation.
    use_color : bool
        Use color information if True.
    use_cuda : bool
        Use CUDA for computations if True.
    log_freq : int
        Log frequency during registration.
    """

    def __init__(
        self,
        source: torch.Tensor | None = None,
        tf_init_params: dict | None = None,
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq: int = 100,
    ) -> None:
        super().__init__(source, use_color=use_color, use_cuda=use_cuda, log_freq=log_freq)
        if tf_init_params is None:
            tf_init_params = {}
        self._tf_type = tf.AffineTransformation
        self._tf_init_params = tf_init_params

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """Initialize affine registration parameters.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud.

        Returns
        -------
        MstepResult
            Initial transformation, sigma2, and q values.
        """
        dim = self._N_DIM
        sigma2 = squared_kernel_sum(self._source[:, :dim], target[:, :dim])
        q = 1.0 + target.shape[0] * dim * 0.5 * torch.log(sigma2)
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float | None = None,
        target_colors: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Compute optimal affine transformation parameters.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud.
        target : torch.Tensor
            Target point cloud.
        estep_res : EstepResult
            Result from expectation step.
        sigma2_p : float | None
            Previous variance.
        target_colors : torch.Tensor | None
            Target color information (unused).
        source_colors : torch.Tensor | None
            Source color information (unused).

        Returns
        -------
        MstepResult
            Optimal affine matrix, translation, and updated variance.
        """
        pt1, p1, px, n_p, _ = estep_res
        dim = CoherentPointDrift._N_DIM

        # Get means
        mu_x = torch.sum(px, dim=0) / n_p
        mu_y = torch.matmul(source.T, p1) / n_p

        # Center point clouds
        target_hat = target - mu_x
        source_hat = source - mu_y

        # Compute affine transformation parameters
        a = torch.matmul(px.T, source_hat) - torch.outer(
            mu_x, (p1.unsqueeze(0) @ source_hat).squeeze()
        )
        yp1y = torch.matmul(source_hat.T * p1, source_hat)
        b = torch.linalg.solve(yp1y.T, a.T).T  # Affine matrix
        t = mu_x - torch.matmul(b, mu_y)  # Translation

        # Update sigma2
        tr_xp1x = torch.trace(torch.matmul(target_hat.T * pt1, target_hat))
        tr_xpyb = torch.trace(torch.matmul(a, b.T))
        sigma2 = (tr_xp1x - tr_xpyb) / (n_p * dim)
        sigma2 = torch.clamp(sigma2, min=torch.finfo(a.dtype).eps)

        # Update q (negative log-likelihood)
        tr_ab = torch.trace(torch.matmul(a, b.T))
        q = (tr_xp1x - 2 * tr_ab + tr_xpyb) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * torch.log(sigma2)

        return MstepResult(tf.AffineTransformation(b, t), sigma2, q)
