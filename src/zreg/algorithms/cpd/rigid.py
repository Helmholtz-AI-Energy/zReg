"""Rigid transformation CPD registration."""

import torch

from .base import CoherentPointDrift
from ._types import MstepResult, EstepResult
from ...core import transforms as tf
from ...utils import squared_kernel_sum

__all__ = ["RigidCPD"]


class RigidCPD(CoherentPointDrift):
    """Coherent Point Drift for rigid transformation.

    Aligns point clouds using rotation, translation, and optional scaling.
    The source point cloud is transformed to align with the target.

    Parameters
    ----------
    source : torch.Tensor | None
        Source point cloud data with shape (N, D).
    update_scale : bool
        If True, optimize the scale parameter during registration.
    tf_init_params : dict
        Parameters to initialize the rigid transformation.
    use_color : bool
        Use color information if True.
    use_cuda : bool
        Use CUDA for computations if True.
    log_freq : int
        Log frequency during registration.
    source_colors : torch.Tensor | None
        Color information for source points.

    Examples
    --------
    >>> source = torch.randn(100, 3)
    >>> target = torch.randn(100, 3)
    >>> cpd = RigidCPD(source, update_scale=True)
    >>> result = cpd.registration(target)
    >>> transformed = result.transformation.transform(source)
    """

    def __init__(
        self,
        source: torch.Tensor | None = None,
        update_scale: bool = True,
        tf_init_params: dict | None = None,
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq: int = 100,
        source_colors: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            source,
            use_color=use_color,
            use_cuda=use_cuda,
            log_freq=log_freq,
            source_colors=source_colors,
        )
        if tf_init_params is None:
            tf_init_params = {}
        fact = {}
        if source is not None:
            fact = {"dtype": source.dtype, "device": source.device}
        self._tf_type = tf.RigidTransformation
        self._update_scale = update_scale
        self.transform = None
        self._tf_init_params = tf_init_params
        self._tf_init_params.update(fact)

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """Initialize rigid registration parameters.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud.

        Returns
        -------
        MstepResult
            Initial transformation, sigma2, and q values.
        """
        sigma2 = squared_kernel_sum(self._source, target)
        q = torch.inf
        if self.transformation is None:
            self.transformation = self._tf_type(**self._tf_init_params)
            # Initial rotation matrix (found empirically for Shah->Kobiski data)
            rot = torch.tensor(
                [[-0.0, -1.0, 0.0], [1.0, -0.0, 0.5], [0.0, 0.5, 1.0]],
                dtype=self._source.dtype,
                device=self._source.device,
            )
            self.transformation.rot = rot
        return MstepResult(self.transformation, sigma2, q)

    def reset_transform(self) -> None:
        """Reset the transformation to identity."""
        if self.transformation is not None:
            self.transformation.reset()

    def maximization_step(
        self,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float | None = None,
        source_colors: torch.Tensor | None = None,
        target_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Perform the maximization step with scale update option.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud.
        estep_res : EstepResult
            Result from expectation step.
        sigma2_p : float | None
            Previous variance.
        source_colors : torch.Tensor | None
            Source color information.
        target_colors : torch.Tensor | None
            Target color information.

        Returns
        -------
        MstepResult
            Updated transformation parameters.
        """
        ret = self._maximization_step(
            self._source,
            target,
            estep_res,
            sigma2_p,
            self._update_scale,
            target_colors=target_colors,
            source_colors=source_colors,
        )
        self.transformation = ret.transformation
        return ret

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float | None = None,
        update_scale: bool = True,
        target_colors: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Compute optimal rigid transformation parameters.

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
        update_scale : bool
            If True, update the scale parameter.
        target_colors : torch.Tensor | None
            Target color information (unused, for API compatibility).
        source_colors : torch.Tensor | None
            Source color information (unused, for API compatibility).

        Returns
        -------
        MstepResult
            Optimal rotation, translation, scale, and updated variance.
        """
        pt1, p1, px, n_p, _ = estep_res
        dim = CoherentPointDrift._N_DIM

        # Calculate means
        mu_x = torch.sum(px, axis=0) / n_p
        mu_y = (source.T @ p1.unsqueeze(1)).squeeze() / n_p

        # Center point clouds
        target_hat = target - mu_x
        source_hat = source - mu_y

        # Cross-covariance matrix
        a = torch.matmul(px.T, source_hat) - torch.outer(
            mu_x, (p1.unsqueeze(0) @ source_hat).squeeze()
        )

        # Optimal rotation via SVD
        u, _, vh = torch.linalg.svd(a, full_matrices=False)
        c = torch.ones(dim, dtype=a.dtype, device=a.device)
        c[-1] = torch.linalg.det(torch.matmul(u, vh))
        rot = torch.matmul(u * c, vh)

        # Optimal scale (Eq. 9-10 from Myronenko & Song 2010)
        tr_atr = torch.trace(torch.matmul(a.T, rot))
        tr_yp1y = torch.trace(torch.matmul(source_hat.T * p1, source_hat))
        if update_scale:
            scale = tr_atr / torch.clamp(tr_yp1y, min=torch.finfo(a.dtype).eps)
        else:
            scale = 1.0

        # Optimal translation
        t = mu_x - scale * torch.matmul(rot, mu_y)
        tr_xp1x = torch.trace(torch.matmul(target_hat.T * pt1, target_hat))

        # Update variance (Eq. 23 from Myronenko & Song 2010). With the
        # optimal scale, s*tr(Y'P1Y) == tr(A'R) and the s^2 term collapses;
        # with a fixed scale s = 1 all three trace terms remain.
        if update_scale:
            sigma2 = (tr_xp1x - scale * tr_atr) / (n_p * dim)
        else:
            sigma2 = (tr_xp1x - 2.0 * tr_atr + tr_yp1y) / (n_p * dim)
        sigma2 = torch.clamp(sigma2, min=torch.finfo(a.dtype).eps)

        # Objective function
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale**2) * tr_yp1y) / (
            2.0 * sigma2 + dim * n_p * 0.5 * torch.log(sigma2)
        )

        return MstepResult(tf.RigidTransformation(rot, t, scale), sigma2, q)
