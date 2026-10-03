"""Rigid transformation CPD registration."""

import torch

from .base import CoherentPointDrift, _mstep_accumulation_dtype, _upcast_estep
from ._types import MstepResult, EstepResult
from ...core import transforms as tf
from ...utils import squared_kernel_sum

__all__ = ["RigidCPD", "SHAH_KOBITSKI_EMPIRICAL_INIT"]

#: Historical "Shah->Kobitski" initial 3x3 matrix, kept as an opt-in constant.
#:
#: This is a general linear initialisation with det = 1, NOT a rotation: its
#: row norms are 1, 1.118 and 1.118, so it shears and scales anisotropically.
#: It was found empirically for raw (unnormalised) Shah->Kobitski data and was
#: previously hard-coded as the RigidCPD start. The values are kept exactly
#: and deliberately not orthogonalised, so opting in reproduces the old
#: behaviour::
#:
#:     RigidCPD(src, tf_init_params={"rot": torch.tensor(SHAH_KOBITSKI_EMPIRICAL_INIT)})
#:
#: RigidCPD starts from the identity by default; no zreg/eval pipeline path
#: passes this constant.
SHAH_KOBITSKI_EMPIRICAL_INIT = (
    (-0.0, -1.0, 0.0),
    (1.0, -0.0, 0.5),
    (0.0, 0.5, 1.0),
)


class RigidCPD(CoherentPointDrift):
    """Coherent Point Drift for rigid transformation.

    Aligns point clouds using rotation, translation, and optional scaling.
    The source point cloud is transformed to align with the target.

    Parameters
    ----------
    source : torch.Tensor | None
        Source point cloud data with shape (N, D).
    update_scale : bool
        If True, optimize the scale parameter during registration. If False,
        the scale stays at the initial transformation's scale
        (``tf_init_params["scale"]``, default 1, or that of a pre-set
        transformation).
    tf_init_params : dict
        Parameters to initialize the rigid transformation (``rot``, ``t``,
        ``scale``). The default start is the identity rotation with zero
        translation. To reproduce the historical start, pass
        ``{"rot": torch.tensor(SHAH_KOBITSKI_EMPIRICAL_INIT)}``. The dict is
        copied, never mutated.
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
        self._tf_init_params = dict(tf_init_params)
        self._tf_init_params.update(fact)

    def set_source(
        self, source: torch.Tensor, source_colors: torch.Tensor | None = None
    ) -> None:
        """Set the source and refresh the default transform's dtype/device.

        The default identity is built from ``_tf_init_params`` in
        ``_initialize``; refreshing dtype/device here keeps it in the source
        dtype when the source is supplied after construction. An already set
        transformation is never replaced or cast.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud.
        source_colors : torch.Tensor | None
            Color information for source points.
        """
        super().set_source(source, source_colors)
        self._tf_init_params.update({"dtype": source.dtype, "device": source.device})

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
        fixed_scale = (
            self.transformation.scale
            if self.transformation is not None
            else self._tf_init_params.get("scale", 1.0)
        )
        ret = self._maximization_step(
            self._source,
            target,
            estep_res,
            sigma2_p,
            self._update_scale,
            target_colors=target_colors,
            source_colors=source_colors,
            fixed_scale=fixed_scale,
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
        fixed_scale: float | torch.Tensor = 1.0,
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
        fixed_scale : float | torch.Tensor
            Scale kept when ``update_scale`` is False (ignored otherwise).

        Returns
        -------
        MstepResult
            Optimal rotation, translation, scale, and updated variance.
        """
        # Reduce in float64 for float32 input and cast the results back
        # (see _mstep_accumulation_dtype): sigma2 is a cancelling difference.
        out_dtype = source.dtype
        acc = _mstep_accumulation_dtype(out_dtype)
        source, target = source.to(acc), target.to(acc)
        pt1, p1, px, n_p, _ = _upcast_estep(estep_res, acc)
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
        elif isinstance(fixed_scale, torch.Tensor):
            scale = fixed_scale.to(dtype=a.dtype, device=a.device)
        else:
            scale = float(fixed_scale)

        # Optimal translation
        t = mu_x - scale * torch.matmul(rot, mu_y)
        tr_xp1x = torch.trace(torch.matmul(target_hat.T * pt1, target_hat))

        # Update variance (Eq. 23 from Myronenko & Song 2010). With the
        # optimal scale, s*tr(Y'P1Y) == tr(A'R) and the s^2 term collapses;
        # with a fixed scale s all three trace terms remain (s = 1 by default).
        if update_scale:
            sigma2 = (tr_xp1x - scale * tr_atr) / (n_p * dim)
        else:
            sigma2 = (tr_xp1x - 2.0 * scale * tr_atr + (scale**2) * tr_yp1y) / (
                n_p * dim
            )
        sigma2 = torch.clamp(sigma2, min=torch.finfo(a.dtype).eps)

        # Objective function: M&S negative log-likelihood; may be negative.
        # DTW uses sigma2, not q (D-02, pairwise_distance_matrix._cpd_dtw_cost).
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale**2) * tr_yp1y) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * torch.log(sigma2)

        if isinstance(scale, torch.Tensor):
            scale = scale.to(out_dtype)
        return MstepResult(
            tf.RigidTransformation(rot.to(out_dtype), t.to(out_dtype), scale),
            sigma2.to(out_dtype),
            q.to(out_dtype),
        )
