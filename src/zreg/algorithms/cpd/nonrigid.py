"""Non-rigid transformation CPD registration."""

import torch

from .base import CoherentPointDrift
from ._types import MstepResult, EstepResult
from .kernels import rbf_kernel_matrix
from ...core import transforms as tf
from ...utils import normalize_point_cloud, squared_kernel_sum
from ...utils.validation import _validate_tensors

__all__ = ["NonRigidCPD", "ConstrainedNonRigidCPD"]


class NonRigidCPD(CoherentPointDrift):
    """Coherent Point Drift for non-rigid transformation.

    Uses a Gaussian RBF kernel to model smooth non-rigid deformations.

    Parameters
    ----------
    source : torch.Tensor | None
        Source point cloud data with shape (N, D).
    beta : float
        RBF kernel bandwidth. Larger values produce smoother deformations.
    lmd : float
        Regularization parameter controlling deformation smoothness.
    use_color : bool
        Not supported; must be False. ``True`` raises NotImplementedError
        (colour-assisted CPD requires RigidCPD).
    use_cuda : bool
        Use CUDA for computations if True.
    log_freq : int
        Log frequency during registration.
    """

    def __init__(
        self,
        source: torch.Tensor | None = None,
        beta: float = 2.0,
        lmd: float = 2.0,
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq: int = 100,
    ) -> None:
        if use_color:
            raise NotImplementedError("use_color=True is only supported by RigidCPD")
        super().__init__(source, use_color=use_color, use_cuda=use_cuda, log_freq=log_freq)
        self._tf_type = tf.NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self._normalized_source = None
        self._tf_obj = None
        if self._source is not None:
            self._normalized_source, _ = normalize_point_cloud(self._source)
            self._tf_obj = self._tf_type(None, self._normalized_source, self._beta)

    def set_source(self, source: torch.Tensor, source_colors: torch.Tensor | None = None) -> None:
        """Set source and initialize the non-rigid transformation object.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud.
        source_colors : torch.Tensor | None
            Source color information.
        """
        _validate_tensors(source, names=["source"])
        if source_colors is not None:
            _validate_tensors(source, source_colors, names=["source", "source_colors"])
        self._source = source
        self._normalized_source, _ = normalize_point_cloud(self._source)
        self._tf_obj = self._tf_type(None, self._normalized_source, self._beta)

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """Initialize non-rigid registration parameters.

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
        self._tf_obj.w = torch.zeros_like(self._source)
        return MstepResult(self._tf_obj, sigma2, q)

    def maximization_step(
        self,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float | None = None,
        target_colors: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Perform non-rigid maximization step.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud.
        estep_res : EstepResult
            Result from expectation step.
        sigma2_p : float | None
            Previous variance.
        target_colors : torch.Tensor | None
            Target color information.
        source_colors : torch.Tensor | None
            Source color information.

        Returns
        -------
        MstepResult
            Updated transformation parameters.
        """
        result = self._maximization_step(
            self._source[:, : self._N_DIM],
            target[:, : self._N_DIM],
            estep_res,
            sigma2_p,
            self._tf_obj,
            self._lmd,
        )
        self.transformation = result.transformation
        return result

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float,
        tf_obj: tf.NonRigidTransformation,
        lmd: float,
    ) -> MstepResult:
        """Compute optimal non-rigid deformation parameters.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud.
        target : torch.Tensor
            Target point cloud.
        estep_res : EstepResult
            Result from expectation step.
        sigma2_p : float
            Previous variance.
        tf_obj : NonRigidTransformation
            Transformation object with RBF kernel.
        lmd : float
            Regularization parameter.

        Returns
        -------
        MstepResult
            Optimal deformation weights and updated variance.
        """
        pt1, p1, px, n_p, _ = estep_res
        dim = CoherentPointDrift._N_DIM

        # Solve for deformation parameters (w)
        w = torch.linalg.solve(
            (p1 * tf_obj.g).T
            + lmd * sigma2_p * torch.eye(source.shape[0], dtype=source.dtype, device=source.device),
            px - (source.T * p1).T,
        )

        # Update transformed source points
        t = source + torch.matmul(tf_obj.g, w)
        tr_xp1x = torch.trace(torch.matmul(target.T * pt1, target))
        tr_pxt = torch.trace(torch.matmul(px.T, t))
        tr_tpt = torch.trace(torch.matmul(t.T * p1, t))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)


class ConstrainedNonRigidCPD(CoherentPointDrift):
    """Extended CPD for non-rigid transformation with point constraints.

    Incorporates prior knowledge about correspondence between specific
    points in source and target point clouds.

    See: https://people.mpi-inf.mpg.de/~golyanik/04_DRAFTS/ECPD2016.pdf

    Parameters
    ----------
    source : torch.Tensor | None
        Source point cloud data.
    beta : float
        RBF kernel bandwidth.
    lmd : float
        Regularization parameter.
    alpha : float
        Degree of reliability of priors. Range: 1e-8 (highly reliable)
        to 1 (highly unreliable).
    use_color : bool
        Not supported; must be False. ``True`` raises NotImplementedError
        (colour-assisted CPD requires RigidCPD).
    use_cuda : bool
        Use CUDA for computations if True.
    idx_source : torch.Tensor | None
        Indices in source for known correspondences.
    idx_target : torch.Tensor | None
        Indices in target for known correspondences.
    log_freq : int
        Log frequency during registration.
    """

    def __init__(
        self,
        source: torch.Tensor | None = None,
        beta: float = 2.0,
        lmd: float = 2.0,
        alpha: float = 1e-8,
        use_color: bool = False,
        use_cuda: bool = False,
        idx_source: torch.Tensor | None = None,
        idx_target: torch.Tensor | None = None,
        log_freq: int = 100,
    ) -> None:
        if use_color:
            raise NotImplementedError("use_color=True is only supported by RigidCPD")
        super().__init__(source, use_color=use_color, use_cuda=use_cuda, log_freq=log_freq)
        self._tf_type = tf.NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self.alpha = alpha
        self._normalized_source = None
        self._tf_obj = None
        self.idx_source = idx_source
        self.idx_target = idx_target
        if self._source is not None:
            self._normalized_source, _ = normalize_point_cloud(self._source)
            self._tf_obj = self._tf_type(None, self._normalized_source, self._beta)

    def set_source(self, source: torch.Tensor, source_colors: torch.Tensor | None = None) -> None:
        """Set source and initialize the transformation object.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud.
        source_colors : torch.Tensor | None
            Source color information.
        """
        _validate_tensors(source, names=["source"])
        if source_colors is not None:
            _validate_tensors(source, source_colors, names=["source", "source_colors"])
        self._source = source
        self._normalized_source, _ = normalize_point_cloud(self._source)
        self._tf_obj = self._tf_type(None, self._normalized_source, self._beta)

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """Initialize constrained non-rigid registration parameters.

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
        self._tf_obj.w = torch.zeros_like(self._source)
        self.p_tilde = torch.zeros(
            (self._source.shape[0], target.shape[0]), dtype=target.dtype, device=target.device
        )
        if self.idx_source is not None and self.idx_target is not None:
            self.p_tilde[self.idx_source, self.idx_target] = 1
        self.p1_tilde = torch.sum(self.p_tilde, dim=1)
        self.px_tilde = torch.matmul(self.p_tilde, target)
        return MstepResult(self._tf_obj, sigma2, q)

    def maximization_step(
        self,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float | None = None,
        target_colors: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Perform constrained non-rigid maximization step.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud.
        estep_res : EstepResult
            Result from expectation step.
        sigma2_p : float | None
            Previous variance.
        target_colors : torch.Tensor | None
            Target color information.
        source_colors : torch.Tensor | None
            Source color information.

        Returns
        -------
        MstepResult
            Updated transformation parameters.
        """
        result = self._maximization_step(
            self._source[:, : self._N_DIM],
            target[:, : self._N_DIM],
            estep_res,
            sigma2_p,
            self._tf_obj,
            self._lmd,
            self.alpha,
            self.p1_tilde,
            self.px_tilde,
        )
        self.transformation = result.transformation
        return result

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float,
        tf_obj: tf.NonRigidTransformation,
        lmd: float,
        alpha: float,
        p1_tilde: torch.Tensor,
        px_tilde: torch.Tensor,
    ) -> MstepResult:
        """Compute optimal constrained non-rigid deformation parameters.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud.
        target : torch.Tensor
            Target point cloud.
        estep_res : EstepResult
            Result from expectation step.
        sigma2_p : float
            Previous variance.
        tf_obj : NonRigidTransformation
            Transformation object.
        lmd : float
            Regularization parameter.
        alpha : float
            Prior reliability degree.
        p1_tilde : torch.Tensor
            Sum of constraint probabilities.
        px_tilde : torch.Tensor
            Weighted sum for constraints.

        Returns
        -------
        MstepResult
            Optimal deformation weights and updated variance.
        """
        pt1, p1, px, n_p, _ = estep_res
        dim = CoherentPointDrift._N_DIM

        # Solve for transformation parameters with constraints
        w = torch.linalg.solve(
            (p1 * tf_obj.g).T
            + sigma2_p / alpha * (p1_tilde * tf_obj.g).T
            + lmd * sigma2_p * torch.eye(source.shape[0], dtype=source.dtype, device=source.device),
            px - (source.T * p1).T + sigma2_p / alpha * (px_tilde - (source.T * p1_tilde).T),
        )
        t = source + torch.matmul(tf_obj.g, w)
        tr_xp1x = torch.trace(torch.matmul(target.T * pt1, target))
        tr_pxt = torch.trace(torch.matmul(px.T, t))
        tr_tpt = torch.trace(torch.matmul(t.T * p1, t))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)
