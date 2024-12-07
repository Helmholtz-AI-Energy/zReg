# This file takes insperation from https://github.com/neka-nat/probreg/
# The core algorithms are the same, but the implementation now makes use of pytorch

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union
import torch

import open3d as o3d
# import six
# from scipy.spatial import distance as scipy_distance

# from . import math_utils as mu
from . import transforms as tf
from . import dataset
from .utils import squared_kernel_sum
import logging


log = logging.getLogger(__name__)

# # Enable TF32 for matrix multiplications
# torch.backends.cuda.matmul.allow_tf32 = True

# # Enable TF32 for convolutions (this is the default)
# torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


EstepResult = namedtuple("EstepResult", ["pt1", "p1", "px", "n_p"])
MstepResult = namedtuple("MstepResult", ["transformation", "sigma2", "q"])
MstepResult.__doc__ = """Result of Maximization step.

    Attributes:
        transformation (tf.Transformation): Transformation from source to target.
        sigma2 (float): Variance of Gaussian distribution.
        q (float): Result of likelihood.
"""


# class DistModule:
#     def __init__(self, xp):
#         self.xp = xp

#     def cdist(self, x1, x2, metric):
#         return torch.stack([torch.sum(self.xp.square(x2 - ts), dim=1) for ts in x1])


class CoherentPointDrift:
    """Coherent Point Drift algorithm.

    This is an abstract class for the Coherent Point Drift (CPD) algorithm.
    It provides a common framework for different types of transformations
    (rigid, affine, nonrigid) which are implemented in inherited classes.
    This class implements the Expectation step of the Expectation-Maximization (EM) algorithm.
    The Maximization step is implemented in the inherited classes.

    Parameters
    ----------
    source : torch.Tensor, optional
        Source point cloud data.
    use_color : bool, optional
        Use color information (if available) in the registration process. Default: False.
    use_cuda : bool, optional
        Use CUDA for computations. Default: False.

    Attributes
    ----------
    _N_DIM : int
        Number of dimensions for point cloud data (default: 3).
    _N_COLOR : int
        Number of color channels (default: 3).
    """

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        source_colors: Optional[torch.Tensor] = None,
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq: bool = True,
    ) -> None:
        """Initialize CPD object."""
        self._source = source
        self._source_colors = None
        self._tf_type = None  # Transformation type (set in inherited classes)
        self._callbacks = []  # List of callbacks to be called during registration
        self._use_color = use_color
        if use_color:
            # todo: add raise if source colors not given
            self._source_colors = source_colors
        self.transformation = None
        self.log_freq = log_freq

    def set_source(self, source: torch.Tensor, source_colors: Optional[torch.Tensor] = None) -> None:
        self._source = source
        if self._use_color and source_colors is not None:
            self._source_colors = source_colors

    def set_callbacks(self, callbacks: List[Callable]) -> None:
        self._callbacks.extend(callbacks)

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """Initialize parameters for the registration process.

        This method is called at the beginning of the registration process.
        It should be implemented in inherited classes to initialize
        transformation-specific parameters.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data.

        Returns
        -------
        MstepResult
            Result object containing initial parameters.
        """
        return MstepResult(None, None, None)

    def _compute_pmat_numerator(self, t_source: torch.Tensor, target: torch.Tensor, sigma2: float) -> torch.Tensor:
        """Compute the numerator of the probability matrix.

        This method calculates the Gaussian kernel density estimate between
        transformed source points and target points.

        Parameters
        ----------
        t_source : torch.Tensor
            Transformed source point cloud data.
        target : torch.Tensor
            Target point cloud data.
        sigma2 : float
            Variance of the Gaussian kernel.

        Returns
        -------
        torch.Tensor
            Numerator of the probability matrix.
        """
        pmat = torch.cdist(t_source, target, p=2).pow(2)  # Pairwise Euclidean distance squared
        pmat = torch.exp(-pmat / (2.0 * sigma2))  # Gaussian kernel
        return pmat

    # @torch.compile
    def expectation_step(
        self,
        t_source: torch.Tensor,
        target: torch.Tensor,
        sigma2: float,
        sigma2_c: float,
        w: float = 0.0,
        target_colors: Optional[torch.Tensor] = None,
        source_colors: Optional[torch.Tensor] = None,
    ) -> EstepResult:
        """Perform the Expectation step of the EM algorithm.

        This method calculates the posterior probabilities of the
        correspondence between transformed source points and target points.

        Parameters
        ----------
        t_source : torch.Tensor
            Transformed source point cloud data.
        target : torch.Tensor
            Target point cloud data.
        sigma2 : float
            Variance of the Gaussian kernel for point coordinates.
        sigma2_c : float
            Variance of the Gaussian kernel for color information.
        w : float, optional
            Weight of the uniform distribution (for outlier handling). Default: 0.0.

        Returns
        -------
        EstepResult
            Result object containing the posterior probabilities and other
            intermediate results of the E-step.
        """
        posdims = t_source.shape[1]
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        pmat = self._compute_pmat_numerator(t_source, target, sigma2)

        c = (2.0 * torch.pi * sigma2) ** (posdims * 0.5)
        c *= w / (1.0 - w) * t_source.shape[0] / target.shape[0]
        den = torch.sum(pmat, dim=0)
        den[den == 0] = torch.finfo(target.dtype).eps

        if self._use_color:
            ncolors = source_colors.shape[1]
            pmat_c = self._compute_pmat_numerator(source_colors, target_colors, sigma2_c)
            den_c = torch.sum(pmat_c, dim=0)
            den_c[den_c == 0] = torch.finfo(pmat_c.dtype).eps
            den = torch.multiply(den, den_c)

            # Calculate the contribution of color information to the denominator
            o_c = t_source.shape[0] * (2 * torch.pi * sigma2_c) ** (0.5 * (posdims + ncolors - 1))
            # print(o_c.shape, pmat_c.shape)
            o_c = o_c * torch.exp(-1.0 / t_source.shape[0] * torch.square(torch.sum(pmat_c, dim=0)) / (2.0 * sigma2_c))
            den += o_c
            c *= (2.0 * torch.pi * sigma2_c) ** (ncolors * 0.5)
            pmat = torch.multiply(pmat, pmat_c)  # Combine color and spatial probabilities

        den += c
        pmat = torch.divide(pmat, den)  # Normalize the probabilities

        pt1 = torch.sum(pmat, dim=0)
        p1 = torch.sum(pmat, dim=1)
        px = torch.matmul(pmat, target)  # .dot
        return EstepResult(pt1, p1, px, torch.sum(p1))

    # @torch.compile
    def maximization_step(
        self,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
        target_colors: Optional[torch.Tensor] = None,
        source_colors: Optional[torch.Tensor] = None,
    ) -> MstepResult:
        """Perform the Maximization step of the EM algorithm.

        This method updates the transformation parameters based on the
        posterior probabilities calculated in the E-step.
        It should be implemented in the inherited classes for specific
        transformation types.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data.
        estep_res : EstepResult
            Result object from the Expectation step.
        sigma2_p : float, optional
            Previous variance of the Gaussian kernel.

        Returns
        -------
        MstepResult
            Result object containing updated transformation parameters and
            other relevant information.
        """
        ret = self._maximization_step(
            self._source, target, estep_res, sigma2_p, target_colors=target_colors, source_colors=source_colors
        )
        self.transformation = ret.transformation
        return ret

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
        target_colors: Optional[torch.Tensor] = None,
        source_colors: Optional[torch.Tensor] = None,
    ) -> MstepResult:
        """Internal method for the Maximization step.

        This method is called by the `maximization_step` method and can be
        overridden in inherited classes to provide specific implementations
        for different transformation types.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud data.
        target : torch.Tensor
            Target point cloud data.
        estep_res : EstepResult
            Result object from the Expectation step.
        sigma2_p : float, optional
            Previous variance of the Gaussian kernel.

        Returns
        -------
        Optional[MstepResult]
            Result object containing updated transformation parameters and
            other relevant information.
        """
        return None

    @torch.compile
    def registration(
        self,
        target: torch.Tensor,
        w: float = 0.0,
        maxiter: int = 50,
        tol: float = 0.001,
        target_colors: Optional[torch.Tensor] = None,
    ) -> MstepResult:
        """Perform the CPD registration process.

        This method iteratively executes the E-step and M-step of the EM
        algorithm until convergence or the maximum number of iterations is reached.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data.
        w : float, optional
            Weight of the uniform distribution (for outlier handling). Default: 0.0.
        maxiter : int, optional
            Maximum number of iterations. Default: 50.
        tol : float, optional
            Tolerance for convergence. Default: 0.001.

        Returns
        -------
        MstepResult
            Result object containing the final transformation parameters and
            other registration information.
        """
        assert self._tf_type is not None, "transformation type is None."
        res = self._initialize(target)
        sigma2_c = 0.0
        if self._use_color:
            sigma2_c = squared_kernel_sum(self._source_colors, target_colors)
        q = res.q  # Initial value of the objective function

        # with Progress(
        #     "[progress.description]{task.description}",
        #     BarColumn(),
        #     TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        #     TimeElapsedColumn(),
        #     TextColumn("[progress.percentage]{task.fields[iter_time]}"),
        #     TimeRemainingColumn(),
        #     TextColumn("[progress.percentage]{task.fields[criteria]}"),
        #     disable=not self.progress_bar,
        # ) as progress:
        #     if self.progress_bar:
        #         task = progress.add_task("[cyan]Registering...", total=maxiter, criteria="", iter_time="")
        # start_time = time.perf_counter()
        # src = self._source.clone()
        for i in range(maxiter):
            # iter_start_time = time.perf_counter()
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(
                t_source,
                target,
                res.sigma2,
                sigma2_c,
                w,
                target_colors=target_colors,
                source_colors=self._source_colors,
            )
            res = self.maximization_step(
                target, estep_res, res.sigma2, target_colors=target_colors, source_colors=self._source_colors
            )

            for c in self._callbacks:
                c(res.transformation)

            # iter_end_time = time.perf_counter()
            # iter_time = iter_end_time - iter_start_time
            # elapsed_time = time.perf_counter() - start_time
            # avg_iter_time = elapsed_time / (i + 1)
            # if self.progress_bar:
            #     progress.update(
            #         task,
            #         advance=1,
            #         criteria=f"Criteria: {res.q:.4f}",
            #         iter_time=f"Avg: {avg_iter_time:.2f}s, Iter: {iter_time:.2f}s",
            #     )
            if self.log_freq > 0 and i % self.log_freq == self.log_freq - 1:
                log.info(f"Registering: iteration {i}/{maxiter}, criteria: {res.q:.4f}")

            if self.log_freq > 0 and abs(res.q - q) < tol:
                # log.info(f"Hit tolerance in iteration {i} (criteria: {res.q:.4f}), exiting.")
                break
            q = res.q
        if self.log_freq > 0:
            log.info(f"End registration at step {i} (criteria: {res.q:.5f})")

        return res


class RigidCPD(CoherentPointDrift):
    """Coherent Point Drift for rigid transformation.

    Note: this will rotate the source to the target
        i.e. target = transform(source)

    Args:
        source (torch.Tensor, optional): Source point cloud data.
        update_scale (bool, optional): If this flag is True, compute the scale parameter.
        tf_init_params (dict, optional): Parameters to initialize transformation.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    """
    Coherent Point Drift for rigid transformation.

    This class implements the Coherent Point Drift (CPD) algorithm for aligning two point clouds
    using a rigid transformation (rotation, translation, and optional scaling). The algorithm
    rotates and translates the source point cloud to align with the target point cloud.

    Parameters
    ----------
    source : torch.Tensor, optional
        Source point cloud data. Shape: (n_points, n_dims), where n_dims >= 2.
    update_scale : bool, optional
        If True, the scale parameter is optimized during registration. Default: True.
    tf_init_params : dict, optional
        Parameters to initialize the rigid transformation. Default: {}.
    use_color : bool, optional
        If True, use color information (if available) for registration. Default: False.
    use_cuda : bool, optional
        If True, use CUDA for computations. Default: False.


    Notes
    -----
    This implementation assumes that the target point cloud is fixed and the source point cloud
    is transformed to align with the target.

    Examples
    --------
    >>> source = torch.randn(100, 3)
    >>> target = torch.randn(100, 3)
    >>> cpd = RigidCPD(source, update_scale=True)
    >>> tf_param, sigma2, q = cpd.registration(target)
    >>> transformed_source = tf_param.transform(source)
    """

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        update_scale: bool = True,
        tf_init_params: Dict = {},
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq: bool = 100,
        source_colors: Optional[torch.Tensor] = None,
    ) -> None:
        super(RigidCPD, self).__init__(
            source, use_color=use_color, use_cuda=use_cuda, log_freq=log_freq, source_colors=source_colors
        )
        fact = {"dtype": source.dtype, "device": source.device}
        self._tf_type = tf.RigidTransformation
        self._update_scale = update_scale
        self.transform = None
        self._tf_init_params = tf_init_params
        self._tf_init_params.update(fact)
        self.log_freq = log_freq

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """
        Initialize the registration parameters.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data. Shape: (n_points, n_dims).

        Returns
        -------
        MstepResult
            Initialization result containing the initial transformation,
            initial variance, and initial objective function value.
        """
        sigma2 = squared_kernel_sum(self._source, target)
        # Initialize Q with a reasonable value based on the initial variance
        q = torch.inf
        if self.transformation is None:
            self.transformation = self._tf_type(**self._tf_init_params)
        return MstepResult(self.transformation, sigma2, q)

    def reset_transform(self):
        if self.transformation is not None:
            self.transformation.reset()

    def maximization_step(
        self,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
        source_colors: Optional[torch.Tensor] = None,
        target_colors: Optional[torch.Tensor] = None,
    ) -> MstepResult:
        """
        Perform the maximization step of the CPD algorithm.

        This method updates the transformation parameters by maximizing the
        likelihood function given the current correspondences between the
        source and target points.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data. Shape: (n_points, n_dims).
        estep_res : EstepResult
            Result of the expectation step, containing the correspondences
            between the source and target points.
        sigma2_p : float, optional
            Previous variance. Default: None.

        Returns
        -------
        MstepResult
            Result of the maximization step, containing the updated
            transformation, updated variance, and updated objective
            function value.
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
        sigma2_p: Optional[float] = None,
        update_scale: bool = True,
        target_colors: Optional[torch.Tensor] = None,
        source_colors: Optional[torch.Tensor] = None,
    ) -> MstepResult:
        """
        Static method for the maximization step.

        This method performs the actual computation for the maximization step.
        It is defined as a static method to allow for easier testing and
        reuse.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud data. Shape: (n_points, n_dims).
        target : torch.Tensor
            Target point cloud data. Shape: (n_points, n_dims).
        estep_res : EstepResult
            Result of the expectation step.
        sigma2_p : float, optional
            Previous variance. Default: None.
        update_scale : bool, optional
            If True, update the scale parameter. Default: True.

        Returns
        -------
        MstepResult
            Result of the maximization step.
        """
        pt1, p1, px, n_p = estep_res
        dim = source.shape[1]
        if source_colors is not None:
            source = torch.cat([source, source_colors], dim=1)
            target = torch.cat([target, target_colors], dim=1)
        # Calculate means of source and target points
        mu_x = torch.sum(px, axis=0) / n_p
        mu_y = (source.T @ p1.unsqueeze(1)).squeeze() / n_p  # .dot

        # Center the point clouds
        target_hat = target - mu_x
        source_hat = source - mu_y

        # Compute the cross-covariance matrix
        a = torch.matmul(px.T, source_hat) - torch.outer(mu_x, (p1.unsqueeze(0) @ source_hat).squeeze())
        # .dot / .outer
        # Compute the optimal rotation using SVD (TODO: does keeping this true make a difference?)
        u, _, vh = torch.linalg.svd(a, full_matrices=False)
        c = torch.ones(dim, dtype=a.dtype, device=a.device)
        c[-1] = torch.linalg.det(torch.matmul(u, vh))  # .dot
        rot = torch.matmul(u * c, vh)  # .dot

        # Compute the optimal scale (if enabled)
        tr_atr = torch.trace(torch.matmul(a.T, rot))  # .dot
        tr_yp1y = torch.trace(torch.matmul(source_hat.T * p1, source_hat))  # .dot
        scale = tr_atr / tr_yp1y if update_scale else 1.0

        # Compute the optimal translation
        t = mu_x - scale * torch.matmul(rot, mu_y)  # .dot
        tr_xp1x = torch.trace(torch.matmul(target_hat.T * pt1, target_hat))  # .dot

        # Update the variance
        if update_scale:
            sigma2 = (tr_xp1x - scale * tr_atr) / (n_p * dim)
        else:
            sigma2 = (tr_xp1x + tr_yp1y - scale * tr_atr) / (n_p * dim)
        sigma2 = max(sigma2, torch.finfo(a.dtype).eps)  # Ensure sigma2 is not too small

        # Update the objective function value
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale**2) * tr_yp1y) / (
            2.0 * sigma2 + dim * n_p * 0.5 * torch.log(sigma2)
        )
        # q += dim * n_p * 0.5 * torch.log(sigma2) #.item()
        return MstepResult(tf.RigidTransformation(rot, t, scale), sigma2, q)


class AffineCPD(CoherentPointDrift):
    """
    Coherent Point Drift for affine transformation.

    This class implements the Coherent Point Drift (CPD) algorithm for aligning
    two point clouds using an affine transformation. It inherits from the
    `CoherentPointDrift` base class and specializes it for affine transformations.

    Parameters
    ----------
    source : torch.Tensor, optional
        Source point cloud data.
    tf_init_params : dict, optional
        Parameters to initialize the affine transformation.
    use_color : bool, optional
        Use color information (if available) for registration.
    use_cuda : bool, optional
        Use CUDA for accelerated computation.

    Attributes
    ----------
    _tf_type : type
        Type of transformation object to use (affine in this case).
    _tf_init_params : dict
        Parameters to initialize the transformation.
    """

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        tf_init_params: Dict = {},
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq=100,
    ) -> None:
        super(AffineCPD, self).__init__(source, use_color, use_cuda, log_freq=log_freq)
        self._tf_type = tf.AffineTransformation
        self._tf_init_params = tf_init_params

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """
        Initialize the registration parameters.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data.

        Returns
        -------
        MstepResult
            Result of the maximization step, containing the initial
            transformation, sigma2, and q.
        """
        dim = self._N_DIM
        sigma2 = squared_kernel_sum(self._source[:, :dim], target[:, :dim])
        # Initialize q (negative log-likelihood)
        q = 1.0 + target.shape[0] * dim * 0.5 * torch.log(sigma2)
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
    ) -> MstepResult:
        """
        Perform the maximization step of the CPD algorithm.

        This step updates the transformation parameters (affine in this case)
        by maximizing the expectation of the complete data log-likelihood.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud data.
        target : torch.Tensor
            Target point cloud data.
        estep_res : EstepResult
            Result of the expectation step.
        sigma2_p : float, optional
            Previous value of sigma2.

        Returns
        -------
        MstepResult
            Result of the maximization step, containing the updated
            transformation, sigma2, and q.
        """
        pt1, p1, px, n_p = estep_res
        dim = CoherentPointDrift._N_DIM

        # get means
        mu_x = torch.sum(px, dim=0) / n_p
        mu_y = torch.matmul(source.T, p1) / n_p  # .dot

        # center point clouds
        target_hat = target - mu_x
        source_hat = source - mu_y

        # compute affine transformations parameters
        a = torch.matmul(px.T, source_hat) - torch.outer(mu_x, torch.dot(p1.T, source_hat))  # .dot  # .dot
        yp1y = torch.matmul(source_hat.T * p1, source_hat)  # .dot
        b = torch.linalg.solve(yp1y.T, a.T).T  # solve for rotation and scaling matrix
        t = mu_x - torch.matmul(b, mu_y)  # .dot  - solver for translation vector

        # update sigma2
        tr_xp1x = torch.trace(torch.matmul(target_hat.T * pt1, target_hat))  # .dot
        tr_xpyb = torch.trace(torch.matmul(a, b.T))  # .dot
        sigma2 = (tr_xp1x - tr_xpyb) / (n_p * dim)
        sigma2 = max(sigma2, torch.finfo(a.dtype).eps)

        # update q (negative log-likelihood)
        tr_ab = torch.trace(torch.matmul(a, b.T))  # .dot
        q = (tr_xp1x - 2 * tr_ab + tr_xpyb) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * torch.log(sigma2)

        return MstepResult(tf.AffineTransformation(b, t), sigma2, q)


class NonRigidCPD(CoherentPointDrift):
    """
    Coherent Point Drift for non-rigid transformation.

    This class implements the Coherent Point Drift (CPD) algorithm for
    non-rigid registration of two point clouds. It uses a Gaussian
    radial basis function (RBF) kernel to model the non-rigid deformation.

    Parameters
    ----------
    source : torch.Tensor, optional
        Source point cloud data. Shape: (n_points, n_dimensions),
        where n_dimensions is typically 2 or 3.
    beta : float, optional
        Parameter of the RBF kernel. Controls the width of the Gaussian kernel.
        Default: 2.0
    lmd : float, optional
        Regularization parameter. Controls the smoothness of the deformation.
        Default: 2.0
    use_color : bool, optional
        Use color information (if available) in the registration process.
        Default: False
    use_cuda : bool, optional
        Use CUDA for GPU acceleration. Default: False

    Attributes
    ----------
    _tf_type : type
        Type of transformation object to use. Set to `NonRigidTransformation`.
    _beta : float
        Parameter of the RBF kernel.
    _lmd : float
        Regularization parameter.
    _tf_obj : NonRigidTransformation
        Instance of the `NonRigidTransformation` class.

    """

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        beta: float = 2.0,
        lmd: float = 2.0,
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq=100,
    ) -> None:
        super(NonRigidCPD, self).__init__(source, use_color, use_cuda, log_freq=log_freq)
        self._tf_type = tf.NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self._tf_obj = None
        if self._source is not None:
            self._tf_obj = self._tf_type(None, self._source, self._beta)

    def set_source(self, source: torch.Tensor) -> None:
        self._source = source
        self._tf_obj = self._tf_type(None, self._source, self._beta)

    def maximization_step(
        self, target: torch.Tensor, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> MstepResult:
        """
        Perform the maximization step of the EM algorithm.

        This step updates the transformation parameters based on the current
        correspondences between the source and target point clouds.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data. Shape: (n_points, n_dimensions)
        estep_res : EstepResult
            Result of the expectation step.
        sigma2_p : float, optional
            Previous variance.

        Returns
        -------
        MstepResult
            Result of the maximization step.
        """
        return self._maximization_step(
            self._source[:, : self._N_DIM],
            target[:, : self._N_DIM],
            estep_res,
            sigma2_p,
            self._tf_obj,
            self._lmd,
        )

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        dim = self._N_DIM
        sigma2 = squared_kernel_sum(self._source[:, :dim], target[:, :dim])
        q = 1.0 + target.shape[0] * dim * 0.5 * torch.log(sigma2)
        self._tf_obj.w = torch.zeros_like(self._source)
        return MstepResult(self._tf_obj, sigma2, q)

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float,
        tf_obj: tf.NonRigidTransformation,
        lmd: float,
    ) -> MstepResult:
        """
        Helper function for the maximization step.

        This function performs the actual computation for updating the
        transformation parameters.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud data.
        target : torch.Tensor
            Target point cloud data.
        estep_res : EstepResult
            Result of the expectation step.
        sigma2_p : float
            Previous variance.
        tf_obj : NonRigidTransformation
            Transformation object.
        lmd : float
            Regularization parameter.

        Returns
        -------
        MstepResult
            Result of the maximization step.
        """
        pt1, p1, px, n_p = estep_res
        dim = CoherentPointDrift._N_DIM

        # Solve for the deformation parameters (w)
        w = torch.linalg.solve(
            (p1 * tf_obj.g).T + lmd * sigma2_p * torch.eye(source.shape[0], dtype=source.dtype, device=source.device),
            px - (source.T * p1).T,
        )

        # Update the transformed source points (t)
        t = source + torch.matmul(tf_obj.g, w)  # .dot
        tr_xp1x = torch.trace(torch.matmul(target.T * pt1, target))  # .dot
        tr_pxt = torch.trace(torch.matmul(px.T, t))  # .dot
        tr_tpt = torch.trace(torch.matmul(t.T * p1, t))  # .dot
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)


class ConstrainedNonRigidCPD(CoherentPointDrift):
    """
    Extended Coherent Point Drift for nonrigid transformation with constraints.

    This class extends the Coherent Point Drift (CPD) algorithm to handle
    nonrigid transformations with the addition of point correspondence constraints.
    It allows for incorporating prior knowledge about the correspondence between
    specific points in the source and target point clouds.

    See: https://people.mpi-inf.mpg.de/~golyanik/04_DRAFTS/ECPD2016.pdf

    Parameters
    ----------
    source : torch.Tensor, optional
        Source point cloud data.
    beta : float, optional
        Parameter of RBF kernel. Default: 2.0.
    lmd : float, optional
        Parameter for regularization term. Default: 2.0.
    alpha : float, optional
        Degree of reliability of priors. Approximately between 1e-8
        (highly reliable) and 1 (highly unreliable). Default: 1e-8.
    use_cuda : bool, optional
        Use CUDA. Default: False.
    use_color : bool, optional
        Use color information (if available). Default: False.
    idx_source : torch.Tensor of ints, optional
        Indices in source matrix for which a correspondence is known.
    idx_target : torch.Tensor of ints, optional
        Indices in target matrix for which a correspondence is known.
    """

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        beta: float = 2.0,
        lmd: float = 2.0,
        alpha: float = 1e-8,
        use_color: bool = False,
        use_cuda: bool = False,
        idx_source: Optional[torch.Tensor] = None,
        idx_target: Optional[torch.Tensor] = None,
        log_freq=100,
    ):
        super(ConstrainedNonRigidCPD, self).__init__(source, use_color, use_cuda, log_freq=log_freq)
        self._tf_type = tf.NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self.alpha = alpha
        self._tf_obj = None
        self.idx_source, self.idx_target = idx_source, idx_target
        if self._source is not None:
            self._tf_obj = self._tf_type(None, self._source, self._beta)

    def set_source(self, source: torch.Tensor) -> None:
        self._source = source
        self._tf_obj = self._tf_type(None, self._source, self._beta)

    def maximization_step(
        self, target: torch.Tensor, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> MstepResult:
        """
        Perform the maximization step of the EM algorithm.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data.
        estep_res : EstepResult
            Result of the expectation step.
        sigma2_p : float, optional
            Initial variance.

        Returns
        -------
        MstepResult
            Result of the maximization step.
        """
        return self._maximization_step(
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

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        dim = self._N_DIM
        sigma2 = squared_kernel_sum(self._source[:, :dim], target[:, :dim])
        q = 1.0 + target.shape[0] * dim * 0.5 * torch.log(sigma2)
        self._tf_obj.w = torch.zeros_like(self._source)
        self.p_tilde = torch.zeros((self._source.shape[0], target.shape[0]), dtype=target.dtype, device=target.device)
        if self.idx_source is not None and self.idx_target is not None:
            self.p_tilde[self.idx_source, self.idx_target] = 1
        self.p1_tilde = torch.sum(self.p_tilde, dim=1)
        self.px_tilde = torch.matmul(self.p_tilde, target)  # .dot
        return MstepResult(self._tf_obj, sigma2, q)

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float,
        tf_obj: tf.NonRigidTransformation,
        lmd: float,
        alpha: float,
        p1_tilde: float,
        px_tilde: float,
    ) -> MstepResult:
        """
        Perform the maximization step of the EM algorithm (static method).

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud data.
        target : torch.Tensor
            Target point cloud data.
        estep_res : EstepResult
            Result of the expectation step.
        sigma2_p : float
            Initial variance.
        tf_obj : tf.NonRigidTransformation
            Non-rigid transformation object.
        lmd : float
            Regularization parameter.
        alpha : float
            Degree of reliability of priors.
        p1_tilde : float
            Sum of probabilities for constrained correspondences.
        px_tilde : float
            Weighted sum of target points for constrained correspondences.

        Returns
        -------
        MstepResult
            Result of the maximization step.
        """
        pt1, p1, px, n_p = estep_res
        dim = CoherentPointDrift._N_DIM

        # Solve for the transformation parameters (w)
        w = torch.linalg.solve(
            (p1 * tf_obj.g).T
            + sigma2_p / alpha * (p1_tilde * tf_obj.g).T
            + lmd * sigma2_p * torch.eye(source.shape[0], dtype=source.dtype, device=source.device),
            px - (source.T * p1).T + sigma2_p / alpha * (px_tilde - (source.T * p1_tilde).T),
        )
        t = source + torch.matmul(tf_obj.g, w)  # .dot
        tr_xp1x = torch.trace(torch.matmul(target.T * pt1, target))  # .dot
        tr_pxt = torch.trace(torch.matmul(px.T, t))  # .dot
        tr_tpt = torch.trace(torch.matmul(t.T * p1, t))  # .dot
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)


def cpd_registration(
    source: Union[torch.Tensor, o3d.t.geometry.PointCloud],
    target: Union[torch.Tensor, o3d.t.geometry.PointCloud],
    tf_type_name: str = "rigid",
    w: float = 0.0,
    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: List[Callable] = [],
    use_color: bool = False,
    log_freq: bool = 100,
    **kwargs: Any,
) -> MstepResult:
    """
    CPD Registration function.

    This function performs point cloud registration using the Coherent Point Drift
    (CPD) algorithm. It supports different transformation types and allows for
    custom callbacks during the registration process.

    Parameters
    ----------
    source : torch.Tensor or o3d.t.geometry.PointCloud
        Source point cloud data.
    target : torch.Tensor or o3d.t.geometry.PointCloud
        Target point cloud data.
    tf_type_name : str, optional
        Transformation type ('rigid', 'affine', 'nonrigid', 'nonrigid_constrained').
        Default: 'rigid'.
    w : float, optional
        Weight of the uniform distribution, 0 < `w` < 1. Default: 0.0.
    maxiter : int, optional
        Maximum number of iterations for the EM algorithm. Default: 50.
    tol : float, optional
        Tolerance for termination. Default: 0.001.
    callbacks : list of callable, optional
        Called after each iteration. `callback(probreg.Transformation)`.
        Default: [].
    use_color : bool, optional
        Use color information (if available). Default: False.

    Keyword Args
    ------------
    update_scale : bool, optional
        If True and `tf_type` is 'rigid', then the scale is treated.
        Default: True.
    tf_init_params : dict, optional
        Parameters to initialize transformation (for 'rigid' or 'affine').

    Returns
    -------
    MstepResult
        Result of the registration (transformation, sigma2, q).
    """
    # Convert from Open3D to torch.Tensor if necessary
    if isinstance(source, o3d.t.geometry.PointCloud):
        source = dataset.open3d_to_torch(source)
    if isinstance(target, o3d.t.geometry.PointCloud):
        target = dataset.open3d_to_torch(target)

    # Concatenate color information if use_color is True
    if use_color:
        sourcei = torch.cat([source["pos"], source["color"]], dim=1)
        targeti = torch.cat([target["pos"], target["color"]], dim=1)
    else:
        sourcei = source["pos"]
        targeti = target["pos"]

    # Instantiate the appropriate CPD object based on tf_type_name
    if tf_type_name == "rigid":
        cpd = RigidCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    elif tf_type_name == "affine":
        cpd = AffineCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    elif tf_type_name == "nonrigid":
        cpd = NonRigidCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    elif tf_type_name == "nonrigid_constrained":
        cpd = ConstrainedNonRigidCPD(sourcei, use_color=use_color, log_freq=log_freq, **kwargs)
    else:
        raise ValueError("Unknown transformation type %s" % tf_type_name)

    cpd.set_callbacks(callbacks)
    return cpd.registration(targeti, w, maxiter, tol)


def init_cpd_from_existing(
    transform,
    source: Union[torch.Tensor, o3d.t.geometry.PointCloud],
    target: Union[torch.Tensor, o3d.t.geometry.PointCloud],
    w: float = 0.0,
    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: List[Callable] = [],
    use_color: bool = False,
    log_freq: int = 100,
):
    # Convert from Open3D to torch.Tensor if necessary
    if isinstance(source, o3d.t.geometry.PointCloud):
        source = dataset.open3d_to_torch(source)
    if isinstance(target, o3d.t.geometry.PointCloud):
        target = dataset.open3d_to_torch(target)

    # Concatenate color information if use_color is True
    if use_color:
        sourcei = torch.cat([source["pos"], source["color"]], dim=1)
        # targeti = torch.cat([target["pos"], target["color"]], dim=1)
    else:
        sourcei = source["pos"]
        # targeti = target["pos"]
    if isinstance(transform, tf.RigidTransformation):
        # rigid case
        cpdobj = RigidCPD(
            sourcei,
            use_color=use_color,
            rot=transform.rot,
            t=transform.t,
            scale=transform.scale,
            device=transform.rot.device,
            dtype=transform.rot.dtype,
            log_freq=log_freq,
        )
    elif isinstance(transform, tf.AffineTransformation):
        # rigid case
        cpdobj = AffineCPD(
            sourcei,
            use_color=use_color,
            b=transform.b,
            t=transform.t,
            device=transform.rot.device,
            dtype=transform.rot.dtype,
            log_freq=log_freq,
        )
    else:
        raise TypeError(f"transform type not known/not implemented: {type(transform)}")
    return cpdobj
