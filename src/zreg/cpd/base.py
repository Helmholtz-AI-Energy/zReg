"""Abstract base class for Coherent Point Drift algorithm."""

from abc import ABC, abstractmethod
from typing import Callable
import logging

import torch

from ._types import EstepResult, MstepResult
from ..validation import _validate_tensors
from ..utils import squared_kernel_sum

__all__ = ["CoherentPointDrift"]

log = logging.getLogger(__name__)

# Enable high precision for matrix operations
torch.set_float32_matmul_precision("high")


class CoherentPointDrift(ABC):
    """Abstract base class for Coherent Point Drift algorithm.

    This class provides a common framework for different types of transformations
    (rigid, affine, nonrigid). Subclasses must implement the initialization and
    maximization step methods.

    Parameters
    ----------
    source : torch.Tensor | None
        Source point cloud data with shape (N, D).
    source_colors : torch.Tensor | None
        Color information for source points with shape (N, C).
    use_color : bool
        Use color information in registration if True.
    use_cuda : bool
        Use CUDA for computations if True.
    log_freq : int
        Log frequency during registration. Set to -1 to disable logging.

    Attributes
    ----------
    _N_DIM : int
        Number of spatial dimensions (default: 3).
    _N_COLOR : int
        Number of color channels (default: 3).
    """

    _N_DIM = 3
    _N_COLOR = 3

    def __init__(
        self,
        source: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
        use_color: bool = False,
        use_cuda: bool = False,
        log_freq: int = True,
    ) -> None:
        """Initialize CPD object."""
        self._source = source
        self._source_colors = None
        self._tf_type = None  # Transformation type (set in subclasses)
        self._callbacks: list[Callable] = []
        self._use_color = use_color
        if use_color:
            self._source_colors = source_colors
        self.transformation = None
        self.log_freq = log_freq

    def set_source(
        self, source: torch.Tensor, source_colors: torch.Tensor | None = None
    ) -> None:
        """Set or update the source point cloud.

        Parameters
        ----------
        source : torch.Tensor
            Source point cloud data.
        source_colors : torch.Tensor | None
            Color information for source points.
        """
        _validate_tensors(source, names=["source"])
        if source_colors is not None:
            _validate_tensors(source, source_colors, names=["source", "source_colors"])
        self._source = source
        if self._use_color and source_colors is not None:
            self._source_colors = source_colors

    def set_callbacks(self, callbacks: list[Callable]) -> None:
        """Add callbacks to be called after each iteration.

        Parameters
        ----------
        callbacks : list[Callable]
            List of callback functions accepting a transformation object.
        """
        self._callbacks.extend(callbacks)

    @abstractmethod
    def _initialize(self, target: torch.Tensor) -> MstepResult:
        """Initialize parameters for the registration process.

        This method must be implemented by subclasses to set up
        transformation-specific initial parameters.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud data.

        Returns
        -------
        MstepResult
            Initial parameters including transformation and sigma2.
        """
        ...

    def _compute_pmat_numerator(
        self, t_source: torch.Tensor, target: torch.Tensor, sigma2: float
    ) -> torch.Tensor:
        """Compute the numerator of the probability matrix.

        Parameters
        ----------
        t_source : torch.Tensor
            Transformed source point cloud.
        target : torch.Tensor
            Target point cloud.
        sigma2 : float
            Variance of the Gaussian kernel.

        Returns
        -------
        torch.Tensor
            Numerator of the probability matrix.
        """
        pmat = torch.cdist(t_source, target, p=2).pow(2)
        pmat = torch.exp(-pmat / (2.0 * sigma2))
        return pmat

    def expectation_step(
        self,
        t_source: torch.Tensor,
        target: torch.Tensor,
        sigma2: float,
        sigma2_c: float,
        w: float = 0.0,
        target_colors: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
    ) -> EstepResult:
        """Perform the Expectation step of the EM algorithm.

        Calculates posterior probabilities of correspondence between
        transformed source points and target points.

        Parameters
        ----------
        t_source : torch.Tensor
            Transformed source point cloud.
        target : torch.Tensor
            Target point cloud.
        sigma2 : float
            Variance of Gaussian kernel for coordinates.
        sigma2_c : float
            Variance of Gaussian kernel for color.
        w : float
            Weight of uniform distribution for outlier handling.
        target_colors : torch.Tensor | None
            Target color information.
        source_colors : torch.Tensor | None
            Source color information.

        Returns
        -------
        EstepResult
            Posterior probabilities and intermediate results.
        """
        posdims = t_source.shape[1]
        assert t_source.ndim == 2 and target.ndim == 2, (
            "source and target must have 2 dimensions."
        )
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

            o_c = t_source.shape[0] * (2 * torch.pi * sigma2_c) ** (
                0.5 * (posdims + ncolors - 1)
            )
            o_c = o_c * torch.exp(
                -1.0
                / t_source.shape[0]
                * torch.square(torch.sum(pmat_c, dim=0))
                / (2.0 * sigma2_c)
            )
            den += o_c
            c *= (2.0 * torch.pi * sigma2_c) ** (ncolors * 0.5)
            pmat = torch.multiply(pmat, pmat_c)

        den += c
        pmat = torch.divide(pmat, den)

        pt1 = torch.sum(pmat, dim=0)
        p1 = torch.sum(pmat, dim=1)
        px = torch.matmul(pmat, target)
        return EstepResult(pt1, p1, px, torch.sum(p1), pmat)

    def maximization_step(
        self,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float | None = None,
        target_colors: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Perform the Maximization step of the EM algorithm.

        This default implementation calls _maximization_step and updates
        the transformation. Subclasses may override for custom behavior.

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
        ret = self._maximization_step(
            self._source,
            target,
            estep_res,
            sigma2_p,
            target_colors=target_colors,
            source_colors=source_colors,
        )
        self.transformation = ret.transformation
        return ret

    @staticmethod
    @abstractmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: float | None = None,
        target_colors: torch.Tensor | None = None,
        source_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Internal maximization step implementation.

        Must be implemented by subclasses for specific transformation types.

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
            Target color information.
        source_colors : torch.Tensor | None
            Source color information.

        Returns
        -------
        MstepResult
            Updated transformation parameters.
        """
        ...

    def registration(
        self,
        target: torch.Tensor,
        w: float = 0.0,
        maxiter: int = 50,
        tol: float = 0.001,
        target_colors: torch.Tensor | None = None,
    ) -> MstepResult:
        """Perform the CPD registration process.

        Iteratively executes E-step and M-step until convergence or
        maximum iterations reached.

        Parameters
        ----------
        target : torch.Tensor
            Target point cloud.
        w : float
            Weight of uniform distribution for outlier handling.
        maxiter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        target_colors : torch.Tensor | None
            Target color information.

        Returns
        -------
        MstepResult
            Final transformation and convergence diagnostics.
        """
        assert self._tf_type is not None, "transformation type is None."
        if self._source is not None:
            _validate_tensors(self._source, target, names=["source", "target"])
        else:
            _validate_tensors(target, names=["target"])
        res = self._initialize(target)
        sigma2_c = 0.0
        if self._use_color:
            sigma2_c = squared_kernel_sum(self._source_colors, target_colors)

        sigma2_history: list[float] = []
        sigma2_clamped = False
        eps = torch.finfo(target.dtype).eps
        n_iters = 0

        running_avg = torch.arange(4, dtype=target.dtype, device=target.device)
        for i in range(maxiter):
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
                target,
                estep_res,
                res.sigma2,
                target_colors=target_colors,
                source_colors=self._source_colors,
            )

            # Clamp sigma2 to safe lower bound
            clamped_sigma2 = torch.clamp(res.sigma2, min=eps)
            if clamped_sigma2 != res.sigma2 and not sigma2_clamped:
                log.warning(
                    "CPD: sigma2 clamped to dtype.eps during registration"
                    " - numerical instability possible."
                )
                sigma2_clamped = True
            res = MstepResult(
                transformation=res.transformation,
                sigma2=clamped_sigma2,
                q=res.q,
            )
            sigma2_history.append(
                res.sigma2.item()
                if isinstance(res.sigma2, torch.Tensor)
                else float(res.sigma2)
            )

            for c in self._callbacks:
                c(res.transformation)

            if self.log_freq > 0 and i % self.log_freq == self.log_freq - 1:
                log.info(f"Registering: iteration {i}/{maxiter}, criteria: {res.q:.4f}")

            running_avg[i % running_avg.shape[0]] = res.q

            if running_avg.abs().diff().mean().abs() < tol:
                if self.log_freq > 0:
                    log.info(
                        f"Hit tolerance in iteration {i} (criteria: {res.q:.4f}), exiting."
                    )
                break

        n_iters = (i + 1) if maxiter > 0 else 0
        if self.log_freq > 0 and maxiter > 0:
            log.info(f"End registration at step {i} (criteria: {res.q:.5f})")

        return MstepResult(
            transformation=res.transformation,
            sigma2=res.sigma2,
            q=res.q,
            n_iters=n_iters,
            sigma2_history=sigma2_history,
        )
