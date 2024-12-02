# This file takes insperation from https://github.com/neka-nat/probreg/
# The core algorithms are the same, but the implementation now makes use of pytorch

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union
import torch

import numpy as np
import open3d as o3d
import time
# import six
# from scipy.spatial import distance as scipy_distance

# from . import math_utils as mu
from . import transforms as tf
from . import dataset
from .utils import squared_kernel_sum
import logging

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, TimeElapsedColumn

log = logging.getLogger(__name__)


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
    This is an abstract class.
    Based on this class, it is inherited by rigid, affine, nonrigid classes
    according to the type of transformation.
    In this class, Estimation step in EM algorithm is implemented and
    Maximazation step is implemented in the inherited classes.

    Args:
        source (torch.Tensor, optional): Source point cloud data.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    _N_DIM = 3
    _N_COLOR = 3

    def __init__(self, source: Optional[torch.Tensor] = None, use_color: bool = False, use_cuda: bool = False) -> None:
        self._source = source
        self._tf_type = None
        self._callbacks = []
        self._use_color = use_color
        # self.xp = np
        # self.distance_module = scipy_distance

    def set_source(self, source: torch.Tensor) -> None:
        self._source = source

    def set_callbacks(self, callbacks: List[Callable]) -> None:
        self._callbacks.extend(callbacks)

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        return MstepResult(None, None, None)

    def _compute_pmat_numerator(self, t_source: torch.Tensor, target: torch.Tensor, sigma2: float) -> torch.Tensor:
        pmat = torch.cdist(t_source, target, p=2).pow(2)  # "sqeuclidean")
        pmat = torch.exp(-pmat / (2.0 * sigma2))
        return pmat

    def expectation_step(
        self,
        t_source: torch.Tensor,
        target: torch.Tensor,
        sigma2: float,
        sigma2_c: float,
        w: float = 0.0,
    ) -> EstepResult:
        """Expectation step for CPD"""
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        pmat = self._compute_pmat_numerator(t_source[:, : self._N_DIM], target[:, : self._N_DIM], sigma2)

        c = (2.0 * torch.pi * sigma2) ** (self._N_DIM * 0.5)
        c *= w / (1.0 - w) * t_source.shape[0] / target.shape[0]
        den = torch.sum(pmat, dim=0)
        den[den == 0] = torch.finfo(target.dtype).eps
        if self._use_color:
            pmat_c = self._compute_pmat_numerator(t_source[:, self._N_DIM :], target[:, self._N_DIM :], sigma2_c)
            den_c = torch.sum(pmat_c, dim=0)
            den_c[den_c == 0] = torch.finfo(pmat_c.dtype).eps
            den = torch.multiply(den, den_c)
            o_c = t_source.shape[0] * (2 * torch.pi * sigma2_c) ** (0.5 * (self._N_DIM + self._N_COLOR - 1))
            # print(o_c.shape, pmat_c.shape)
            o_c = o_c * torch.exp(-1.0 / t_source.shape[0] * torch.square(torch.sum(pmat_c, dim=0)) / (2.0 * sigma2_c))
            den += o_c
            c *= (2.0 * torch.pi * sigma2_c) ** (self._N_COLOR * 0.5)
            pmat = torch.multiply(pmat, pmat_c)
        den += c

        pmat = torch.divide(pmat, den)
        pt1 = torch.sum(pmat, dim=0)
        p1 = torch.sum(pmat, dim=1)
        px = torch.matmul(pmat, target[:, : self._N_DIM])  # .dot
        return EstepResult(pt1, p1, px, torch.sum(p1))

    def maximization_step(
        self, target: torch.Tensor, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> Optional[MstepResult]:
        return self._maximization_step(self._source[:, : self._N_DIM], target[:, : self._N_DIM], estep_res, sigma2_p)

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
    ) -> Optional[MstepResult]:
        return None

    def registration(self, target: torch.Tensor, w: float = 0.0, maxiter: int = 50, tol: float = 0.001) -> MstepResult:
        assert self._tf_type is not None, "transformation type is None."
        res = self._initialize(target[:, : self._N_DIM])
        sigma2_c = 0.0
        if self._use_color:
            sigma2_c = squared_kernel_sum(self._source[:, self._N_DIM :], target[:, self._N_DIM :])
        q = res.q

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TextColumn("[progress.percentage]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("[progress.percentage]{task.fields[iter_time]}"),
            TimeRemainingColumn(),
            TextColumn("[progress.percentage]{task.fields[criteria]}"),
        ) as progress:
            task = progress.add_task("[cyan]Registering...", total=maxiter, criteria="", iter_time="")
            start_time = time.perf_counter()
            for i in range(maxiter):
                iter_start_time = time.perf_counter()
                t_source = res.transformation.transform(self._source)
                estep_res = self.expectation_step(t_source, target, res.sigma2, sigma2_c, w)
                res = self.maximization_step(target, estep_res, res.sigma2)
                for c in self._callbacks:
                    c(res.transformation)

                iter_end_time = time.perf_counter()
                iter_time = iter_end_time - iter_start_time

                elapsed_time = time.perf_counter() - start_time
                avg_iter_time = elapsed_time / (i + 1)

                progress.update(
                    task,
                    advance=1,
                    criteria=f"Criteria: {res.q:.4f}",
                    iter_time=f"Avg: {avg_iter_time:.2f}s, Iter: {iter_time:.2f}s",
                )
                log.debug(f"Registering: iteration {i}/{maxiter}, criteria: {res.q:.4f}")

                if abs(res.q - q) < tol:
                    log.info(f"Hit tolerance in iteration {i} (criteria: {res.q:.4f}), exiting")
                    break
                q = res.q

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

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        update_scale: bool = True,
        tf_init_params: Dict = {},
        use_color: bool = False,
        use_cuda: bool = False,
    ) -> None:
        super(RigidCPD, self).__init__(source, use_color, use_cuda)
        self._tf_type = tf.RigidTransformation
        self._update_scale = update_scale
        self._tf_init_params = tf_init_params

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        dim = self._N_DIM
        sigma2 = squared_kernel_sum(self._source[:, :dim], target[:, :dim])
        q = 1.0 + target.shape[0] * dim * 0.5 * torch.log(sigma2)
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    def maximization_step(
        self, target: torch.Tensor, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> MstepResult:
        return self._maximization_step(
            self._source[:, : self._N_DIM], target[:, : self._N_DIM], estep_res, sigma2_p, self._update_scale
        )

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
        update_scale: bool = True,
    ) -> MstepResult:
        pt1, p1, px, n_p = estep_res
        dim = CoherentPointDrift._N_DIM
        mu_x = torch.sum(px, axis=0) / n_p
        mu_y = (source.T @ p1.unsqueeze(1)).squeeze() / n_p  # .dot
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = torch.matmul(px.T, source_hat) - torch.outer(
            mu_x, (p1.unsqueeze(0) @ source_hat).squeeze()
        )  # .dot / .outer
        u, _, vh = torch.linalg.svd(a, full_matrices=True)
        c = torch.ones(dim, dtype=a.dtype, device=a.device)
        c[-1] = torch.linalg.det(torch.matmul(u, vh))  # .dot
        rot = torch.matmul(u * c, vh)  # .dot
        tr_atr = torch.trace(torch.matmul(a.T, rot))  # .dot
        tr_yp1y = torch.trace(torch.matmul(source_hat.T * p1, source_hat))  # .dot
        scale = tr_atr / tr_yp1y if update_scale else 1.0
        t = mu_x - scale * torch.matmul(rot, mu_y)  # .dot
        tr_xp1x = torch.trace(torch.matmul(target_hat.T * pt1, target_hat))  # .dot
        if update_scale:
            sigma2 = (tr_xp1x - scale * tr_atr) / (n_p * dim)
        else:
            sigma2 = (tr_xp1x + tr_yp1y - scale * tr_atr) / (n_p * dim)
        sigma2 = max(sigma2, torch.finfo(a.dtype).eps)
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale**2) * tr_yp1y) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * np.log(sigma2).item()
        return MstepResult(tf.RigidTransformation(rot, t, scale), sigma2, q)


class AffineCPD(CoherentPointDrift):
    """Coherent Point Drift for affine transformation.

    Args:
        source (torch.Tensor, optional): Source point cloud data.
        tf_init_params (dict, optional): Parameters to initialize transformation.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        tf_init_params: Dict = {},
        use_color: bool = False,
        use_cuda: bool = False,
    ) -> None:
        super(AffineCPD, self).__init__(source, use_color, use_cuda)
        self._tf_type = tf.AffineTransformation
        self._tf_init_params = tf_init_params

    def _initialize(self, target: torch.Tensor) -> MstepResult:
        dim = self._N_DIM
        sigma2 = squared_kernel_sum(self._source[:, :dim], target[:, :dim])
        q = 1.0 + target.shape[0] * dim * 0.5 * torch.log(sigma2)
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    @staticmethod
    def _maximization_step(
        source: torch.Tensor,
        target: torch.Tensor,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
    ) -> MstepResult:
        pt1, p1, px, n_p = estep_res
        dim = CoherentPointDrift._N_DIM
        mu_x = torch.sum(px, dim=0) / n_p
        mu_y = torch.matmul(source.T, p1) / n_p  # .dot
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = torch.matmul(px.T, source_hat) - torch.outer(mu_x, torch.dot(p1.T, source_hat))  # .dot  # .dot
        yp1y = torch.matmul(source_hat.T * p1, source_hat)  # .dot
        b = torch.linalg.solve(yp1y.T, a.T).T
        t = mu_x - torch.matmul(b, mu_y)  # .dot
        tr_xp1x = torch.trace(torch.matmul(target_hat.T * pt1, target_hat))  # .dot
        tr_xpyb = torch.trace(torch.matmul(a, b.T))  # .dot
        sigma2 = (tr_xp1x - tr_xpyb) / (n_p * dim)
        tr_ab = torch.trace(torch.matmul(a, b.T))  # .dot
        sigma2 = max(sigma2, torch.finfo(a.dtype).eps)
        q = (tr_xp1x - 2 * tr_ab + tr_xpyb) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * torch.log(sigma2)
        return MstepResult(tf.AffineTransformation(b, t), sigma2, q)


class NonRigidCPD(CoherentPointDrift):
    """Coherent Point Drift for nonrigid transformation.

    Args:
        source (torch.Tensor, optional): Source point cloud data.
        beta (float, optional): Parameter of RBF kernel.
        lmd (float, optional): Parameter for regularization term.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(
        self,
        source: Optional[torch.Tensor] = None,
        beta: float = 2.0,
        lmd: float = 2.0,
        use_color: bool = False,
        use_cuda: bool = False,
    ) -> None:
        super(NonRigidCPD, self).__init__(source, use_color, use_cuda)
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
        pt1, p1, px, n_p = estep_res
        dim = CoherentPointDrift._N_DIM
        w = torch.linalg.solve(
            (p1 * tf_obj.g).T + lmd * sigma2_p * torch.eye(source.shape[0], dtype=source.dtype, device=source.device),
            px - (source.T * p1).T,
        )
        t = source + torch.matmul(tf_obj.g, w)  # .dot
        tr_xp1x = torch.trace(torch.matmul(target.T * pt1, target))  # .dot
        tr_pxt = torch.trace(torch.matmul(px.T, t))  # .dot
        tr_tpt = torch.trace(torch.matmul(t.T * p1, t))  # .dot
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)


class ConstrainedNonRigidCPD(CoherentPointDrift):
    """
       Extended Coherent Point Drift for nonrigid transformation.
       Like CoherentPointDrift, but allows to add point correspondance constraints
       See: https://people.mpi-inf.mpg.de/~golyanik/04_DRAFTS/ECPD2016.pdf

    Args:
        source (torch.Tensor, optional): Source point cloud data.
        beta (float, optional): Parameter of RBF kernel.
        lmd (float, optional): Parameter for regularization term.
        alpha (float): Degree of reliability of priors.
            Approximately between 1e-8 (highly reliable) and 1 (highly unreliable)
        use_cuda (bool, optional): Use CUDA.
        use_color (bool, optional): Use color information (if available).
        idx_source (torch.Tensor of ints, optional): Indices in source matrix
            for which a correspondance is known
        idx_target (torch.Tensor of ints, optional): Indices in target matrix
            for which a correspondance is known
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
    ):
        super(ConstrainedNonRigidCPD, self).__init__(source, use_color, use_cuda)
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
        pt1, p1, px, n_p = estep_res
        dim = CoherentPointDrift._N_DIM
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


def registration_cpd(
    source: Union[torch.Tensor, o3d.t.geometry.PointCloud],
    target: Union[torch.Tensor, o3d.t.geometry.PointCloud],
    tf_type_name: str = "rigid",
    w: float = 0.0,
    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: List[Callable] = [],
    use_color: bool = False,
    **kwargs: Any,
) -> MstepResult:
    """CPD Registraion.

    Args:
        source (torch.Tensor): Source point cloud data.
        target (torch.Tensor): Target point cloud data.
        tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid', 'nonrigid_constrained')
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`
        use_color (bool, optional): Use color information (if available).

    Keyword Args:
        update_scale (bool, optional): If this flag is true and tf_type is rigid transformation,
            then the scale is treated. The default is true.
        tf_init_params (dict, optional): Parameters to initialize transformation (for rigid or affine).

    Returns:
        MstepResult: Result of the registration (transformation, sigma2, q)
    """
    # convert from o3d to dict structure
    if isinstance(source, o3d.t.geometry.PointCloud):
        source = dataset.open3d_to_torch(source)
    if isinstance(target, o3d.t.geometry.PointCloud):
        target = dataset.open3d_to_torch(target)

    if use_color:
        sourcei = torch.cat([source["pos"], source["color"]], dim=1)
        targeti = torch.cat([target["pos"], target["color"]], dim=1)
    else:
        sourcei = source["pos"]
        targeti = target["pos"]
    if tf_type_name == "rigid":
        cpd = RigidCPD(sourcei, use_color=use_color, **kwargs)
    elif tf_type_name == "affine":
        cpd = AffineCPD(sourcei, use_color=use_color, **kwargs)
    elif tf_type_name == "nonrigid":
        cpd = NonRigidCPD(sourcei, use_color=use_color, **kwargs)
    elif tf_type_name == "nonrigid_constrained":
        cpd = ConstrainedNonRigidCPD(sourcei, use_color=use_color, **kwargs)
    else:
        raise ValueError("Unknown transformation type %s" % tf_type_name)
    cpd.set_callbacks(callbacks)
    return cpd.registration(targeti, w, maxiter, tol)
