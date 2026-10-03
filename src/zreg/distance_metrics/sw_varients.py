# much of these functions stem from: https://github.com/VinAIResearch/PointSWD/
# minor changes made in most functions for usability and for the target use-case

import torch
import torch.nn as nn
from pathlib import Path
import os
import tempfile

from ..utils.validation import _validate_tensors

__all__ = [
    "SlicedWassersteinDistance",
    "MaxSlicedWassersteinDistance",
    "ProjectedWassersteinDistance",
    "AdaptiveSlicedWassersteinDistance",
    "OrthogonalSlicedWassersteinDistance",
    "GeneralisedSlicedWassersteinDistance",
]


def minibatch_rand_projections(batchsize, dim, num_projections=1000, **kwargs):
    projections = torch.randn((batchsize, num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections**2, dim=2, keepdim=True))
    return projections


def proj_onto_unit_sphere(vectors):
    """
    input: vectors: [batchsize, num_projs, dim]
    """
    norm = torch.sqrt(torch.sum(vectors**2, dim=2, keepdim=True))
    # Clamp to avoid division by zero for zero-length vectors
    norm = torch.clamp(norm, min=torch.finfo(vectors.dtype).eps)
    return vectors / norm


def _sample_minibatch_orthogonal_projections(batch_size, dim, num_projections):
    projections = torch.zeros((batch_size, num_projections, dim))
    projections = torch.stack(
        [torch.nn.init.orthogonal_(projections[i]) for i in range(projections.shape[0])],
        dim=0,
    )
    return projections


def _coupling_indices(n, m, device):
    """Index pairs and weights of the 1-D quantile coupling of two empirical measures.

    The optimal (monotone) coupling between the uniform empirical measures on
    ``n`` and ``m`` sorted atoms transports mass ``w[k]`` from the ``ix[k]``-th
    smallest atom of the first measure to the ``iy[k]``-th smallest atom of the
    second. The breakpoints of the two quantile functions are the multiples
    ``i * m`` and ``j * n`` on the common grid ``[0, n * m]``; consecutive
    breakpoints delimit at most ``n + m - 1`` segments. For ``n == m`` the
    coupling is the identity pairing with weights ``1 / n``.

    Parameters
    ----------
    n, m : int
        Number of atoms of the two measures.
    device : str or torch.device
        Device of the returned tensors.

    Returns
    -------
    ix, iy : torch.LongTensor
        Non-decreasing indices into the sorted atoms, shape ``[K]``.
    w : torch.Tensor
        float64 weights summing to 1, shape ``[K]``; cast to the data dtype at use.

    Raises
    ------
    ValueError
        If ``n <= 0`` or ``m <= 0`` (empty point set).
    """
    if n <= 0 or m <= 0:
        raise ValueError(f"Cannot couple an empty point set: got n={n}, m={m}")
    b = torch.unique(
        torch.cat(
            [
                torch.arange(1, n + 1, device=device) * m,
                torch.arange(1, m + 1, device=device) * n,
            ]
        )
    )  # sorted breakpoints in (0, n * m]
    w = torch.diff(b, prepend=b.new_zeros(1)).to(torch.float64) / (n * m)
    ix = (b + m - 1) // m - 1
    iy = (b + n - 1) // n - 1
    return ix, iy, w


def _sorted_pow_sum(xproj, yproj, degree):
    """Per-projection sum of ``|sort(x) - sort(y)|**degree`` for the sort-sum SW variants.

    Parameters
    ----------
    xproj : torch.Tensor
        Projected source data, shape ``[B, L, N]``.
    yproj : torch.Tensor
        Projected target data, shape ``[B, L, M]``.
    degree : float
        Exponent ``p``.

    Returns
    -------
    torch.Tensor
        Shape ``[B, L]``.

    Notes
    -----
    For ``N == M`` this is exactly the legacy expression (sum over the paired
    sorted points; bit-identical). For ``N != M`` the sorted projections are
    paired by the 1-D quantile coupling (:func:`_coupling_indices`) and the
    weighted coupling cost is multiplied by ``max(N, M)`` so magnitudes stay
    comparable to the equal-N "sum over points" convention (legacy max(N, M)
    cardinality scaling). The coupling itself is exact, but the scaled value
    depends on sample multiplicity by design: duplicating every point of one
    cloud doubles it. It is not the normalised empirical Wasserstein distance.
    """
    n, m = xproj.shape[2], yproj.shape[2]
    if n == m:
        return torch.sum(torch.pow(torch.abs(torch.sort(xproj)[0] - torch.sort(yproj)[0]), degree), dim=2)
    ix, iy, w = _coupling_indices(n, m, device=xproj.device)
    diff = torch.sort(xproj)[0][:, :, ix] - torch.sort(yproj)[0][:, :, iy]
    return max(n, m) * torch.sum(torch.pow(torch.abs(diff), degree) * w.to(diff.dtype), dim=2)


def compute_practical_moments_sw(x, y, num_projections=30, degree=2.0, **kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    if x.dtype != y.dtype:
        raise RuntimeError(f"Different dtypes btw x/y : {x.dtype}/{y.dtype}")
    dim = x.size(2)
    if y.size(2) != dim:
        raise ValueError(
            f"Expected y to have dimension {dim} to match x and projections, "
            f"but got y.size(2)={y.size(2)}"
        )
    batch_size = x.size(0)
    projections = minibatch_rand_projections(batch_size, dim, num_projections)
    projections = projections.to(dtype=x.dtype, device=x.device)
    # projs.shape: [batchsize, num_projs, dim]

    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort_pow_p_get_sum = _sorted_pow_sum(xproj.transpose(1, 2), yproj.transpose(1, 2), degree)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def compute_practical_moments_sw_with_predefined_projections(x, y, projections, degree=2.0, **kwargs):
    """
    x, y: [batch size, num points, dim]
    projections: [batch size, num projs, dim]
    """
    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort_pow_p_get_sum = _sorted_pow_sum(xproj.transpose(1, 2), yproj.transpose(1, 2), degree)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def _compute_practical_moments_sw_with_projected_data(xproj, yproj, degree=2.0, **kwargs):
    _sort_pow_p_get_sum = _sorted_pow_sum(xproj.transpose(1, 2), yproj.transpose(1, 2), degree)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def _circular(x, theta):
    """The circular defining function for generalized Radon transform
    Inputs
    X:  [batch size, num_points, d] - d: dim of 1 point
    theta: [batch size, L, d] that parameterizes for L projections
    """
    x_s = torch.stack([x for _ in range(theta.shape[1])], dim=2)
    theta_s = torch.stack([theta for _ in range(x.shape[1])], dim=1)
    z_s = x_s - theta_s
    return torch.sqrt(torch.sum(z_s**2, dim=3))


def _linear(x, theta):
    """
    x: [batch size, num_points, d] - d: dim of 1 point
    theta: [batch size, L, d] that parameterizes for L projections
    """
    xproj = x.bmm(theta.transpose(1, 2))
    return xproj


class BaseWD(nn.Module):
    """Base class for Sliced Wasserstein Distance variants.

    This class handles batch dimension logic uniformly across all SWD variants.
    Subclasses implement `_forward()` with the specific distance computation.

    Batch Dimension Handling
    ------------------------
    - If input has 2 dimensions (n_points, dim), a batch dimension is added
    - If `nobatchdim=True`, inputs are always unsqueezed to add batch dim
    - After computation, the batch dimension is squeezed back if it was added

    All SWD variant classes (SlicedWassersteinDistance, MaxSlicedWassersteinDistance,
    etc.) inherit from this base to share the batch handling logic.

    Parameters
    ----------
    nobatchdim : bool
        If True, always treat inputs as unbatched (add batch dim).
    device : str or torch.device
        Device to use for computation.

    See Also
    --------
    SlicedWassersteinDistance : Fixed number of random projections
    AdaptiveSlicedWassersteinDistance : Adaptive number of projections
    MaxSlicedWassersteinDistance : Optimized single projection
    OrthogonalSlicedWassersteinDistance : Orthogonal projections
    GeneralisedSlicedWassersteinDistance : Circular/linear defining functions
    ProjectedWassersteinDistance : Point-wise projected distance
    """

    def __init__(self, nobatchdim, device):
        super().__init__()
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.nobatchdim = nobatchdim

    def forward(self, x, y, *args, **kwargs):
        """Compute the distance between point clouds ``x`` and ``y``.

        Parameters
        ----------
        x : torch.Tensor
            Source points, shape ``[N, D]`` or ``[B, N, D]``.
        y : torch.Tensor
            Target points, shape ``[M, D]`` or ``[B, M, D]``. ``M`` may differ
            from ``N`` (1-D quantile coupling; see :func:`_sorted_pow_sum`).

        Returns
        -------
        torch.Tensor
            Scalar distance.

        Raises
        ------
        ValueError
            If ``x`` or ``y`` contains non-finite values, has fewer than two
            dimensions, or is an empty point set (``N == 0`` or ``M == 0``).
        """
        _validate_tensors(x, y, names=["x", "y"])

        # Validate minimum dimensions before unsqueeze to avoid silent shape errors
        if x.ndim < 2 or y.ndim < 2:
            raise ValueError(
                f"Expected x and y to have at least 2 dimensions (n_points, dim), "
                f"but got x.ndim={x.ndim}, y.ndim={y.ndim}"
            )
        if x.shape[-2] == 0 or y.shape[-2] == 0:
            raise ValueError(
                f"Sliced Wasserstein distances need non-empty point sets, but got an empty x or y: "
                f"x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
            )

        xsqueeze = False
        if x.ndim < 3 or self.nobatchdim:
            x = x.unsqueeze(0)
            xsqueeze = True
        ysqueeze = False
        if y.ndim < 3 or self.nobatchdim:
            y = y.unsqueeze(0)
            ysqueeze = True

        loss = self._forward(x, y, *args, **kwargs)
        if xsqueeze:
            x = x.squeeze()
        if ysqueeze:
            y = y.squeeze()
        return loss

    def _forward(self, x, y, *args, **kwargs): ...


class SlicedWassersteinDistance(BaseWD):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device=None, nobatchdim=True, **kwargs):
        super().__init__(nobatchdim=nobatchdim, device=device)
        self.num_projs = num_projs
        self.nobatchdim = nobatchdim

    def _forward(self, x, y, *args, **kwargs):
        """
        x: [batch_size, N, dim], y: [batch_size, M, dim]; N and M may differ (1-D quantile
        coupling under the legacy max(N, M) cardinality scaling, see _sorted_pow_sum)
        """
        squared_sw_2, _ = compute_practical_moments_sw(x, y, num_projections=self.num_projs)
        squared_sw_2 = squared_sw_2.mean(dim=0)
        return squared_sw_2


class AdaptiveSlicedWassersteinDistance(BaseWD):
    """
    Adaptive sliced wasserstein algorithm for estimating SWD
    """

    def __init__(
        self,
        init_projs=20,
        step_projs=10,
        k=2.0,
        loop_rate_thresh=0.05,
        projs_history=None,
        max_slices=500,
        nobatchdim=True,
        epsilon=0.5,
        degree=2.0,
        device=None,
        **kwargs,
    ):
        super().__init__(nobatchdim=nobatchdim, device=device)
        self.init_projs = init_projs
        self.step_projs = step_projs
        self.k = k
        self.loop_rate_thresh = loop_rate_thresh
        self.max_slices = max_slices
        self.epsilon = epsilon
        self.degree = degree
        # Append MPI rank to filename to prevent race conditions when multiple
        # processes write/delete the same file during parallel execution.
        if projs_history is not None:
            rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
            if rank != 0:
                base, ext = os.path.splitext(projs_history)
                projs_history = f"{base}_rank{rank}{ext}"
        self.projs_history = projs_history

    def _forward(self, x, y, *args, **kwargs):
        """
        x, y: [batch size, num points in point cloud, 3]
        """
        # allow to adjust epsilon
        eps = self.epsilon if "epsilon" not in kwargs else kwargs["epsilon"]
        degree = self.degree if "degree" not in kwargs else kwargs["degree"]

        n = self.init_projs
        max_slices = self.max_slices
        step_projs = self.step_projs

        first_moment_sw_p_pow_p, second_moment_sw_p_pow_p = compute_practical_moments_sw(
            x, y, num_projections=n, degree=degree
        )

        # check ASW condition
        loop_conditions = (self.k**2 * (second_moment_sw_p_pow_p - first_moment_sw_p_pow_p**2)) > ((n - 1) * eps**2)
        # the ratio of point clouds in the batch satifying the ASW condition.
        loop_rate = loop_conditions.sum(dim=0) * 1.0 / loop_conditions.shape[0]

        while (loop_rate > self.loop_rate_thresh) and ((n + step_projs) <= max_slices):
            # sample next s projections
            first_moment_s_sw, second_moment_s_sw = compute_practical_moments_sw(
                x,
                y,
                num_projections=step_projs,
                degree=degree,
            )
            # update first and second moments
            first_moment_sw_p_pow_p = (n * first_moment_sw_p_pow_p + step_projs * first_moment_s_sw) / (n + step_projs)
            second_moment_sw_p_pow_p = (n * second_moment_sw_p_pow_p + step_projs * second_moment_s_sw) / (
                n + step_projs
            )
            n = n + step_projs
            loop_conditions = (self.k**2 * (second_moment_sw_p_pow_p - first_moment_sw_p_pow_p**2)) > ((n - 1) * eps**2)
            loop_rate = loop_conditions.sum(dim=0) * 1.0 / loop_conditions.shape[0]

        if self.projs_history is not None:
            with open(self.projs_history, "a") as fp:
                fp.write(str(n) + "\n")
        else:
            # Auto-cleanup: use tempfile that deletes on close
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=True) as fp:
                fp.write(str(n) + "\n")
        return first_moment_sw_p_pow_p.mean(dim=0)

    def remove_history(self):
        if self.projs_history is None:
            return
        file = Path(self.projs_history)
        if file.exists():
            # remove the proj history...need to do this after every distance
            try:
                os.remove(self.projs_history)
            except FileNotFoundError:
                # preventing race condition when running in parallel
                pass


class MaxSlicedWassersteinDistance(BaseWD):
    """Max-sliced Wasserstein distance.

    Max-SW distance was proposed in paper "Max-Sliced Wasserstein Distance and its use for GANs" - CVPR'19
    The way to estimate it was proposed in paper "Generalized Sliced Wasserstein Distance" - NeurIPS'19

    A single projection direction is optimised with Adam to maximise the
    sliced distance; the distance is then evaluated along the optimised
    direction. The inner maximisation runs on detached copies of the inputs,
    so it never back-propagates into the caller's graph; only the final
    evaluation is differentiable with respect to ``x`` and ``y``.

    Parameters
    ----------
    device : str or torch.device, optional
        Device to use for computation.
    nobatchdim : bool, default True
        If True, always treat inputs as unbatched (add batch dim).
    max_sw_num_iters : int, default 50
        Number of Adam steps of the inner projection maximisation. A
        ``max_sw_num_iters`` keyword passed to ``forward`` overrides it.
    max_sw_lr : float, default 1e-4
        Adam learning rate of the inner projection maximisation. A
        ``max_sw_lr`` keyword passed to ``forward`` overrides it.
    """

    def __init__(self, device=None, nobatchdim=True, max_sw_num_iters=50, max_sw_lr=1e-4, **kwargs):
        super().__init__(nobatchdim=nobatchdim, device=device)
        self.max_sw_num_iters = max_sw_num_iters
        self.max_sw_lr = max_sw_lr

    def _forward(self, x, y, *args, **kwargs):
        """
        x: [batch_size, N, dim], y: [batch_size, M, dim]; N and M may differ (1-D quantile
        coupling under the legacy max(N, M) cardinality scaling, see _sorted_pow_sum)
        """
        dim = x.size(2)
        # generate on CPU (unchanged RNG stream), then follow the input dtype/device
        projections = minibatch_rand_projections(batchsize=x.size(0), dim=dim, num_projections=1)
        projections = projections.to(dtype=x.dtype, device=x.device).requires_grad_(True)
        # projs.shape: [batchsize, num_projs, dim]

        num_iter = kwargs.get("max_sw_num_iters", self.max_sw_num_iters)
        lr = kwargs.get("max_sw_lr", self.max_sw_lr)
        optimizer = torch.optim.Adam([projections], lr=lr)

        # the inner maximisation must not reach the caller's graph (U3-new-1)
        xd, yd = x.detach(), y.detach()
        for i in range(num_iter):
            # compute loss
            xproj = xd.bmm(projections.transpose(1, 2))

            yproj = yd.bmm(projections.transpose(1, 2))

            _sort_pow_2_get_sum = _sorted_pow_sum(xproj.transpose(1, 2), yproj.transpose(1, 2), 2.0)

            negative_first_moment = -(_sort_pow_2_get_sum.mean(dim=1))

            # perform optimization
            optimizer.zero_grad()
            hold = negative_first_moment.mean()
            hold.backward()
            optimizer.step()
            # project onto unit sphere (in-place to preserve Adam's parameter reference)
            projections.data = proj_onto_unit_sphere(projections.data)

        projections_no_grad = projections.detach()
        loss, _ = compute_practical_moments_sw_with_predefined_projections(x, y, projections_no_grad)

        return loss.mean(dim=0)


class OrthogonalSlicedWassersteinDistance(BaseWD):
    """
    Orthogonal estimation for SWD was proposed in paper "Orthogonal estimation of Wasserstein Distance - AISTATS'19"
    """

    def __init__(self, num_projs, device=None, nobatchdim=True, **kwargs):
        super().__init__(nobatchdim=nobatchdim, device=device)
        self.num_projs = num_projs

    def _forward(self, x, y, *args, **kwargs):
        """
        x: [batch_size, N, dim], y: [batch_size, M, dim]; N and M may differ (1-D quantile
        coupling under the legacy max(N, M) cardinality scaling, see _sorted_pow_sum)
        """
        dim = x.shape[2]
        if self.num_projs > dim:
            raise ValueError(
                f"OrthogonalSlicedWassersteinDistance requires num_projs <= dim, "
                f"but got num_projs={self.num_projs}, dim={dim}"
            )

        projections = torch.zeros(
            (x.shape[0], self.num_projs, x.shape[2]),
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )

        projections = torch.stack(
            [torch.nn.init.orthogonal_(projections[i]) for i in range(projections.shape[0])],
            dim=0,
        )

        loss, _ = compute_practical_moments_sw_with_predefined_projections(x, y, projections)

        return loss.mean(dim=0)


class GeneralisedSlicedWassersteinDistance(BaseWD):
    """
    Generalized SW distance was proposed in paper "Generalized Sliced Wasserstein Distance" - NeurIPS'19
    """

    def __init__(self, num_projs, degree=2.0, g_type="circular", device=None, nobatchdim=True, **kwargs):
        super().__init__(nobatchdim=nobatchdim, device=device)
        self.num_projs = num_projs
        self.g_type = g_type
        self.degree = degree

    def _forward(self, x, y, *args, **kwargs):
        """
        x: [batch_size, N, dim], y: [batch_size, M, dim]; N and M may differ (1-D quantile
        coupling under the legacy max(N, M) cardinality scaling, see _sorted_pow_sum)
        """
        dim = x.size(2)
        batch_size = x.size(0)
        projections = minibatch_rand_projections(batch_size, dim, self.num_projs)
        projections = projections.to(dtype=x.dtype, device=x.device)

        if self.g_type == "circular":
            xproj = _circular(x, projections)
            yproj = _circular(y, projections)
        elif self.g_type == "linear":
            xproj = _linear(x, projections)
            yproj = _linear(y, projections)
        else:
            raise NotImplementedError

        deg = self.degree if "degree" not in kwargs else kwargs["degree"]

        loss, _ = _compute_practical_moments_sw_with_projected_data(xproj, yproj, deg)

        return loss.mean(dim=0)


class ProjectedWassersteinDistance(BaseWD):
    """
    Projected Wasserstein distance was proposed in paper "Orthogonal estimation of Wasserstein Distance - AISTATS'19"

    Points are paired by their order along each random projection and the
    squared differences of the paired points are averaged (a mean, not a sum).
    For unequal point counts (N != M) the points are paired by the 1-D
    quantile coupling (:func:`_coupling_indices`) and the result is the
    weighted analogue of that mean (weights sum to 1). Deliberately there is
    NO max(N, M) factor here, unlike the sort-sum variants: N == M stays
    bit-identical and continuous, and the value is invariant to duplicating
    every point of a cloud.
    """

    def __init__(self, num_projs, device=None, orthogonal=False, nobatchdim=True, **kwargs):
        super().__init__(nobatchdim=nobatchdim, device=device)
        self.num_projs = num_projs
        self.orthogonal = orthogonal

    def _forward(self, x, y, *args, **kwargs):
        """
        x: [batch_size, N, dim], y: [batch_size, M, dim]; N and M may differ (1-D quantile
        coupling as a weighted mean, no max(N, M) factor; see the class docstring)
        """

        dim = x.size(2)
        batch_size = x.size(0)
        if self.orthogonal:
            projections = _sample_minibatch_orthogonal_projections(batch_size, dim, self.num_projs)
            projections = projections.to(dtype=x.dtype, device=x.device)
        else:
            projections = minibatch_rand_projections(batch_size, dim, self.num_projs)
            projections = projections.to(dtype=x.dtype, device=x.device)
        # print(projections)
        xproj = _linear(x, projections).transpose(1, 2)  # [bs, num_slices, num_points]
        yproj = _linear(y, projections).transpose(1, 2)  # [bs, num_slices, num_points]

        xproj_argsort = torch.argsort(xproj, dim=2)
        yproj_argsort = torch.argsort(yproj, dim=2)

        _sorted_x = torch.stack([x[i][xproj_argsort[i]] for i in range(x.shape[0])], dim=0)
        _sorted_y = torch.stack([y[i][yproj_argsort[i]] for i in range(y.shape[0])], dim=0)

        n, m = x.shape[1], y.shape[1]
        if n == m:
            loss = torch.mean((_sorted_x - _sorted_y) ** 2)
            return loss
        # unequal N: quantile coupling as a weighted mean over points (weights
        # sum to 1); deliberately no max(N, M) factor -- see the class docstring
        ix, iy, w = _coupling_indices(n, m, device=x.device)
        sx = _sorted_x[:, :, ix]  # [B, L, K, dim]
        sy = _sorted_y[:, :, iy]
        loss = (((sx - sy) ** 2) * w.to(x.dtype)[None, None, :, None]).sum(dim=2).mean()
        return loss
