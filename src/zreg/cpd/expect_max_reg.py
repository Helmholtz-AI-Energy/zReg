import torch
import logging


log = logging.getLogger(__name__)


class ExpMaxRegistration(object):
    """
    Expectation maximization point cloud registration.

    Parameters
    ----------
    target : torch.Tensor  (X)
        NxD array of target points.
    source : torch.Tensor  (Y)
        MxD array of source points. (TODO: these will be registered to the targets?)
    sigma2 : float, optional
        Initial variance of the Gaussian mixture model.
    max_iterations : int, optional
        Maximum number of iterations.
    tolerance : float, optional
        Tolerance for convergence.
    w : float, optional
        Contribution of the uniform distribution to account for outliers.


    Attributes
    ----------
    transformed_source : torch.Tensor
        MxD array of transformed source points.
    sigma2 : float
        Variance of the Gaussian mixture model.
    num_targ_pts : int  (N)
        Number of target points.
    num_src_pts : int  (M)
        Number of source points.
    dimensionality : int (D)
        Dimensionality of source and target points.
    iteration : int
        Current iteration.
    max_iterations : int
        Maximum number of iterations.
    tolerance : float
        Tolerance for convergence.
    w : float
        Contribution of the uniform distribution.
    q : float
        Objective function value.
    diff : float
        Difference between consecutive objective function values.
    probs : torch.Tensor
        MxN array of probabilities.
    sum_probs_target : torch.Tensor
        Nx1 array, sum of probabilities for each target point.
    sum_probs_source : torch.Tensor
        Mx1 array, sum of probabilities for each source point.
    sum_probs : float
        Sum of all probabilities.
    probs_targets : torch.Tensor
        MxD array, probs @ targets
    """

    def __init__(
        self,
        target,
        source,
        sigma2=None,
        max_iterations=None,
        tolerance=None,
        w=None,
        *args,
        **kwargs,
    ):
        """
        Initialize the ExpMaxRegistration class.
        """
        if target.ndim != 2:
            raise ValueError("The target point cloud (target) must be a 2D array.")

        if source.ndim != 2:
            raise ValueError("The source point cloud (source) must be a 2D array.")

        if target.shape[1] != source.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions."
            )

        if sigma2 is not None and sigma2 <= 0:
            raise ValueError(
                f"Expected a positive value for sigma2 instead got: {sigma2}"
            )

        if max_iterations is not None and max_iterations < 0:
            raise ValueError(
                f"Expected a positive integer for max_iterations instead got: {max_iterations}"
            )
        elif not isinstance(max_iterations, int):
            max_iterations = int(max_iterations)

        if tolerance is not None and tolerance < 0:
            raise ValueError(
                f"Expected a positive float for tolerance instead got: {tolerance}"
            )

        if w is not None and (w < 0 or w >= 1):
            raise ValueError(
                f"Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {w}"
            )

        self.target = target
        self.fact = {"dtype": target.dtype, "device": target.device}
        self.source = source
        self.transformed_source = source.clone()
        self.sigma2 = (
            self.initialize_sigma2(target, source) if sigma2 is None else sigma2
        )
        (self.num_targ_pts, self.dimensionality) = self.target.shape
        self.num_src_pts = self.source.shape[0]
        self.tolerance = 1e-3 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = float("inf")
        self.q = float("inf")
        self.probs = torch.zeros((self.num_src_pts, self.num_targ_pts), **self.fact)
        self.sum_probs_target = torch.zeros((self.num_targ_pts,), **self.fact)
        self.sum_probs_source = torch.zeros((self.num_src_pts,), **self.fact)
        self.probs_targets = torch.zeros(
            (self.num_src_pts, self.dimensionality), **self.fact
        )
        self.sum_probs = 0

    def initialize_sigma2(self, target, source):
        """
        Initialize the variance (sigma2).

        Attributes
        ----------
        target: numpy array
            NxD array of points for target.

        source: numpy array
            MxD array of points for source.

        Returns
        -------
        sigma2: float
            Initial variance.
        """
        (num_targ_pts, dimensionality) = target.shape
        (num_src_pts, _) = source.shape
        diff = target[None, :, :] - source[:, None, :]
        err = diff**2
        return torch.sum(err) / (dimensionality * num_src_pts * num_targ_pts)

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.

        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.

        Returns
        -------
        self.transformed_source: numpy array
            MxD array of transformed source points.

        registration_parameters:
            Returned params dependent on registration method used.
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {
                    "iteration": self.iteration,
                    "error": self.q,
                    "target": self.target,
                    "source": self.transformed_source,
                }
                callback(**kwargs)
            print(f"diff = {self.diff}")

        return self.transformed_source, self.get_registration_parameters()

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes."
        )

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes."
        )

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes."
        )

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes."
        )

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        probs = torch.sum(
            (self.target[None, :, :] - self.transformed_source[:, None, :]) ** 2, axis=2
        )  # (num_src_pts, num_targ_pts)
        probs = torch.exp(-probs / (2 * self.sigma2))
        c = (
            (2 * torch.pi * self.sigma2) ** (self.dimensionality / 2)
            * self.w
            / (1.0 - self.w)
            * self.num_src_pts
            / self.num_targ_pts
        )

        den = torch.sum(probs, axis=0, keepdims=True)  # (1, num_targ_pts)
        den = torch.clip(den, torch.finfo(self.target.dtype).eps, None) + c

        self.probs = torch.divide(probs, den)
        self.sum_probs_target = torch.sum(self.probs, axis=0)
        self.sum_probs_source = torch.sum(self.probs, axis=1)
        self.sum_probs = torch.sum(self.sum_probs_source)
        self.probs_targets = torch.matmul(self.probs, self.target)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()
