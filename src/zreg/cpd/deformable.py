import torch

from .expect_max_reg import ExpMaxRegistration


class DeformableRegistration(ExpMaxRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    low_rank: bool
        Whether to use low rank approximation.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation.

    constrained: bool
        Whether to use constrained registration.

    e_alpha: float (positive)
        Reliability of correspondence priors for constrained registration. Between 1e-8 (very reliable) and 1 (very unreliable).

    source_id: numpy.ndarray (int)
        Indices for the points to be used as correspondences in the source array for constrained registration.

    target_id: numpy.ndarray (int)
        Indices for the points to be used as correspondences in the target array for constrained registration.
    """

    def __init__(
        self,
        alpha=None,
        beta=None,
        low_rank=False,
        num_eig=100,
        constrained=False,
        e_alpha=None,
        source_id=None,
        target_id=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if alpha is not None and alpha <= 0:
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(
                    alpha
                )
            )

        if beta is not None and beta <= 0:
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(
                    beta
                )
            )

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.W = torch.zeros((self.num_src_pts, self.dimensionality), **self.fact)
        self.gauss = gaussian_kernel(self.source, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.gauss, self.num_eig)
            self.inv_S = torch.diag(1.0 / self.S)
            self.S = torch.diag(self.S)
            self.E = 0.0

        self.constrained = constrained
        if self.constrained:
            if e_alpha is not None and e_alpha <= 0:
                raise ValueError(
                    "Expected a positive value for regularization parameter e_alpha. Instead got: {}".format(
                        e_alpha
                    )
                )

            if source_id.ndim != 1:
                raise ValueError(
                    "The source ids (source_id) must be a 1D array of ints."
                )

            if target_id.ndim != 1:
                raise ValueError(
                    "The target ids (target_id) must be a 1D array of ints."
                )

            self.e_alpha = 1e-8 if e_alpha is None else e_alpha
            self.source_id = source_id
            self.target_id = target_id
            self.P_tilde = torch.zeros(
                (self.num_src_pts, self.num_targ_pts), **self.fact
            )
            self.P_tilde[self.source_id, self.target_id] = 1
            self.sum_probs_source_tilde = torch.sum(self.P_tilde, dim=1)
            self.PX_tilde = torch.dot(self.P_tilde, self.target)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        if not self.low_rank:
            A = (
                self.sum_probs_source.diag() @ self.gauss
            ) + self.alpha * self.sigma2 * torch.eye(self.num_src_pts, self.fact)  # dot
            B = self.probs_targets - (self.sum_probs_source.diag() @ self.source)  # dot

            if self.constrained:
                A += self.sigma2 * (1 / self.e_alpha)
                A *= torch.dot(
                    torch.diag(self.sum_probs_source_tilde), self.gauss
                )  # dot

                B += self.sigma2 * (1 / self.e_alpha)
                B *= self.PX_tilde - torch.dot(
                    torch.diag(self.sum_probs_source_tilde), self.source
                )  # dot

            self.W = torch.linalg.solve(A, B)

        else:
            dP = torch.diag(self.sum_probs_source)
            if self.constrained:
                dP += self.sigma2 * (1 / self.e_alpha)
                dP *= torch.diag(self.sum_probs_source_tilde)

            dPQ = torch.matmul(dP, self.Q)
            F = self.probs_targets - torch.matmul(dP, self.source)
            # in original code, disagreement between two methods: alternative is probs_targets - np.dot(np.diag(self.sum_probs_source), self.source)
            if self.constrained:
                F += self.sigma2 * (1 / self.e_alpha)
                F *= self.PX_tilde - torch.dot(
                    self.sum_probs_source_tilde.diag(), self.source
                )  # dot

            self.W = 1 / (self.alpha * self.sigma2)
            hold = torch.linalg.solve(
                self.alpha * self.sigma2 * self.inv_S + (self.Q.T @ dPQ),
                self.Q.T @ F,
            )
            self.W *= F - (dPQ @ hold)

            QtW = torch.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * torch.trace(QtW.T @ (self.S @ QtW))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.source used.

        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.


        """
        if Y is not None:
            G = gaussian_kernel(target=Y, beta=self.beta, Y=self.source)
            return Y + torch.dot(G, self.W)  # dot
        else:
            if not self.low_rank:
                self.transformed_source = self.source + torch.dot(self.gauss, self.W)
            else:
                self.transformed_source = self.source + (
                    self.Q @ (self.S @ (self.Q.T @ self.W))
                )

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        # self.q = torch.inf

        xPx = torch.dot(
            self.sum_probs_target.T,
            torch.sum(torch.multiply(self.target, self.target), axis=1),
        )
        yPy = torch.dot(
            self.sum_probs_source.T,
            torch.sum(
                torch.multiply(self.transformed_source, self.transformed_source), axis=1
            ),
        )
        trPXY = torch.sum(torch.multiply(self.transformed_source, self.probs_targets))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.sum_probs * self.dimensionality)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = torch.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.gauss: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.gauss, self.W


def gaussian_kernel(target, beta, Y=None):
    if Y is None:
        Y = target
    diff = target[:, None, :] - Y[None, :, :]
    diff = torch.square(diff)
    diff = torch.sum(diff, 2)
    return torch.exp(-diff / (2 * beta**2))


def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = torch.linalg.eigh(G)
    eig_indices = list(torch.argsort(torch.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S
