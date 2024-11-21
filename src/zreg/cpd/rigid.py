import torch

from .expect_max_reg import ExpMaxRegistration


class RigidRegistration(ExpMaxRegistration):
    """
    Rigid registration.

    Attributes
    ----------
    R: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.

    t: numpy array
        1xD initial translation vector.

    s: float (positive)
        scaling parameter.

    A: numpy array
        Utility array used to calculate the rotation matrix.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    """

    # Additional parameters used in this class, but not inputs.
    # YPY: float
    #     Denominator value used to update the scale factor.
    #     Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    # target_hat: numpy array
    #     Centered target point cloud.
    #     Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    def __init__(self, R=None, t=None, s=None, scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fact = {"dtype": self.target.dtype, "device": self.target.device}
        if self.dimensionality != 2 and self.dimensionality != 3:
            raise ValueError(
                "Rigid registration only supports 2D or 3D point clouds. Instead got {}.".format(
                    self.dimensionality
                )
            )

        if R is not None and (
            (R.ndim != 2)
            or (R.shape[0] != self.dimensionality)
            or (R.shape[1] != self.dimensionality)
            or not is_positive_semi_definite(R)
        ):
            raise ValueError(
                "The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.".format(
                    self.dimensionality, self.dimensionality, R
                )
            )

        if t is not None and (
            (t.ndim != 2) or (t.shape[0] != 1) or (t.shape[1] != self.dimensionality)
        ):
            raise ValueError(
                "The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.".format(
                    self.dimensionality, t
                )
            )

        if s is not None and (not isinstance(s, numbers.Number) or s <= 0):
            raise ValueError(
                "The scale factor must be a positive number. Instead got: {}.".format(s)
            )

        self.R = torch.eye(self.dimensionality, **fact) if R is None else R
        self.t = (
            torch.atleast_2d(torch.zeros((1, self.dimensionality)), **fact)
            if t is None
            else t
        )
        self.s = 1 if s is None else s
        self.scale = scale

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # target point cloud mean
        mu_target = torch.divide(torch.sum(self.probs_targets, axis=0), self.sum_probs)
        # source point cloud mean
        muY = torch.divide(
            torch.sum(np.dot(torch.transpose(self.P), self.source), axis=0),
            self.sum_probs,
        )

        self.target_hat = self.target - torch.tile(mu_target, (self.num_targ_pts, 1))
        # centered source point cloud
        Y_hat = self.source - torch.tile(muY, (self.num_src_pts, 1))
        self.YPY = np.dot(
            torch.transpose(self.sum_probs_source),
            torch.sum(torch.multiply(Y_hat, Y_hat), axis=1),
        )

        self.A = np.dot(torch.transpose(self.target_hat), torch.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = torch.linalg.svd(self.A, full_matrices=True)
        C = torch.ones((self.dimensionality,))
        C[self.dimensionality - 1] = torch.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = torch.transpose(np.dot(np.dot(U, torch.diag(C)), V))
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.scale is True:
            self.s = (
                torch.trace(np.dot(torch.transpose(self.A), torch.transpose(self.R)))
                / self.YPY
            )
        else:
            pass
        self.t = torch.transpose(mu_target) - self.s * np.dot(
            torch.transpose(self.R), torch.transpose(muY)
        )

    def transform_point_cloud(self, source=None):
        """
        Update a point cloud using the new estimate of the rigid transformation.

        Attributes
        ----------
        source: numpy array
            Point cloud to be transformed - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.source used.


        Returns
        -------
        If source is None, returns None.
        Otherwise, returns the transformed source.
        """
        if source is None:
            self.transformed_source = self.s * np.dot(self.source, self.R) + self.t
            return
        else:
            return self.s * np.dot(source, self.R) + self.t

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAR = torch.trace(np.dot(self.A, self.R))
        xPx = np.dot(
            torch.transpose(self.Pt1),
            torch.sum(torch.multiply(self.target_hat, self.target_hat), axis=1),
        )
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / (
            2 * self.sigma2
        ) + self.dimensionality * self.sum_probs / 2 * torch.log(self.sigma2)
        self.diff = torch.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.sum_probs * self.dimensionality)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the rigid transformation parameters.

        Returns
        -------
        self.s: float
            Current estimate of the scale factor.

        self.R: numpy array
            Current estimate of the rotation matrix.

        self.t: numpy array
            Current estimate of the translation vector.
        """
        return self.s, self.R, self.t


def is_positive_semi_definite(R):
    return torch.all(torch.linalg.eigvals(R) > 0)
