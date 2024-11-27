import torch

from .expect_max_reg import ExpMaxRegistration


class RigidRegistration(ExpMaxRegistration):
    """
    Rigid registration.

    Attributes
    ----------
    rotm: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.

    translationv: numpy array
        1xD initial translation vector.

    scale: float (positive)
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

    def __init__(self, rotm=None, translationv=None, scale=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.dimensionality != 2 and self.dimensionality != 3:
            raise ValueError(
                "Rigid registration only supports 2D or 3D point clouds. Instead got {}.".format(
                    self.dimensionality
                )
            )

        if rotm is not None and (
            (rotm.ndim != 2)
            or (rotm.shape[0] != self.dimensionality)
            or (rotm.shape[1] != self.dimensionality)
            or not is_positive_semi_definite(rotm)
        ):
            raise ValueError(
                "The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.".format(
                    self.dimensionality, self.dimensionality, rotm
                )
            )

        if translationv is not None and (
            (translationv.ndim != 2)
            or (translationv.shape[0] != 1)
            or (translationv.shape[1] != self.dimensionality)
        ):
            raise ValueError(
                "The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.".format(
                    self.dimensionality, translationv
                )
            )

        if scale is not None and scale <= 0:
            raise ValueError(
                "The scale factor must be a positive number. Instead got: {}.".format(
                    scale
                )
            )

        self.rotm = (
            torch.eye(self.dimensionality, **self.fact) if rotm is None else rotm
        )
        self.translationv = (
            torch.atleast_2d(torch.zeros((1, self.dimensionality)), **self.fact)
            if translationv is None
            else translationv
        )
        self.scale = 1.0 if scale is None else scale

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # target point cloud mean
        mu_target = torch.divide(torch.sum(self.probs_targets, axis=0), self.sum_probs)
        # source point cloud mean
        mu_source = torch.divide(
            torch.sum((self.probs.T @ self.source), axis=0), self.sum_probs
        )  # dot

        self.target_hat = self.target - torch.tile(mu_target, (self.num_targ_pts, 1))
        # centered source point cloud (Y)
        Y_hat = self.source - torch.tile(mu_source, (self.num_src_pts, 1))
        self.YPY = torch.dot(
            self.sum_probs_source.T,
            torch.sum(torch.multiply(Y_hat, Y_hat), axis=1),
        )  # dot

        self.A = self.target_hat.T @ self.probs.T  # dot
        self.A = torch.dot(self.A, Y_hat)  # dot

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        u, _, vh = torch.linalg.svd(self.A, full_matrices=True)
        c = torch.ones((self.dimensionality,), self.fact)
        c[self.dimensionality - 1] = torch.linalg.det(u @ vh)  # dot

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.rotm = (u @ torch.diag(c) @ vh).T  # dot
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.scale != 1.0:
            self.scale = torch.trace(torch.dot(self.A.T, self.rotm.T)) / self.YPY  # dot

        self.translationv = mu_target.T - self.scale * torch.dot(
            self.rotm.T, mu_source.T
        )  # dot

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
            self.transformed_source = (
                self.scale * (self.source @ self.rotm) + self.translationv
            )  # dot
            return
        else:
            return self.scale * (source @ self.rotm) + self.translationv  # dot

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAR = torch.trace(self.A @ self.rotm)
        xPx = torch.dot(
            self.Pt1.T,
            torch.sum(torch.multiply(self.target_hat, self.target_hat), axis=1),
        )  # dot
        self.q = (xPx - 2 * self.scale * trAR + self.scale * self.scale * self.YPY) / (
            2 * self.sigma2
        )
        self.q += self.dimensionality * self.sum_probs / 2 * torch.log(self.sigma2)
        self.diff = torch.abs(self.q - qprev)
        self.sigma2 = (xPx - self.scale * trAR) / (self.sum_probs * self.dimensionality)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the rigid transformation parameters.

        Returns
        -------
        self.scale: float
            Current estimate of the scale factor.

        self.rotm: numpy array
            Current estimate of the rotation matrix.

        self.translationv: numpy array
            Current estimate of the translation vector.
        """
        return self.scale, self.rotm, self.translationv


def is_positive_semi_definite(rotm):
    return torch.all(torch.linalg.eigvals(rotm) > 0)
