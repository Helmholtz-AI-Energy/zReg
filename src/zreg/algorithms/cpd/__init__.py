"""Coherent Point Drift (CPD) registration algorithms.

This package provides implementations of the CPD algorithm for point cloud
registration, supporting rigid, affine, and non-rigid transformations.

Public API
----------
Classes:
    CoherentPointDrift : Abstract base class for CPD algorithms
    RigidCPD : Rigid transformation registration
    AffineCPD : Affine transformation registration
    NonRigidCPD : Non-rigid (deformable) registration
    ConstrainedNonRigidCPD : Non-rigid registration with point constraints

Functions:
    cpd_registration : Convenience function for registration
    init_cpd_from_existing : Initialize CPD from existing transformation

Types:
    EstepResult : E-step result namedtuple
    MstepResult : M-step result namedtuple with convergence diagnostics

Utilities:
    rbf_kernel_matrix : RBF kernel computation for non-rigid registration

Constants:
    SHAH_KOBITSKI_EMPIRICAL_INIT : Historical opt-in RigidCPD start (general
        linear 3x3, det = 1, not a rotation); RigidCPD defaults to identity
"""

# Types (always needed)
from ._types import EstepResult, MstepResult

# Base class
from .base import CoherentPointDrift

# Registration variants
from .rigid import RigidCPD, SHAH_KOBITSKI_EMPIRICAL_INIT
from .affine import AffineCPD
from .nonrigid import NonRigidCPD, ConstrainedNonRigidCPD

# Convenience functions
from ._registration import cpd_registration, init_cpd_from_existing

# Kernel utilities
from .kernels import rbf_kernel_matrix

__all__ = [
    # Base class
    "CoherentPointDrift",
    # Registration variants
    "RigidCPD",
    "AffineCPD",
    "NonRigidCPD",
    "ConstrainedNonRigidCPD",
    # Convenience functions
    "cpd_registration",
    "init_cpd_from_existing",
    # Types
    "EstepResult",
    "MstepResult",
    # Utilities
    "rbf_kernel_matrix",
    # Constants
    "SHAH_KOBITSKI_EMPIRICAL_INIT",
]
