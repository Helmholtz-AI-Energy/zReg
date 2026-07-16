"""Registration methods: CPD alternatives and wrappers."""

from .icp import ICPRegistration
from .swd_aligner import SlicedWassersteinAligner

__all__ = ["ICPRegistration", "SlicedWassersteinAligner"]
