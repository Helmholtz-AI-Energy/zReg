"""Configuration utilities for zReg.

This module provides explicit configuration functions for PyTorch settings
that affect global state. These settings are NOT applied at import time;
users must call configure_pytorch() explicitly if they want these defaults.

Examples
--------
>>> from zreg.config import configure_pytorch
>>> configure_pytorch()  # Apply recommended settings
>>> configure_pytorch(matmul_precision="highest")  # Custom precision
"""

import logging
import torch

__all__ = ["configure_pytorch"]

log = logging.getLogger(__name__)


def configure_pytorch(
    matmul_precision: str = "high",
    default_device: str | torch.device | None = None,
) -> None:
    """Configure PyTorch global settings for optimal zReg performance.

    This function applies PyTorch configuration settings that were previously
    set at module import time. Calling this function is OPTIONAL; zReg will
    work correctly without it, but these settings may improve performance.

    Parameters
    ----------
    matmul_precision : str, optional
        Float32 matrix multiplication precision. Options: "highest", "high",
        "medium". Default "high" uses TensorFloat-32 on supported hardware.
        See torch.set_float32_matmul_precision() for details.
    default_device : str or torch.device, optional
        Default device for tensor creation. If None, no default device is set
        and tensors are created on CPU unless explicitly specified.
        WARNING: Setting a global default device affects ALL tensor operations
        in your process, not just zReg.

    Examples
    --------
    >>> from zreg.config import configure_pytorch
    >>> configure_pytorch()  # Use recommended defaults
    >>> configure_pytorch(matmul_precision="highest")  # Maximum precision
    >>> configure_pytorch(default_device="cuda:0")  # GPU default (use carefully)

    Notes
    -----
    Previously, zReg set these configurations at import time:
    - torch.set_float32_matmul_precision("high") in cpd/base.py
    - torch.set_default_device(device) in distances/sw_varients.py

    This was problematic because:
    1. Users embedding zReg had these settings silently applied
    2. Different modules could conflict (e.g., different default devices)
    3. No way to opt out

    Now these settings require explicit opt-in via this function.
    """
    torch.set_float32_matmul_precision(matmul_precision)
    log.debug(f"Set torch.float32_matmul_precision to '{matmul_precision}'")

    if default_device is not None:
        torch.set_default_device(default_device)
        log.debug(f"Set torch.default_device to '{default_device}'")
