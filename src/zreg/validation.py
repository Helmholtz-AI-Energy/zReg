"""Input validation utilities for zReg public API entry points.

Provides _validate_tensors() to check for NaN/inf values and device consistency
before any computation begins, enabling early and informative failure.
"""

import torch
from typing import Sequence

__all__ = ["_validate_tensors"]


def _validate_tensors(
    *tensors: torch.Tensor,
    names: Sequence[str],
    check_finite: bool = True,
    check_device: bool = True,
) -> None:
    """Validate one or more tensors before computation.

    Checks that:
    1. The number of tensors matches the number of names.
    2. (If check_finite) Each tensor contains only finite values (no NaN/inf).
    3. (If check_device and len(tensors) > 1) All tensors reside on the same device.

    Parameters
    ----------
    *tensors : torch.Tensor
        One or more tensors to validate.
    names : Sequence[str]
        Names corresponding to each tensor (used in error messages).
    check_finite : bool, optional
        Whether to check for NaN/inf values. Default: True.
    check_device : bool, optional
        Whether to check that all tensors are on the same device. Default: True.

    Raises
    ------
    ValueError
        If the number of tensors does not match the number of names.
        If any tensor contains NaN or inf values (when check_finite=True).
        If tensors reside on different devices (when check_device=True and len > 1).
    """
    if len(tensors) != len(names):
        raise ValueError(
            f"_validate_tensors received {len(tensors)} tensor(s) but {len(names)} name(s). "
            "The number of tensors and names must match."
        )

    if check_finite:
        for tensor, name in zip(tensors, names):
            if not torch.isfinite(tensor).all():
                if torch.isnan(tensor).any():
                    kind = "NaN"
                else:
                    kind = "inf"
                raise ValueError(
                    f"Expected {name} to be a finite tensor, but found {kind} values in {name} "
                    f"(shape: {tuple(tensor.shape)})"
                )

    if check_device and len(tensors) > 1:
        ref_name = names[0]
        ref_device = tensors[0].device
        for tensor, name in zip(tensors[1:], names[1:]):
            if tensor.device != ref_device:
                raise ValueError(
                    f"{ref_name} is on {ref_device} but {name} is on {tensor.device} "
                    f"\u2014 all tensors must be on the same device"
                )
