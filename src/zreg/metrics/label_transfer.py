"""Label transfer F1 metric with sentinel masking."""

import torch
from sklearn.metrics import f1_score as _sklearn_f1

__all__ = ["compute_f1"]


def compute_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = "weighted",
    zero_division: int = 0,
) -> float:
    """Compute F1 score for label transfer evaluation with sentinel masking.

    Applies a sentinel mask to exclude entries where ``y_true == -1`` before
    computing the F1 score via scikit-learn. Both tensors are moved to CPU and
    converted to NumPy before delegation to sklearn.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth label tensor of shape ``(N,)``. Entries equal to ``-1``
        are treated as unlabelled and are excluded from the computation.
    y_pred : torch.Tensor
        Predicted label tensor of shape ``(N,)``. Must have the same length as
        ``y_true``.
    average : str, optional
        Averaging strategy passed to ``sklearn.metrics.f1_score``. Common
        choices are ``"weighted"`` (default, recommended for HPO) and
        ``"macro"`` (equal weight per class, used for reporting). Any value
        accepted by sklearn is forwarded without modification. Passing an
        unsupported value (e.g. ``"banana"``) will raise a ``ValueError``
        from sklearn.
    zero_division : int, optional
        Value to use when there is a zero division in the F1 computation
        (e.g. a class has no predicted or true positives). Forwarded directly
        to ``sklearn.metrics.f1_score``. Default ``0`` means undefined scores
        contribute zero to the average.

    Returns
    -------
    float
        F1 score as a Python ``float`` in the range ``[0.0, 1.0]``. Returns
        ``0.0`` when all entries are masked out (no labelled samples after
        applying the sentinel mask).

    Raises
    ------
    ValueError
        If ``y_true`` or ``y_pred`` is not 1-D, or if they have different
        lengths.

    Notes
    -----
    **Sentinel masking:** The label ``-1`` is the conventional sentinel code
    used throughout zReg to mark unlabelled or ignored points. Entries where
    ``y_true == -1`` are dropped from both ``y_true`` and ``y_pred`` before
    the F1 computation. This is important for datasets where only a subset of
    points carry ground-truth labels.

    **CPU entry:** scikit-learn does not accept CUDA tensors. Both masked
    arrays are transferred to CPU via ``.detach().cpu().numpy()`` before
    calling sklearn, so CUDA tensors are safe to pass as inputs.

    Examples
    --------
    >>> import torch
    >>> from zreg.metrics.label_transfer import compute_f1
    >>> y = torch.tensor([0, 1, 2, 0, 1])
    >>> compute_f1(y, y)  # perfect prediction
    1.0
    >>> compute_f1(torch.tensor([-1, -1, -1]), torch.tensor([0, 0, 0]))
    0.0
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D, got shapes "
            f"{tuple(y_true.shape)} and {tuple(y_pred.shape)}"
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true length {y_true.shape[0]} != y_pred length {y_pred.shape[0]}"
        )

    mask = y_true != -1
    if mask.sum().item() == 0:
        return 0.0

    y_true_np = y_true[mask].detach().cpu().numpy()
    y_pred_np = y_pred[mask].detach().cpu().numpy()

    score = _sklearn_f1(y_true_np, y_pred_np, average=average, zero_division=zero_division)
    return float(score)
