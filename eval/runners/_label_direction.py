"""Label-transfer direction contract shared by the eval runner and the optimizer.

Phase 59 NUM-04 / D-01.  In paired Kobitski->Shah runs the *target* (Shah)
carries the 3-class germ-layer labels (``load_shah_from_csv`` puts the
``layer`` column into ``pc["label"]``), while the *source* (Kobitski) is
unlabeled — its ``label`` field holds arbitrary colour indices.  Transferring
source->target there writes meaningless Kobitski colour indices onto Shah.
``c60a943`` introduced ``EvalConfig.label_source`` so such configs can take
labels from the target; ``c68c63c`` removed it on the false premise that the
source is always the label provider, which silently re-reversed the transfer
direction for every Kobitski->Shah run.  This module restores the switch in
one place so that ``EvaluationRunner`` (plan 59-05) and ``HyperparamOptimizer``
(plan 59-06) cannot drift apart.

Both functions are pure: no config mutation, no I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    # zreg.* before torch before eval.* (macOS-ARM libomp SIGABRT); eval.config
    # imports zreg internally, so importing it only for typing is safe.
    from eval.config import EvalConfig

__all__ = ["resolve_label_transfer_pair", "f1_unavailable_reason"]

_T = TypeVar("_T")

_F1_UNAVAILABLE_PAIRED_TARGET = (
    "f1 unavailable: pipeline_mode='paired' with label_source='target' has no ground truth "
    "on the receiving (aligned source) side; f1_score zero-filled"
)
_F1_UNAVAILABLE_PAIRED_TARGET_GT = (
    "f1 unavailable: pipeline_mode='paired' with label_source='target' and ground_truth_path "
    "set is not supported yet (neither scorer pairs the ground truth with the receiver frame "
    "via the warp path); f1_score zero-filled"
)


def resolve_label_transfer_pair(config: EvalConfig, aligned: _T, target: _T) -> tuple[_T, _T]:
    """Return ``(provider, receiver)`` for ``LabelTransferStage.run``.

    Parameters
    ----------
    config : EvalConfig
        Evaluation config; only ``label_source`` is read.
    aligned : dict
        The aligned source trajectory (``frame -> zRegPointCloud``).
    target : dict
        The target trajectory (``frame -> zRegPointCloud``).

    Returns
    -------
    tuple
        ``(target, aligned)`` when ``config.label_source == "target"``,
        otherwise ``(aligned, target)``.  The objects are returned by identity,
        never copied.

    Notes
    -----
    The ``LabelTransferStage`` output is keyed by the *receiver's* frame keys.
    Every consumer that pairs transferred labels with positions must therefore
    read positions from the receiver, and every consumer that shows the
    original labels must read them from the provider.
    """
    if config.label_source == "target":
        return target, aligned
    return aligned, target


def f1_unavailable_reason(config: EvalConfig) -> str | None:
    """Explain why F1 cannot be computed for this config, or return ``None``.

    In paired mode with ``label_source='target'`` the labels flow onto the
    aligned source (Kobitski), which has no ground-truth class labels, so F1
    is undefined; callers zero-fill ``f1_score`` and record this reason as an
    explicit sanity flag (orchestrator resolution A2/A3) instead of reporting
    a silently wrong F1.

    An external ``ground_truth_path`` does not make F1 available yet (WR-04):
    ``EvaluationRunner`` would compare the GT of the last *source* frame with
    labels on the DTW-matched receiver frame, and ``HyperparamOptimizer``
    ignores ``ground_truth_path`` entirely.  Until both scorers read the GT
    per receiver frame through the warp path, this combination is reported
    as unavailable too.

    Parameters
    ----------
    config : EvalConfig
        Evaluation config; ``pipeline_mode``, ``label_source`` and
        ``ground_truth_path`` are read.

    Returns
    -------
    str or None
        A non-empty, human-readable reason when F1 is unavailable, else ``None``.
    """
    if config.pipeline_mode == "paired" and config.label_source == "target":
        if config.ground_truth_path is not None:
            return _F1_UNAVAILABLE_PAIRED_TARGET_GT
        return _F1_UNAVAILABLE_PAIRED_TARGET
    return None
