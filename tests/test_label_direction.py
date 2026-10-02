"""Tests for eval.runners._label_direction (Phase 59 NUM-04, D-01).

The helper is the single place that decides which cloud provides the labels
for ``LabelTransferStage`` and when F1 is unavailable because the receiving
side has no ground truth.  Plans 59-05 (eval_runner) and 59-06 (optimizer)
consume it; this module pins its contract.
"""

from eval.config import EvalConfig
from eval.runners._label_direction import f1_unavailable_reason, resolve_label_transfer_pair


def _cfg(tmp_path, **kwargs):
    return EvalConfig(data_path=str(tmp_path / "data.tracklets"), **kwargs)


def test_resolve_pair_default_source(tmp_path):
    """label_source='source' keeps (aligned, target) as (provider, receiver)."""
    aligned, target = {"a": 1}, {"b": 2}
    provider, receiver = resolve_label_transfer_pair(_cfg(tmp_path, label_source="source"), aligned, target)
    assert provider is aligned
    assert receiver is target


def test_resolve_pair_target(tmp_path):
    """label_source='target' swaps: the target provides, the aligned source receives."""
    aligned, target = {"a": 1}, {"b": 2}
    cfg = _cfg(tmp_path, label_source="target", pipeline_mode="paired")
    provider, receiver = resolve_label_transfer_pair(cfg, aligned, target)
    assert provider is target
    assert receiver is aligned


def test_f1_unavailable_reason(tmp_path):
    """A reason is returned only for paired + target + no external ground truth."""
    assert f1_unavailable_reason(_cfg(tmp_path)) is None

    reason = f1_unavailable_reason(_cfg(tmp_path, pipeline_mode="paired", label_source="target"))
    assert isinstance(reason, str) and reason
    assert "label_source='target'" in reason

    with_gt = _cfg(
        tmp_path,
        pipeline_mode="paired",
        label_source="target",
        ground_truth_path=str(tmp_path / "gt.csv"),
    )
    assert f1_unavailable_reason(with_gt) is None
