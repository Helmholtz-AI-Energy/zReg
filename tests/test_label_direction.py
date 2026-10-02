"""Tests for eval.runners._label_direction and its runner wiring (Phase 59 NUM-04, D-01).

The helper is the single place that decides which cloud provides the labels
for ``LabelTransferStage`` and when F1 is unavailable because the receiving
side has no ground truth.  Plans 59-05 (eval_runner) and 59-06 (optimizer)
consume it; this module pins its contract and, from plan 59-05 on, the
end-to-end behaviour of ``EvaluationRunner._run_single`` with the real
``LabelTransferStage`` / ``AlignmentStage`` / ``MetricsEngine`` (no stage
patching).  Unequal point counts and disjoint label sets make a wrong
transfer direction observable.
"""

# zreg.* before torch before eval.* (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud

import pytest
import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.runners import EvaluationRunner
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


# ---------------------------------------------------------------------------
# Plan 59-05: EvaluationRunner._run_single wiring (real stages, no patching)
# ---------------------------------------------------------------------------

PROVIDER_CLASSES = {10, 11, 12}  # "Shah-like" germ-layer labels on the target
SOURCE_CLASSES = set(range(8))  # "Kobitski-like" arbitrary colour indices
N_TARGET = 40
N_SOURCE = 25

LT_PARAMS = {"k_neighbours": 3, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.5}
ALIGN_PARAMS = {
    "window_size": 10,
    "step": 1,
    "cpd_penalty": "rigid",
    "dtw_dist_fn": "euclidean",
    "n_breakpoints": 5,
    "alignment_method": "cpd",
}


def _labelled_traj(n_points, classes, keys, seed):
    """Frames with points in the unit cube and labels cycling through ``classes``."""
    gen = torch.Generator().manual_seed(seed)
    cls = torch.tensor(sorted(classes), dtype=torch.long)
    out = {}
    for k in keys:
        pos = torch.rand(n_points, 3, generator=gen)
        label = cls[torch.arange(n_points) % cls.numel()]
        out[k] = zRegPointCloud(pos=pos, label=label, id=label.clone())
    return out


@pytest.fixture
def kobitski_like():
    """Receiver-side source: 25 points/frame, labels in {0..7}, frames 0..2."""
    return _labelled_traj(N_SOURCE, SOURCE_CLASSES, range(3), seed=1)


@pytest.fixture
def shah_like():
    """Provider-side target: 40 points/frame, labels in {10, 11, 12}, frames 0..2."""
    return _labelled_traj(N_TARGET, PROVIDER_CLASSES, range(3), seed=2)


def _paired_cfg(tmp_path, **kwargs):
    base = dict(
        data_path=str(tmp_path / "kobitski.tracklets"),
        target_data_path=str(tmp_path / "shah.csv"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        run_alignment=False,
        run_label_transfer=True,
        label_transfer_method="knn_voting",
        save_plots=False,
    )
    base.update(kwargs)
    return EvalConfig(**base)


def _runner(config, params=None):
    runner = EvaluationRunner(config, dict(params if params is not None else LT_PARAMS))
    runner.factory = DataFactory(config)  # real factory; ground_truth_path None -> in-memory, no I/O
    return runner


def _values(tensor):
    return set(int(v) for v in tensor.unique().tolist())


def test_run_single_target_direction_transfers_provider_labels(tmp_path, kobitski_like, shah_like):
    """label_source='target': Shah's labels land on the aligned Kobitski frames (D-01)."""
    runner = _runner(_paired_cfg(tmp_path, label_source="target"))
    result = runner._run_single(kobitski_like, shah_like, runner.params)
    transferred = result["label"].transferred_labels
    assert sorted(transferred) == [0, 1, 2]
    for k, labels in transferred.items():
        assert _values(labels) <= PROVIDER_CLASSES, k
        assert labels.shape[0] == N_SOURCE, k


def test_run_single_target_direction_provider_and_receiver(tmp_path, kobitski_like, shah_like):
    """The result carries the provider (target) and receiver (aligned source) by identity."""
    runner = _runner(_paired_cfg(tmp_path, label_source="target"))
    result = runner._run_single(kobitski_like, shah_like, runner.params)
    # stage_input is the source itself when run_alignment is False
    assert result["label_receiver"] is kobitski_like
    assert result["label_provider"] is shah_like
    assert 0.0 <= result["metrics"].knn_consistency <= 1.0


def test_run_single_default_direction_provider_and_receiver(tmp_path, kobitski_like, shah_like):
    """label_source='source' (default) keeps the source->target transfer unchanged."""
    runner = _runner(_paired_cfg(tmp_path))
    result = runner._run_single(kobitski_like, shah_like, runner.params)
    for k, labels in result["label"].transferred_labels.items():
        assert _values(labels) <= SOURCE_CLASSES, k
        assert labels.shape[0] == N_TARGET, k
    assert result["label_provider"] is kobitski_like
    assert result["label_receiver"] is shah_like
    assert not any(f.startswith("f1 unavailable") for f in result["sanity_flags"])


def test_run_single_target_direction_f1_unavailable_flag(tmp_path, kobitski_like, shah_like):
    """Swapped paired mode without external GT: F1 zero-filled (worst) and flagged."""
    runner = _runner(_paired_cfg(tmp_path, label_source="target"))
    result = runner._run_single(kobitski_like, shah_like, runner.params)
    assert result["metrics"].f1_score == 0.0
    assert result["metrics"].normalized["f1"] == 0.0
    assert any(f.startswith("f1 unavailable") for f in result["sanity_flags"])


def test_run_single_label_transfer_disabled_has_no_provider(tmp_path):
    """run_label_transfer=False: no provider/receiver; only the label stage is flagged.

    Unequal per-frame point counts (27 vs 42) exercise the knn fallback that pairs
    target positions with a zero prediction; it must not raise before zero-fill.
    """
    source = _labelled_traj(27, SOURCE_CLASSES, range(4), seed=3)
    target = _labelled_traj(42, PROVIDER_CLASSES, range(4), seed=4)
    config = _paired_cfg(tmp_path, run_alignment=True, run_label_transfer=False)
    runner = _runner(config, {**ALIGN_PARAMS, **LT_PARAMS})
    result = runner._run_single(source, target, runner.params)
    assert result["align"] is not None  # the real AlignmentStage ran
    assert result["label"] is None
    assert result["label_provider"] is None
    assert result["label_receiver"] is None
    assert result["metrics"].f1_score == 0.0
    assert result["metrics"].knn_consistency == 0.0
    flags = result["sanity_flags"]
    assert any(f.startswith("stage unavailable: label transfer") for f in flags)
    assert not any(f.startswith("stage unavailable: alignment") for f in flags)


def test_run_single_target_direction_label_transfer_disabled(tmp_path):
    """label_source='target' with label transfer off: GT is not read, nothing raises."""
    source = _labelled_traj(27, SOURCE_CLASSES, range(4), seed=3)
    target = _labelled_traj(42, PROVIDER_CLASSES, range(4), seed=4)
    config = _paired_cfg(tmp_path, label_source="target", run_alignment=True, run_label_transfer=False)
    runner = _runner(config, {**ALIGN_PARAMS, **LT_PARAMS})
    result = runner._run_single(source, target, runner.params)
    assert result["label_provider"] is None and result["label_receiver"] is None
    assert result["metrics"].f1_score == 0.0
    assert any(f.startswith("stage unavailable: label transfer") for f in result["sanity_flags"])
