"""label_metadata.json describes exactly the frames of label_trajectory.csv (VIZ-03).

Phase 63 VIZ-03 hardening.  The fix itself is Phase 59 ``19d0e3d``: label
metadata ``frame_count`` / ``frame_indices`` come from the transferred
(receiver) frame keys instead of ``len(dataset)``.  Before that, a run whose
source had 5 frames and whose target had 3 reported ``frame_count == 5`` while
the CSV held only 3 frames.  This module pins the invariant directly: the
metadata frame set equals the frame set read back from the CSV, in both label
directions.

Real runner + exporter, no mocks: ``EvaluationRunner._run_single`` (real
``LabelTransferStage``, ``MetricsEngine``) and ``export_trajectory``.  The
DataFactory is real; ``ground_truth_path`` is unset, so no file I/O happens.
Helpers are copied from tests/test_label_direction.py (not imported).
"""

import csv
import json
from pathlib import Path

# zreg.* before torch before eval.* (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud

import pytest
import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.runners import EvaluationRunner
from eval.tracking import export_trajectory

PROVIDER_CLASSES = {10, 11, 12}
SOURCE_CLASSES = set(range(8))
N_TARGET = 40
N_SOURCE = 25
LT_PARAMS = {"k_neighbours": 3, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.5}


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


def _export(tmp_path, **cfg_kwargs) -> Path:
    """5 source frames vs 3 target frames through the real runner and exporter."""
    source = _labelled_traj(N_SOURCE, SOURCE_CLASSES, range(5), seed=1)
    target = _labelled_traj(N_TARGET, PROVIDER_CLASSES, range(3), seed=2)
    config = _paired_cfg(tmp_path, **cfg_kwargs)
    runner = EvaluationRunner(config, dict(LT_PARAMS))
    runner.factory = DataFactory(config)
    result = runner._run_single(source, target, runner.params)
    out = tmp_path / "exp"
    export_trajectory(result, source, config, out, target=target)
    return out


def _csv_frames(path: Path) -> set[int]:
    with open(path, newline="") as f:
        return {int(row["frame_idx"]) for row in csv.DictReader(f)}


def _raise(token):
    raise ValueError(f"non-strict JSON constant {token!r}")


def _assert_metadata_matches_csv(out: Path) -> None:
    frames = _csv_frames(out / "label_trajectory.csv")
    assert frames, "label_trajectory.csv has no rows"
    with open(out / "label_metadata.json") as f:
        meta = json.load(f)
    assert meta["frame_count"] == len(frames)
    assert meta["frame_indices"] == sorted(frames)


def test_default_direction_metadata_matches_csv(tmp_path):
    """Default direction (label_source 'source'): metadata frames == CSV frames."""
    _assert_metadata_matches_csv(_export(tmp_path))


def test_target_direction_metadata_matches_csv(tmp_path):
    """label_source 'target' (5 receiver vs 3 provider frames): metadata frames == CSV frames."""
    _assert_metadata_matches_csv(_export(tmp_path, label_source="target"))


@pytest.mark.parametrize("cfg_kwargs", [{}, {"label_source": "target"}], ids=["source", "target"])
def test_label_metadata_strict_json(tmp_path, cfg_kwargs):
    """label_metadata.json parses strictly (no Infinity/NaN literals)."""
    out = _export(tmp_path, **cfg_kwargs)
    text = (out / "label_metadata.json").read_text()
    try:
        json.loads(text, parse_constant=_raise)
    except ValueError as e:
        pytest.fail(f"non-strict JSON in label_metadata.json: {e}")
