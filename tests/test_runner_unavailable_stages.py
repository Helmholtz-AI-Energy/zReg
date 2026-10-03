"""Unavailable-stage metrics in EvaluationRunner are never reported as perfect.

Phase 59 NUM-05 / D-05 (review MEDIUM, eval_runner.py:398): when
``run_alignment`` is false, the alignment metrics (chamfer, hausdorff,
path_smoothness, temporal_stability) were overwritten with ``0.0``, which
normalises to ``1.0`` = perfect and hid a disjoint-frame ``inf`` from plan
59-04.  They are now reported as ``+inf`` (normalised ``0.0``) with an explicit
``"stage unavailable: alignment ..."`` sanity flag.

Every runner here is constructible: ``EvaluationRunner.__init__`` rejects
a config with both stage flags off (eval_runner.py ``"At least one stage must be enabled"``), so every
disabled-alignment fixture keeps label transfer on (knn_voting, labelled
source).  Only data loading (the I/O boundary) is stubbed; the stages and the
metrics engine are real.
"""

import json
import math

# zreg.* before torch before eval.* (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_labels, generate_trajectory

import pytest
import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.metrics import MetricsEngine
from eval.runners import EvaluationRunner

ALIGN_PARAMS = {
    "window_size": 10,
    "step": 1,
    "cpd_penalty": "rigid",
    "dtw_dist_fn": "euclidean",
    "n_breakpoints": 5,
    "alignment_method": "cpd",
}
LT_PARAMS = {"k_neighbours": 3, "dist_metric": "euclidean", "smoothing": 0.0, "threshold": 0.5}
PARAMS = {**ALIGN_PARAMS, **LT_PARAMS}

ALIGN_METRICS = ("chamfer_distance", "hausdorff_distance", "path_smoothness", "temporal_stability")
ALIGN_NORM_KEYS = ("chamfer", "hausdorff", "path_smoothness", "temporal_stability")


def _labelled(keys, n_points=30, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k in keys:
        pos = torch.rand(n_points, 3, generator=gen)
        label = torch.arange(n_points) % 4
        out[k] = zRegPointCloud(pos=pos, label=label, id=label.clone())
    return out


def _base_config(tmp_path, **kwargs):
    base = dict(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        run_alignment=False,
        run_label_transfer=True,
        label_transfer_method="knn_voting",
        save_plots=False,
    )
    base.update(kwargs)
    return EvalConfig(**base)


def _runner(config):
    runner = EvaluationRunner(config, dict(PARAMS))
    runner.factory = DataFactory(config)
    return runner


def test_both_stages_disabled_still_rejected(tmp_path):
    """The invariant the fixtures rely on: both stages off is rejected at construction."""
    config = _base_config(tmp_path, run_label_transfer=False)
    with pytest.raises(ValueError, match="At least one stage must be enabled"):
        EvaluationRunner(config, dict(PARAMS))


def test_disabled_alignment_identical_clouds_not_perfect(tmp_path):
    """Identical source/target clouds with alignment off must not score as perfect alignment."""
    data = _labelled(range(3))
    config = _base_config(tmp_path)
    runner = _runner(config)
    result = runner._run_single(data, data, runner.params)
    metrics = result["metrics"]
    for name in ALIGN_METRICS:
        assert getattr(metrics, name) == math.inf, name
    for key in ALIGN_NORM_KEYS:
        assert metrics.normalized[key] == 0.0, key
    assert any(f.startswith("stage unavailable: alignment") for f in result["sanity_flags"])
    assert MetricsEngine(config).compute_score(metrics) < 1.0


def test_disabled_alignment_disjoint_frames_not_perfect(tmp_path):
    """Disjoint frame keys with alignment off: chamfer/hausdorff normalise to 0.0, not 1.0."""
    source = _labelled([0, 1, 2], seed=1)
    target = _labelled([10, 11, 12], seed=2)
    runner = _runner(_base_config(tmp_path))
    result = runner._run_single(source, target, runner.params)
    assert result["metrics"].normalized["chamfer"] == 0.0
    assert result["metrics"].normalized["hausdorff"] == 0.0


def test_disabled_alignment_report_serialises(tmp_path, monkeypatch):
    """A full run with inf alignment metrics still writes a loadable report and plots."""
    source = _labelled(range(3), seed=1)
    target = _labelled(range(3), seed=2)
    monkeypatch.setattr(DataFactory, "load_real", lambda self: source)
    monkeypatch.setattr(DataFactory, "load_target", lambda self: target)
    config = _base_config(tmp_path, save_plots=True)
    report = EvaluationRunner(config, dict(PARAMS)).run()
    assert report.metrics.chamfer_distance == math.inf
    with open(tmp_path / "out" / "eval_report.json") as f:
        data = json.load(f)
    # Phase 63 IN-01: strict JSON on disk — null plus a non_finite_fields marker.
    assert data["metrics"]["chamfer_distance"] is None
    assert "metrics.chamfer_distance" in data["non_finite_fields"]
    assert (tmp_path / "out" / "metrics_summary.pdf").exists()


def test_enabled_alignment_unchanged(tmp_path):
    """With alignment on, chamfer stays finite and no alignment-unavailable flag is raised."""
    source = generate_labels(generate_trajectory(n_points=20, n_frames=4, seed=0), n_labels=4, seed=0)
    target = generate_labels(generate_trajectory(n_points=20, n_frames=4, seed=42), n_labels=4, seed=0)
    runner = _runner(_base_config(tmp_path, run_alignment=True))
    result = runner._run_single(source, target, runner.params)
    assert math.isfinite(result["metrics"].chamfer_distance)
    assert not any(f.startswith("stage unavailable: alignment") for f in result["sanity_flags"])
