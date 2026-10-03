"""Every JSON artifact this repo writes is strict RFC-8259 JSON (Phase 63 IN-01, D-04).

Since Phase 59 non-finite metrics are part of the result contract: failed HPO
trials score ``-inf`` and unavailable stages/metrics are reported as ``+inf``
(normalised ``0.0``).  Python's ``json.dump`` writes those as the literals
``Infinity`` / ``NaN``, which jq, JavaScript and other strict parsers reject.
The writers now replace non-finite floats with ``null`` and record the dotted
paths in a ``non_finite_fields`` list (top level of ``eval_report.json`` and
``benchmark_report.json``; per trial dict in ``search_history.json``).
``best_params.json`` stays a flat params dict without a marker key (D-09).

Strictness oracle: ``json.loads(text, parse_constant=_raise)``.  The stdlib
calls ``parse_constant`` only for ``Infinity``, ``-Infinity`` and ``NaN``, so a
non-strict file makes ``_assert_strict`` fail with ``pytest.fail``.

The writers under test are real (``EvaluationRunner.run`` / ``save_report``,
``HyperparamOptimizer.run`` / ``save_best_params``,
``LabelTransferBenchmark.save_report``).  Only data loading (the I/O
boundary) is stubbed in the evaluation-runner test.  ``eval.strict_json`` is
imported only inside the helper unit tests so the writer tests reach their
strict-parse assertion on trees that do not have the helper.
"""

import json
import math
from pathlib import Path

# zreg.* before torch before eval.* (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud

import pytest
import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.runners import EvaluationRunner, LabelTransferBenchmark
from eval.runners.optimizer import HyperparamOptimizer
from eval.types import BenchmarkReport, MethodBenchmarkResult

INF = float("inf")
NAN = float("nan")


def _raise(token):
    raise ValueError(f"non-strict JSON constant {token!r}")


def _assert_strict(path):
    """Parse ``path`` strictly; a non-finite literal is an assertion-level failure."""
    text = Path(path).read_text()
    try:
        return json.loads(text, parse_constant=_raise)
    except ValueError as e:
        pytest.fail(f"non-strict JSON in {path}: {e}")


# ---------------------------------------------------------------------------
# Helper unit tests (new contract: eval.strict_json)
# ---------------------------------------------------------------------------


def test_sanitize_non_finite_replaces_and_records_paths():
    from eval.strict_json import sanitize_non_finite

    obj = {"a": INF, "b": [1.0, NAN], "c": {"d": -INF}, "e": 2.0}
    clean, paths = sanitize_non_finite(obj)
    assert clean == {"a": None, "b": [1.0, None], "c": {"d": None}, "e": 2.0}
    assert paths == ["a", "b[1]", "c.d"]
    # Input is not mutated.
    assert obj["a"] == INF and math.isnan(obj["b"][1]) and obj["c"]["d"] == -INF


def test_sanitize_non_finite_prefix_tuple_and_passthrough():
    from eval.strict_json import sanitize_non_finite

    clean, paths = sanitize_non_finite({"x": (INF, 1, "s", None, True)}, path="root")
    assert clean == {"x": [None, 1, "s", None, True]}
    assert paths == ["root.x[0]"]
    assert sanitize_non_finite([]) == ([], [])


def test_dump_strict_output_parses_strictly(tmp_path):
    from eval.strict_json import dump_strict

    out = tmp_path / "x.json"
    with open(out, "w") as f:
        dump_strict({"a": [INF, -INF, NAN, 0.5]}, f, indent=2)
    assert _assert_strict(out) == {"a": [None, None, None, 0.5]}


# ---------------------------------------------------------------------------
# eval_report.json (EvaluationRunner.save_report)
# ---------------------------------------------------------------------------

_PARAMS = {
    "window_size": 10,
    "step": 1,
    "cpd_penalty": "rigid",
    "dtw_dist_fn": "euclidean",
    "n_breakpoints": 5,
    "alignment_method": "cpd",
    "k_neighbours": 3,
    "dist_metric": "euclidean",
    "smoothing": 0.0,
    "threshold": 0.5,
}


def _labelled(keys, n_points=30, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k in keys:
        pos = torch.rand(n_points, 3, generator=gen)
        label = torch.arange(n_points) % 4
        out[k] = zRegPointCloud(pos=pos, label=label, id=label.clone())
    return out


def test_eval_report_strict_json(tmp_path, monkeypatch):
    """Alignment disabled -> chamfer is +inf in memory, null + marker on disk."""
    source = _labelled(range(3), seed=1)
    target = _labelled(range(3), seed=2)
    monkeypatch.setattr(DataFactory, "load_real", lambda self: source)
    monkeypatch.setattr(DataFactory, "load_target", lambda self: target)
    config = EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="paired",
        run_alignment=False,
        run_label_transfer=True,
        label_transfer_method="knn_voting",
        save_plots=False,
    )
    report = EvaluationRunner(config, dict(_PARAMS)).run()
    assert report.metrics.chamfer_distance == math.inf  # in-memory contract unchanged

    data = _assert_strict(tmp_path / "out" / "eval_report.json")
    assert data["metrics"]["chamfer_distance"] is None
    assert "metrics.chamfer_distance" in data["non_finite_fields"]
    assert data["metrics"]["f1_score"] is not None


# ---------------------------------------------------------------------------
# HPO artifacts (HyperparamOptimizer.run / save_best_params)
# ---------------------------------------------------------------------------


def _hpo_cfg(tmp_path, k_values) -> EvalConfig:
    """Cheap sanity-tier synthetic config; k=3 succeeds, k=10000 raises."""
    return EvalConfig(
        data_path=str(tmp_path / "unused.mat"),
        output_dir=str(tmp_path / "out"),
        pipeline_mode="synthetic",
        run_alignment=True,
        run_label_transfer=True,
        transform_spec={
            "type": "subsample_pair",
            "synthesize": True,
            "seed": 1,
            "n_classes": 3,
            "n_points": 80,
            "source_fraction": 0.8,
            "target_fraction": 0.8,
            "rotation_deg": 0.0,
            "rotation_axis": [0.0, 0.0, 1.0],
            "scale_factor": 1.0,
        },
        tier="sanity",
        n_trials=1,
        search_strategy="grid",
        search_space={"k_neighbours": list(k_values)},
    )


def test_search_history_and_best_params_strict_json(tmp_path):
    """A real HPO trial has temporal_stability=+inf (no per-frame transforms)."""
    result = HyperparamOptimizer(_hpo_cfg(tmp_path, [3])).run()
    assert any(not math.isfinite(t.metrics.temporal_stability) for t in result.history), (
        "scenario precondition: a history trial carries a non-finite metric"
    )
    out = tmp_path / "out"

    history = _assert_strict(out / "search_history.json")
    assert len(history) == len(result.history)
    for trial in history:
        assert trial["metrics"]["temporal_stability"] is None
        assert "metrics.temporal_stability" in trial["non_finite_fields"]

    best = _assert_strict(out / "best_params.json")
    assert best == result.best_params  # D-09: flat params dict
    assert "non_finite_fields" not in best

    _assert_strict(out / "failed_trials.json")


def test_failed_trials_strict_json(tmp_path):
    """All trials failing: run() raises after writing a strict failed_trials.json."""
    with pytest.raises(RuntimeError, match="All 1 HPO trials failed"):
        HyperparamOptimizer(_hpo_cfg(tmp_path, [10000])).run()
    records = _assert_strict(tmp_path / "out" / "failed_trials.json")
    assert len(records) == 1
    assert records[0]["params"] == {"k_neighbours": 10000}


def test_save_best_params_failed_trials_non_finite_strict(tmp_path):
    """A failure record carrying a non-finite value is still written strictly."""
    opt = HyperparamOptimizer(_hpo_cfg(tmp_path, [3]))
    result = opt.run()
    record = {"params": {"k_neighbours": 3}, "tier": "sanity", "score": -INF,
              "error": "non-finite score", "error_type": "ValueError", "rank": 0}
    result = result.model_copy(update={"failed_trials": [record]})
    out = tmp_path / "again"
    out.mkdir()
    opt.save_best_params(result, out)
    records = _assert_strict(out / "failed_trials.json")
    assert records[0]["score"] is None


# ---------------------------------------------------------------------------
# benchmark_report.json (LabelTransferBenchmark.save_report)
# ---------------------------------------------------------------------------


def test_benchmark_report_strict_json(tmp_path):
    report = BenchmarkReport(
        params={"k_neighbours": 5},
        results=[
            MethodBenchmarkResult(
                method="knn_voting", dataset_source="synthetic_holdout",
                f1_score=NAN, knn_consistency=0.9, latency_seconds=INF, n_pairs=3,
            ),
        ],
    )
    config = EvalConfig(data_path=str(tmp_path / "unused.mat"), output_dir=str(tmp_path / "out"))
    out_path = LabelTransferBenchmark(config).save_report(report, tmp_path)
    data = _assert_strict(out_path)
    assert data["results"][0]["f1_score"] is None
    assert data["results"][0]["latency_seconds"] is None
    assert data["results"][0]["knn_consistency"] == 0.9
    assert data["non_finite_fields"] == ["results[0].f1_score", "results[0].latency_seconds"]
