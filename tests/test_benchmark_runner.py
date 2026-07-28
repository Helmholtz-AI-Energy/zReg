"""Wave-0 tests for Phase 49's label-transfer benchmarking machinery.

This module is the highest-risk Wave-0 gate for the whole benchmarking track
(49-01-PLAN.md).  It empirically proves RESEARCH Pattern 2 / Assumption A1:
``method='cpd_weighted'`` succeeds when fed RAW (non-CPD-aligned) source/
target point clouds alongside an ``align_result`` obtained from a single
``AlignmentStage(alignment_method='cpd')`` run — i.e. the comparison runner
does NOT need to feed ``align_result.aligned_cloud`` into
``LabelTransferStage.run()``, only the raw trajectories plus the
``align_result`` object itself.  If this assumption were false, Plan 02's
comparison-runner design would need to change before any orchestration code
is written.

It also ships two fixtures reused by Plan 02/03:

- ``benchmark_smoke_checkpoint`` — a freshly-initialized pointnet2/egnn
  checkpoint factory (D-01: smoke-test-only, mirrors
  tests/test_label_transfer_stage.py's ``learned_smoke_checkpoint``).
- ``write_shah_fixture_csv`` — a synthetic Shah-format CSV writer
  (Pitfall 2 — real-data-format fixture), loadable via
  ``DataFactory.load_real()`` with ``config.data_format == "csv"``.

And a self-contained round-trip test for Task 1's new ``eval.types`` models.
"""

import csv
import json
import random

import pytest

# zreg.* / eval.* MUST precede torch — macOS-ARM libomp SIGABRT rule
# (tests/conftest.py:20-24, eval/data_factory.py:19-35, eval/types.py:48-53).
from zreg.data_generation import generate_trajectory, generate_labels
from zreg.models import PointNet2LabelTransfer, EGNNLabelTransfer

import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory, split_seeds
from eval.runners import LabelTransferBenchmark
from eval.stages import AlignmentStage, LabelTransferStage
from eval.types import BenchmarkReport, MethodBenchmarkResult
from train_label_transfer import save_checkpoint

N_CLASSES = 4

# ---------------------------------------------------------------------------
# Shared fixtures — reused by Plan 02 / Plan 03
# ---------------------------------------------------------------------------

LEARNED_METHOD_CASES = [
    (
        "pointnet2",
        "pointnet2_checkpoint_path",
        PointNet2LabelTransfer,
        {"n_classes": N_CLASSES, "hidden_dim": 32},
    ),
    (
        "egnn",
        "egnn_checkpoint_path",
        EGNNLabelTransfer,
        {"n_classes": N_CLASSES, "hidden_dim": 32, "n_layers": 4},
    ),
]


@pytest.fixture
def benchmark_smoke_checkpoint(tmp_path):
    """Factory fixture: build a fresh, untrained model + smoke-test checkpoint.

    Mirrors tests/test_label_transfer_stage.py's ``learned_smoke_checkpoint``
    EXACTLY (D-01 — freshly-initialized weights are all that's needed to
    prove the benchmark runner's load->infer->report *wiring* is correct,
    not model quality; the real, meaningful benchmark run happens later
    against a real trained checkpoint).

    n_classes=N_CLASSES MUST match the data's ``generate_labels(n_classes=...)``
    vocabulary or ``one_hot`` raises ``RuntimeError`` (Pitfall 1).
    """

    def _build(method: str) -> str:
        for m, _field, model_cls, hyperparams in LEARNED_METHOD_CASES:
            if m == method:
                model = model_cls(**hyperparams)
                path = str(tmp_path / f"{method}.pt")
                save_checkpoint(model, method, hyperparams, epoch=0, path=path)
                return path
        raise ValueError(f"unknown method: {method!r}")

    return _build


def write_shah_fixture_csv(path, n_points: int, n_classes: int, seed: int = 0) -> None:
    """Write a synthetic Shah-format CSV fixture (Pitfall 2 — real-data format).

    Writes a header row ``x,y,z,t,layer,id`` followed by ``n_points`` data
    rows, all belonging to a single frame (``t=1``, 1-indexed per
    ``load_shah_from_csv``'s ``t - 1`` conversion).

    Parameters
    ----------
    path : str or Path
        Destination CSV file path.
    n_points : int
        Number of point rows to write.
    n_classes : int
        Layer values are drawn from ``range(n_classes)`` (MUST stay
        ``< n_classes`` per Pitfall 1 — a mismatch with a downstream
        ``n_classes`` vocabulary makes ``one_hot`` raise).
    seed : int, default 0
        Seed for the stdlib ``random.Random`` instance driving point
        coordinates and layer assignment.
    """
    rng = random.Random(seed)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "t", "layer", "id"])
        for i in range(n_points):
            x = rng.uniform(-1.0, 1.0)
            y = rng.uniform(-1.0, 1.0)
            z = rng.uniform(-1.0, 1.0)
            t = 1
            layer = rng.randrange(n_classes)
            writer.writerow([x, y, z, t, layer, i])


# ---------------------------------------------------------------------------
# TestCpdWeightedRawInput — RESEARCH Pattern 2 / Assumption A1 (highest risk)
# ---------------------------------------------------------------------------


class TestCpdWeightedRawInput:
    """Empirically resolves RESEARCH Assumption A1 / Pattern 2 before Plan 02 relies on it.

    ``method='cpd_weighted'`` MUST succeed when fed RAW (non-CPD-aligned)
    source/target point clouds alongside an ``align_result`` obtained from a
    single ``AlignmentStage(alignment_method='cpd')`` run — NOT
    ``align_result.aligned_cloud``. This is the MEDIUM-confidence claim the
    entire Plan 02 comparison-runner design rests on.
    """

    def test_cpd_weighted_raw_input(self, tmp_path) -> None:
        """cpd_weighted succeeds against raw source/target + a real align_result."""
        source = generate_labels(
            generate_trajectory(n_points=15, n_frames=3, seed=100),
            n_classes=N_CLASSES,
            seed=100,
        )
        target = generate_trajectory(n_points=15, n_frames=3, seed=101)

        config = EvalConfig(data_path=str(tmp_path / "unused.mat"))

        align_params = {
            "window_size": 10,
            "step": 1,
            "cpd_penalty": "rigid",  # only "rigid" dispatches correctly upstream (Pitfall 3)
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "alignment_method": "cpd",
        }
        # Run AlignmentStage(alignment_method="cpd") EXACTLY ONCE.
        align_result = AlignmentStage(config).run(source, target, align_params)

        label_params = {
            "k_neighbours": 5,
            "dist_metric": "euclidean",
            "smoothing": 0.0,
            "threshold": 0.0,
            "method": "cpd_weighted",
        }
        # Feed the SAME RAW source/target (NOT align_result.aligned_cloud).
        result = LabelTransferStage(config).run(
            source, target, label_params, align_result=align_result
        )

        assert result.transferred_labels  # non-empty
        target_keys = sorted(target.keys())
        for tk in target_keys[: min(len(source), len(target))]:
            tensor = result.transferred_labels[tk]
            assert tensor.ndim == 1
            n_target_points = target[tk]["pos"].shape[0]
            assert tensor.shape[0] == n_target_points
            assert torch.all(tensor >= 0)
            assert torch.all(tensor < N_CLASSES)


# ---------------------------------------------------------------------------
# TestBenchmarkReportRoundTrip — Task 1 type coverage
# ---------------------------------------------------------------------------


class TestBenchmarkReportRoundTrip:
    """Self-contained coverage of eval.types.MethodBenchmarkResult/BenchmarkReport."""

    def test_json_round_trip(self, tmp_path) -> None:
        """model_dump() -> json.dump() -> json.load() reproduces an equal dict."""
        report = BenchmarkReport(
            params={"k_neighbours": 5},
            results=[
                MethodBenchmarkResult(
                    method="egnn",
                    dataset_source="synthetic_holdout",
                    f1_score=0.87,
                    knn_consistency=0.91,
                    latency_seconds=0.012,
                    n_pairs=3,
                ),
                MethodBenchmarkResult(
                    method="pointnet2",
                    dataset_source="real_shah",
                    error="boom",
                ),
            ],
            notes=["point-count scale mismatch observed (D-03)"],
        )

        dumped = report.model_dump()
        out_path = tmp_path / "report.json"
        with open(out_path, "w") as f:
            json.dump(dumped, f)

        with open(out_path, "r") as f:
            reloaded = json.load(f)

        assert reloaded == dumped
        assert reloaded["results"][1]["error"] == "boom"
        assert reloaded["results"][1]["f1_score"] is None


# ---------------------------------------------------------------------------
# TestShahFixtureCsvLoadable — must_haves truth: loadable via DataFactory
# ---------------------------------------------------------------------------


class TestShahFixtureCsvLoadable:
    """write_shah_fixture_csv output is loadable via DataFactory.load_real(data_format='csv')."""

    def test_loadable_via_data_factory(self, tmp_path) -> None:
        csv_path = tmp_path / "shah_fixture.csv"
        write_shah_fixture_csv(csv_path, n_points=10, n_classes=N_CLASSES, seed=7)

        config = EvalConfig(data_path=str(csv_path), data_format="csv")
        dataset = DataFactory(config).load_real()

        assert dataset  # non-empty
        for frame in dataset.values():
            assert frame["pos"].shape[1] == 3


# ---------------------------------------------------------------------------
# Plan 02 — LabelTransferBenchmark dimension tests (D-02.1-4 + graceful
# degradation + save_report round-trip)
# ---------------------------------------------------------------------------
#
# Shared helpers below reuse this module's `benchmark_smoke_checkpoint`
# fixture and `write_shah_fixture_csv` helper (Plan 01) rather than
# redefining them.
#
# DEVIATION (Rule 1 — auto-fixed bug): 49-02-PLAN.md's Task 1 action text and
# 49-RESEARCH.md's Pattern 1 skeleton both specify
# `params.get("cpd_penalty", 1.0)` as LabelTransferBenchmark.compare_methods's
# internal AlignmentStage default. `AlignmentStage.VALID_CPD = (None, "rigid",
# "affine", "nonrigid")` — a numeric 1.0 default is not a valid CPD-type
# value and would make `AlignmentStage.validate_params` raise unconditionally
# whenever "cpd_weighted" is requested without an explicit override, breaking
# the plan's own must_have that every successful method (including
# cpd_weighted) produces populated f1_score/knn_consistency. Fixed the
# default in `eval/runners/benchmark_runner.py` to `"rigid"` (mirroring
# 49-01-SUMMARY.md's own precedent: "only 'rigid' dispatches correctly
# upstream") and use `cpd_penalty="rigid"` explicitly below instead of the
# plan's literal `1.0`.

BENCHMARK_PARAMS: dict = {
    # LabelTransferStage.REQUIRED_PARAMS
    "k_neighbours": 5,
    "dist_metric": "euclidean",
    "smoothing": 0.0,
    "threshold": 0.0,
    # AlignmentStage.REQUIRED_PARAMS (minus cpd_penalty, which
    # LabelTransferBenchmark.compare_methods defaults internally — supplied
    # explicitly here for clarity, see DEVIATION note above)
    "window_size": 10,
    "step": 1,
    "dtw_dist_fn": "euclidean",
    "n_breakpoints": 5,
    "cpd_penalty": "rigid",
}


def _build_benchmark_config(checkpoint_factory) -> EvalConfig:
    """Build an EvalConfig wired to both learned-method smoke checkpoints."""
    return EvalConfig(
        data_path="unused",
        pointnet2_checkpoint_path=checkpoint_factory("pointnet2"),
        egnn_checkpoint_path=checkpoint_factory("egnn"),
    )


def _build_holdout_pair(config: EvalConfig, seeds):
    """Build a multi-frame (source, target) pair from held-out synthetic seeds.

    Mirrors 49-RESEARCH.md Pattern 3: merges DataFactory.generate_training_set
    triples into one multi-frame dict keyed by seed INDEX (not seed value) for
    LabelTransferStage's contiguous sequential pairing.
    """
    triples = DataFactory(config).generate_training_set(seeds, n_classes=N_CLASSES)
    source = {i: t.source_cloud for i, t in enumerate(triples)}
    target = {i: t.target_cloud for i, t in enumerate(triples)}
    return source, target


class TestCompareMethods:
    """D-01 / D-02.1: F1/knn_consistency comparison across all four methods."""

    def test_all_four_methods_produce_results(self, benchmark_smoke_checkpoint) -> None:
        config = _build_benchmark_config(benchmark_smoke_checkpoint)
        source, target = _build_holdout_pair(config, range(0, 4))

        benchmark = LabelTransferBenchmark(config)
        report = benchmark.compare_methods(
            source, target, dict(BENCHMARK_PARAMS),
            dataset_source="synthetic_holdout", has_ground_truth=True,
        )

        assert len(report.results) == 4
        assert {r.method for r in report.results} == set(LabelTransferStage.VALID_METHODS)
        for r in report.results:
            if r.error is None:
                assert isinstance(r.f1_score, float)
                assert isinstance(r.knn_consistency, float)
                assert r.n_pairs > 0


class TestLatency:
    """D-02.3: latency_seconds is populated and strictly > 0 for every successful method."""

    def test_latency_positive_for_successful_methods(self, benchmark_smoke_checkpoint) -> None:
        config = _build_benchmark_config(benchmark_smoke_checkpoint)
        source, target = _build_holdout_pair(config, range(0, 4))

        benchmark = LabelTransferBenchmark(config)
        report = benchmark.compare_methods(
            source, target, dict(BENCHMARK_PARAMS),
            dataset_source="synthetic_holdout", has_ground_truth=True,
        )

        assert report.results  # non-empty
        for r in report.results:
            if r.error is None:
                assert r.latency_seconds is not None
                assert r.latency_seconds > 0.0


class TestRealDataDimension:
    """D-02.2 / Pitfall 1: real-data (Shah CSV fixture) dimension never computes F1."""

    def test_real_data_f1_none_no_crash(self, benchmark_smoke_checkpoint, tmp_path) -> None:
        config = _build_benchmark_config(benchmark_smoke_checkpoint)
        csv_path = tmp_path / "shah_fixture.csv"
        write_shah_fixture_csv(csv_path, n_points=40, n_classes=N_CLASSES, seed=0)
        shah_config = config.model_copy(update={"data_path": str(csv_path), "data_format": "csv"})
        real = DataFactory(shah_config).load_real()

        benchmark = LabelTransferBenchmark(shah_config)
        report = benchmark.compare_methods(
            real, real, dict(BENCHMARK_PARAMS),
            dataset_source="real_shah_qualitative", has_ground_truth=False,
        )

        assert len(report.results) == 4
        for r in report.results:
            assert r.f1_score is None
            if r.error is None:
                assert r.knn_consistency is not None


class TestLeakageGuard:
    """D-02.4: run_leakage_guard(held_out_seeds=<explicit range>) returns a valid BenchmarkReport."""

    def test_held_out_seeds_leakage_guard_returns_four_results(self, benchmark_smoke_checkpoint) -> None:
        config = _build_benchmark_config(benchmark_smoke_checkpoint)
        # split_seeds guarantees train_seeds/held_out_seeds are disjoint BY CONSTRUCTION
        # (Phase 46 D-03). Whether held_out_seeds is also disjoint from whatever seeds a
        # REAL trained checkpoint used is caller-verified, NOT enforced by
        # LabelTransferBenchmark (Assumption A2) — this smoke checkpoint was never
        # trained on any seed, so any range is trivially "held out" for this test.
        train_seeds, held_out_seeds = split_seeds(n_train=6, n_val=4, base_seed=0)

        benchmark = LabelTransferBenchmark(config)
        report = benchmark.run_leakage_guard(held_out_seeds, dict(BENCHMARK_PARAMS), n_classes=N_CLASSES)

        assert len(report.results) == 4


class TestErrorHandling:
    """D-03 / Pitfall 1/3: one method's failure does not crash the whole comparison."""

    def test_error_handling_missing_checkpoint_records_error(self, benchmark_smoke_checkpoint) -> None:
        # pointnet2_checkpoint_path intentionally omitted -> LabelTransferStage raises
        # ValueError for pointnet2; knn_voting has no checkpoint dependency.
        config = EvalConfig(data_path="unused")
        source, target = _build_holdout_pair(config, range(0, 4))

        benchmark = LabelTransferBenchmark(config, methods=("knn_voting", "pointnet2"))
        report = benchmark.compare_methods(
            source, target, dict(BENCHMARK_PARAMS),
            dataset_source="synthetic_holdout", has_ground_truth=True,
        )

        results_by_method = {r.method: r for r in report.results}
        pointnet2_result = results_by_method["pointnet2"]
        assert isinstance(pointnet2_result.error, str)
        assert pointnet2_result.f1_score is None
        assert pointnet2_result.knn_consistency is None
        assert pointnet2_result.latency_seconds is None

        knn_result = results_by_method["knn_voting"]
        assert knn_result.error is None


class TestSaveReportRoundTrip:
    """save_report writes benchmark_report.json and round-trips via json.load."""

    def test_save_report_round_trip(self, benchmark_smoke_checkpoint, tmp_path) -> None:
        config = _build_benchmark_config(benchmark_smoke_checkpoint)
        source, target = _build_holdout_pair(config, range(0, 3))

        benchmark = LabelTransferBenchmark(config)
        report = benchmark.compare_methods(
            source, target, dict(BENCHMARK_PARAMS),
            dataset_source="synthetic_holdout", has_ground_truth=True,
        )

        out_path = benchmark.save_report(report, tmp_path)

        assert out_path == tmp_path / "benchmark_report.json"
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert isinstance(data["results"], list)
        assert len(data["results"]) == len(report.results)
