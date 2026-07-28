"""LabelTransferBenchmark: multi-method comparison runner for FRAME-06 methods (Phase 49).

This module implements ``LabelTransferBenchmark``, which runs all four
``LabelTransferStage`` methods (``knn_voting``, ``cpd_weighted``, ``pointnet2``,
``egnn``) against an identical raw source/target pair, computes
``compute_f1``/``knn_consistency``/latency per method, and records per-method
failures gracefully (``error`` field) instead of crashing the whole comparison
run.  Additive to ``EvaluationRunner`` (unchanged, single-method per-trial
orchestration) — this class is the distinct "compare N methods on the same
data" use case (49-RESEARCH.md Pattern 1/6).

Key design decisions implemented here:

- **Pattern 2 / Pitfall 3 (raw-input fairness):** All four methods receive the
  SAME raw (non-CPD-aligned) ``source``/``target`` pair. ``AlignmentStage`` is
  run AT MOST ONCE per ``compare_methods()`` call, only when ``"cpd_weighted"``
  is among ``self.methods``, using
  ``config.model_copy(update={"alignment_method": "cpd"})`` — never trusting
  the caller's base ``config.alignment_method`` value. ``cpd_weighted``'s
  ``align_result.estep_results[tk].pmat`` is index-space and independent of
  whether the ``source``/``target`` positions passed to
  ``LabelTransferStage.run()`` are raw or CPD-aligned (49-01's Wave-0
  empirical proof).
- **Pitfall 1 (real-data F1):** F1 is NEVER computed for the real-data
  dimension (``has_ground_truth=False``) — only ``knn_consistency``, since
  real Kobitski/Shah ``pc["label"]`` is not the learned models' training
  vocabulary.
- **D-03 (graceful degradation):** Each method's ``ValueError``/``RuntimeError``
  is caught and recorded as an ``error`` field without crashing the rest of
  the comparison.
- **Pattern 3 / Assumption A2 (held-out seeds):** ``held_out_seeds`` is an
  explicit required parameter to ``run_leakage_guard`` — never hardcoded
  inside the class. Checkpoints carry no seed-provenance metadata.

No point-count-regime guard, timeout, or subsampling is added here (D-03 —
scope-expanding mitigations are explicitly out of scope for this phase).
"""

import json
import time
from pathlib import Path
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/metrics.py:53-67, eval/types.py:48-53.
from zreg.evaluation import compute_f1, knn_consistency

import torch  # noqa: F401 — ensures consistent import order for downstream callers

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.stages import AlignmentStage, LabelTransferStage
from eval.types import BenchmarkReport, MethodBenchmarkResult

__all__ = ["LabelTransferBenchmark"]


class LabelTransferBenchmark:
    """Orchestrates a multi-method label-transfer comparison run (Phase 49).

    Runs every method in ``self.methods`` against an identical raw
    source/target pair, computing F1 (when ground truth is available),
    knn_consistency, amortized per-frame-pair latency, and recording
    per-method failures gracefully as an ``error`` string.

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration. Stored as ``self.config``.
        ``config.pointnet2_checkpoint_path``/``config.egnn_checkpoint_path``
        must be set when ``"pointnet2"``/``"egnn"`` are in ``methods`` and the
        caller expects those methods to succeed (otherwise they fail
        gracefully with a recorded ``error``, per D-03).
    methods : tuple[str, ...], optional
        Methods to compare. Defaults to
        ``LabelTransferStage.VALID_METHODS`` (all four: ``knn_voting``,
        ``cpd_weighted``, ``pointnet2``, ``egnn``).

    Attributes
    ----------
    config : EvalConfig
        The evaluation configuration passed at construction.
    methods : tuple[str, ...]
        The methods this benchmark compares.

    Notes
    -----
    This class never calls ``torch.load`` directly — ``LabelTransferStage``
    (unchanged, Phase 48) owns checkpoint deserialization with
    ``weights_only=True``.
    """

    def __init__(
        self,
        config: EvalConfig,
        methods: tuple[str, ...] = LabelTransferStage.VALID_METHODS,
    ) -> None:
        """Store the evaluation configuration and the methods to compare.

        Parameters
        ----------
        config : EvalConfig
            Validated evaluation configuration.
        methods : tuple[str, ...], optional
            Methods to compare. Defaults to ``LabelTransferStage.VALID_METHODS``.
        """
        self.config = config
        self.methods = methods

    def compare_methods(
        self,
        source: dict[int, Any],
        target: dict[int, Any],
        params: dict[str, Any],
        dataset_source: str,
        has_ground_truth: bool = True,
    ) -> BenchmarkReport:
        """Run every method in ``self.methods`` against the same raw source/target pair.

        Parameters
        ----------
        source : dict[int, zRegPointCloud]
            Source trajectory keyed by integer frame index. The SAME raw
            (non-CPD-aligned) dict is passed to every method's
            ``LabelTransferStage.run()`` call (Pattern 2) — including
            ``cpd_weighted``, which only needs ``align_result.estep_results``
            for its posterior, not an ``aligned_cloud`` substitution.
        target : dict[int, zRegPointCloud]
            Target trajectory keyed by integer frame index.
        params : dict[str, Any]
            Flat hyperparameter dict. Must contain both
            ``LabelTransferStage.REQUIRED_PARAMS`` (``k_neighbours``,
            ``dist_metric``, ``smoothing``, ``threshold``) AND, when
            ``"cpd_weighted"`` is among ``self.methods``,
            ``AlignmentStage.REQUIRED_PARAMS`` minus ``cpd_penalty``
            (``window_size``, ``step``, ``dtw_dist_fn``, ``n_breakpoints``) —
            the caller is responsible for supplying these so the internal
            ``AlignmentStage`` call has everything it needs.  ``cpd_penalty``
            (a CPD-type string — ``"rigid"``, ``"affine"``, or ``"nonrigid"``,
            NOT a numeric magnitude despite the name) is read from ``params``
            if present, defaulting to ``"rigid"`` otherwise — see Notes.
        dataset_source : str
            Identifier for the dataset this run was computed against, e.g.
            ``"synthetic_holdout"`` or ``"real_shah_qualitative"``. Stamped
            onto every ``MethodBenchmarkResult``.
        has_ground_truth : bool, optional
            When ``True`` (default), F1 is computed against
            ``target[tk]["label"]`` for every successful method. When
            ``False`` (real-data dimension, Pitfall 1), F1 is NEVER computed
            — only ``knn_consistency``.

        Returns
        -------
        BenchmarkReport
            ``results`` has exactly ``len(self.methods)`` entries, one per
            method, each carrying ``dataset_source``. A method whose
            ``LabelTransferStage.run()`` raises ``ValueError``/``RuntimeError``
            yields a result with ``error`` populated and all metric fields
            ``None`` (D-03) — the loop continues for the remaining methods.

        Notes
        -----
        **AlignmentStage runs at most once (Pitfall 3):** When
        ``"cpd_weighted"`` is in ``self.methods``, ``AlignmentStage.run`` is
        called exactly once, BEFORE the method loop, using
        ``self.config.model_copy(update={"alignment_method": "cpd"})`` —
        never the caller's base ``config.alignment_method``. The other three
        methods receive ``align_result=None``.
        """
        align_result = None
        if "cpd_weighted" in self.methods:
            # AlignmentStage.VALID_CPD = (None, "rigid", "affine", "nonrigid") — the
            # default here MUST be one of these string CPD-type values, never a bare
            # penalty magnitude (a float default would make validate_params raise
            # unconditionally). "rigid" mirrors 49-01's own Wave-0 precedent (Pitfall 3
            # — "only 'rigid' dispatches correctly upstream").
            align_params = {**params, "cpd_penalty": params.get("cpd_penalty", "rigid")}
            align_result = AlignmentStage(
                self.config.model_copy(update={"alignment_method": "cpd"})
            ).run(source, target, align_params)

        results: list[MethodBenchmarkResult] = []
        for method in self.methods:
            t0 = time.perf_counter()
            try:
                lr = LabelTransferStage(self.config).run(
                    source,
                    target,
                    {**params, "method": method},
                    align_result=align_result if method == "cpd_weighted" else None,
                )
                elapsed = time.perf_counter() - t0
                keys = sorted(lr.transferred_labels)
                n_pairs = len(keys)

                f1 = None
                if has_ground_truth:
                    y_pred = torch.cat([lr.transferred_labels[k] for k in keys])
                    y_true = torch.cat([target[k]["label"] for k in keys])
                    f1 = compute_f1(y_true, y_pred)

                tk = keys[-1]
                knn = knn_consistency(target[tk]["pos"], lr.transferred_labels[tk])

                results.append(
                    MethodBenchmarkResult(
                        method=method,
                        dataset_source=dataset_source,
                        f1_score=f1,
                        knn_consistency=knn,
                        latency_seconds=elapsed / max(n_pairs, 1),
                        n_pairs=n_pairs,
                        error=None,
                    )
                )
            except (ValueError, RuntimeError) as e:
                results.append(
                    MethodBenchmarkResult(
                        method=method,
                        dataset_source=dataset_source,
                        f1_score=None,
                        knn_consistency=None,
                        latency_seconds=None,
                        error=str(e),
                    )
                )

        return BenchmarkReport(params=dict(params), results=results)

    def run_leakage_guard(
        self,
        held_out_seeds,
        params: dict[str, Any],
        n_classes: int,
        dataset_source: str = "synthetic_holdout",
    ) -> BenchmarkReport:
        """Compare methods against held-out synthetic seeds never seen during training.

        Parameters
        ----------
        held_out_seeds : Iterable[int]
            Seeds to build the evaluation triples from. This is a REQUIRED
            caller-supplied argument — the caller is responsible for choosing
            a range disjoint from whatever seeds the checkpoint being
            evaluated was actually trained on. Checkpoints produced by
            ``train_label_transfer.py`` carry no seed-provenance metadata
            (49-RESEARCH.md Assumption A2 / Pattern 3), so this class CANNOT
            verify disjointness from the checkpoint file alone. Never
            hardcode a "safe" range inside this method.
        params : dict[str, Any]
            Forwarded to :meth:`compare_methods` unchanged.
        n_classes : int
            Forwarded to ``DataFactory.generate_training_set``.
        dataset_source : str, optional
            Identifier stamped onto every result. Default
            ``"synthetic_holdout"``.

        Returns
        -------
        BenchmarkReport
            Result of :meth:`compare_methods` called with
            ``has_ground_truth=True`` over the held-out triples, merged into
            a single multi-frame source/target pair keyed by seed index (not
            seed value) for contiguous ``LabelTransferStage`` pairing
            (49-RESEARCH.md Pattern 3).
        """
        triples = DataFactory(self.config).generate_training_set(held_out_seeds, n_classes=n_classes)
        source = {i: t.source_cloud for i, t in enumerate(triples)}
        target = {i: t.target_cloud for i, t in enumerate(triples)}
        return self.compare_methods(source, target, params, dataset_source, has_ground_truth=True)

    def save_report(self, report: BenchmarkReport, output_dir: str | Path) -> Path:
        """Write the benchmark report to ``benchmark_report.json`` in ``output_dir``.

        Mirrors ``EvaluationRunner.save_report`` exactly (D-11 fixed
        filename convention, ``model_dump()`` + ``json.dump()`` — the
        tensor-serializing alternative pydantic method is deliberately
        avoided here, T-49-04).

        Parameters
        ----------
        report : BenchmarkReport
            Populated benchmark report to serialise. ``params`` values must
            stay JSON-primitive (int/float/str/bool/None).
        output_dir : str or Path
            Directory where ``benchmark_report.json`` is written. Created if
            it does not already exist.

        Returns
        -------
        Path
            ``Path(output_dir) / "benchmark_report.json"``.
        """
        out_path = Path(output_dir) / "benchmark_report.json"
        data = report.model_dump()  # D-11 — never the tensor-serializing alternative
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        return out_path
