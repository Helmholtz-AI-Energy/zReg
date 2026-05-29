"""EvaluationRunner: orchestration layer for the FRAME-07 evaluation framework.

This module implements ``EvaluationRunner``, which composes ``DataFactory``,
``AlignmentStage``, ``LabelTransferStage``, ``MetricsEngine``, and ``EvalReport``
into a single ``run()`` call that produces a complete evaluation report and writes
``eval_report.json`` to ``config.output_dir``.

Key design decisions implemented here:

- **D-01** Flat params dict passed to both stages.  Alignment keys
  (``window_size``, ``step``, ``cpd_penalty``, ``dtw_dist_fn``,
  ``n_breakpoints``) and label-transfer keys (``k_neighbours``,
  ``dist_metric``, ``smoothing``, ``threshold``) are disjoint.
- **D-04** Zero-fill (0.0) for skipped-stage metrics.  ``StageMetrics`` stays
  all-float; callers zero-weight skipped metrics via ``config.metric_weights``.
- **D-06** If both ``run_alignment=False`` AND ``run_label_transfer=False``,
  ``__init__`` raises ``ValueError("At least one stage must be enabled")``
  immediately (fail-fast).
- **D-10 / D-11** ``save_report`` writes ``Path(output_dir) / "eval_report.json"``
  using ``report.model_dump()`` + ``json.dump(data, f, indent=2)``.  Never uses
  ``model_dump_json()`` (RESEARCH Pitfall 2).
- **D-12** ``run()`` calls ``Path(config.output_dir).mkdir(parents=True,
  exist_ok=True)`` as its first executable line — before any stage execution or
  file writes.

Notes
-----
**AlignResult has no ``transforms`` field.**
``compute_stage_metrics`` is called with ``transforms=[]``; the underlying
``temporal_stability([])`` function returns 0.0 (RESEARCH Pitfall 3).

**``params`` must contain only JSON-primitive values** (int, float, str, bool,
None) for ``save_report`` to succeed.  If any value is a ``torch.Tensor``,
``json.dump(report.model_dump(), f)`` will raise ``TypeError`` (RESEARCH
Pitfall 2).

**Frame selection for ``compute_stage_metrics``.**
Source frame = ``sorted_keys[0]`` (first), target frame = ``sorted_keys[-1]``
(last).  This gives the maximum temporal span (RESEARCH Assumption A2, Open Q3).
"""

import json
from pathlib import Path
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/metrics.py:53-67, eval/types.py:48-53.
from zreg.dataset import zRegPointCloud

import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.metrics import MetricsEngine
from eval.stages import AlignmentStage, LabelTransferStage
from eval.types import AlignResult, EvalReport, LabelResult, StageMetrics
from eval.viz import plot_metrics_summary, plot_point_cloud

__all__ = ["EvaluationRunner"]


class EvaluationRunner:
    """Orchestrates DataFactory → AlignmentStage → LabelTransferStage →
    MetricsEngine → EvalReport (FRAME-07).

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration.  Stored as ``self.config``.
        Controls which stages run (``run_alignment``, ``run_label_transfer``),
        where output is written (``output_dir``), and whether plots are saved
        (``save_plots``).
    params : dict[str, Any]
        Flat hyperparameter dict passed to both stages (D-01).  Keys are
        disjoint across stages:

        - AlignmentStage keys: ``window_size``, ``step``, ``cpd_penalty``,
          ``dtw_dist_fn``, ``n_breakpoints``.
        - LabelTransferStage keys: ``k_neighbours``, ``dist_metric``,
          ``smoothing``, ``threshold``.

        Stored as ``self.params`` (shallow copy, D-01 / Pitfall 7).

    Attributes
    ----------
    config : EvalConfig
        The evaluation configuration passed at construction.
    params : dict[str, Any]
        Shallow copy of the hyperparameter dict.
    engine : MetricsEngine
        Metrics engine constructed from ``config`` at construction time
        (store on ``__init__`` for test mockability, RESEARCH Open Q2).

    Raises
    ------
    ValueError
        If both ``config.run_alignment`` and ``config.run_label_transfer`` are
        ``False`` (D-06 fail-fast guard).  Raised before any attribute is stored.

    Notes
    -----
    **Per-trial pattern (D-02):** ``run()`` uses ``self.params`` with no override
    argument.  Phase 22 ``HyperparamOptimizer`` creates a new
    ``EvaluationRunner(config, trial_params)`` per trial — one runner = one
    fixed param set = one report.

    **D-03:** ``run()`` instantiates ``DataFactory(self.config)`` internally;
    callers provide only ``config`` and ``params``.
    """

    def __init__(self, config: EvalConfig, params: dict[str, Any]) -> None:
        """Construct the runner, validate stage flags, and store engine.

        Parameters
        ----------
        config : EvalConfig
            Validated evaluation configuration.
        params : dict[str, Any]
            Flat hyperparameter dict (D-01).  Shallow-copied on storage.

        Raises
        ------
        ValueError
            If both ``config.run_alignment`` and ``config.run_label_transfer``
            are ``False`` — nonsensical configuration (D-06).
        """
        if not config.run_alignment and not config.run_label_transfer:
            raise ValueError("At least one stage must be enabled")
        self.config = config
        self.params = dict(params)  # shallow copy — D-01 / Pitfall 7
        self.engine = MetricsEngine(config)

    def run(self) -> EvalReport:
        """Execute the full evaluation pipeline and return an EvalReport.

        The pipeline follows the FRAME-07 orchestration contract:

        1. ``Path(config.output_dir).mkdir(parents=True, exist_ok=True)`` — D-12
        2. Instantiate ``DataFactory`` and load real dataset.
        3. Call ``_run_single(dataset, self.params)`` to execute stages and
           compute metrics.
        4. Construct ``EvalReport`` from results.
        5. Call ``save_report(report, config.output_dir)`` to write JSON.
        6. Return the ``EvalReport``.

        Returns
        -------
        EvalReport
            Populated report with ``params``, ``metrics``, ``aggregated_metrics``,
            ``per_dataset``, ``plot_paths`` (populated when
            ``config.save_plots=True``), and ``sanity_flags``.

        Notes
        -----
        **save_plots branch (D-09, D-12, FRAME-08):**
        When ``config.save_plots=True``, ``run()`` calls ``plot_point_cloud``
        and ``plot_metrics_summary`` from ``eval.viz`` after building a
        preliminary ``EvalReport``.  Because ``EvalReport`` is frozen (Pitfall
        6), the final report is constructed via
        ``preliminary_report.model_copy(update={"plot_paths": plot_paths})``
        rather than re-constructing from scratch.

        Two PDFs are produced when ``save_plots=True``:

        - ``point_cloud.pdf`` — written only when ``run_alignment=True``
          (requires ``AlignResult`` for 3D scatter; skipped when
          ``run_alignment=False`` because no ``AlignResult`` is available).
        - ``metrics_summary.pdf`` — always written when ``save_plots=True``,
          regardless of which stages ran.

        Both files are written to ``config.output_dir`` alongside
        ``eval_report.json``.  ``save_report`` is called AFTER ``plot_paths``
        is finalised so the persisted JSON reflects the final ``plot_paths``
        list.

        ``D-12``: ``Path(config.output_dir).mkdir(...)`` is the first line of
        ``run()`` — the viz parent directory is guaranteed to exist when the
        save_plots branch executes.
        """
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)  # D-12

        self.factory = DataFactory(self.config)
        dataset = self.factory.load_real()

        result = self._run_single(dataset, self.params)

        agg = self.engine.aggregate([result["metrics"]])
        # per_dataset expects dict[str, dict[str, float]] — inner dict must be
        # flat metric-name → float mapping.  Extract mean values from agg for
        # the single-dataset case (aggregate returns {metric: {mean, std, min, max}}).
        per_dataset_flat: dict[str, float] = {
            metric: stats["mean"] for metric, stats in agg.items()
        }

        # Step 1: Build a preliminary report (plot_paths=[]; populated below).
        preliminary_report = EvalReport(
            params=dict(self.params),
            metrics=result["metrics"],
            aggregated_metrics=agg,
            per_dataset={"dataset": per_dataset_flat},
            plot_paths=[],
            sanity_flags=result["sanity_flags"],
        )

        # Step 2: Conditionally render plots and collect paths.
        output_dir_path = Path(self.config.output_dir)
        plot_paths: list[str] = []
        if self.config.save_plots:
            if result["align"] is not None:
                pc_path = output_dir_path / "point_cloud.pdf"
                plot_point_cloud(result["align"], pc_path)
                plot_paths.append(str(pc_path))
            summary_path = output_dir_path / "metrics_summary.pdf"
            plot_metrics_summary(preliminary_report, summary_path)
            plot_paths.append(str(summary_path))

        # Step 3: Build the final frozen report with the resolved plot_paths list.
        # EvalReport is frozen (Pitfall 6) — must use model_copy to update.
        report = preliminary_report.model_copy(update={"plot_paths": plot_paths})

        # Step 4: Persist JSON and return.
        self.save_report(report, self.config.output_dir)
        return report

    def _run_single(
        self,
        dataset: dict[int, zRegPointCloud],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute stages and compute metrics for a single dataset.

        Implements the argument assembly recipe from 21-RESEARCH.md §_run_single.
        Stage execution is conditional on ``config.run_alignment`` /
        ``config.run_label_transfer``; skipped stages are zero-filled per D-04.

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Full trajectory as returned by ``DataFactory.load_real()``, keyed
            by integer frame index.
        params : dict[str, Any]
            Flat hyperparameter dict.  Passed verbatim to both stages; each
            stage's ``validate_params`` ignores keys it does not own.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``"metrics"``: ``StageMetrics`` instance (raw + normalized).
            - ``"sanity_flags"``: ``list[str]`` from
              ``MetricsEngine.sanity_check``.
            - ``"align"``: ``AlignResult | None``.
            - ``"label"``: ``LabelResult | None``.

        Notes
        -----
        **``transforms=[]``:** ``AlignResult`` has no transforms field.
        ``temporal_stability([])`` returns 0.0 — intentional (RESEARCH Pitfall 3).

        **Source/target frame selection:** source = ``sorted_keys[0]``,
        target = ``sorted_keys[-1]`` — maximum temporal span (RESEARCH A2).

        **Zero-fill (D-04):** Skipped-stage metrics are zeroed via
        ``model_copy(update={...})`` because ``StageMetrics`` is a frozen
        pydantic model (RESEARCH Pitfall 6).
        """
        align_result = None
        label_result = None

        # --- Stage execution ---
        if self.config.run_alignment:
            align_result = AlignmentStage(self.config).run(dataset, params)
            stage_input = align_result.aligned_cloud
        else:
            stage_input = dataset  # D-05

        if self.config.run_label_transfer:
            label_result = LabelTransferStage(self.config).run(stage_input, params)

        # --- Argument assembly for compute_stage_metrics (8 positional args) ---
        sorted_keys = sorted(dataset.keys())
        source_frame = dataset[sorted_keys[0]]
        target_frame = dataset[sorted_keys[-1]]
        source = source_frame["pos"]   # shape (N, 3)
        target = target_frame["pos"]   # shape (M, 3)

        warp_path = align_result.warp_path if align_result else []
        transforms = []  # AlignResult has no transform objects — temporal_stability([]) returns 0.0

        # Ground truth for target frame
        gt = self.factory.get_ground_truth(dataset)  # {frame_key: id_tensor}
        y_true = gt[sorted_keys[-1]]
        if label_result is not None:
            y_pred = label_result.transferred_labels[sorted_keys[-1]]
        else:
            y_pred = torch.zeros_like(y_true)  # zero-fill D-04

        points_for_knn = target
        labels_for_knn = y_pred

        metrics = self.engine.compute_stage_metrics(
            source, target, warp_path, transforms,
            y_true, y_pred, points_for_knn, labels_for_knn,
            k_neighbours=params.get("k_neighbours", 10),
        )

        # Zero-fill for skipped stages (D-04) — frozen model requires model_copy (Pitfall 6)
        if not self.config.run_alignment:
            metrics = metrics.model_copy(update={
                "chamfer_distance": 0.0,
                "hausdorff_distance": 0.0,
                "path_smoothness": 0.0,
                "temporal_stability": 0.0,
            })
            metrics = metrics.model_copy(update={"normalized": self.engine.normalize(metrics)})
        if not self.config.run_label_transfer:
            metrics = metrics.model_copy(update={"f1_score": 0.0, "knn_consistency": 0.0})
            metrics = metrics.model_copy(update={"normalized": self.engine.normalize(metrics)})

        flags = self.engine.sanity_check(align=align_result, label=label_result, metrics=metrics)
        return {"metrics": metrics, "sanity_flags": flags, "align": align_result, "label": label_result}

    def save_report(
        self,
        report: EvalReport,
        output_dir: str | Path,
    ) -> Path:
        """Write the evaluation report to ``eval_report.json`` in ``output_dir``.

        Uses ``report.model_dump()`` + ``json.dump()`` (D-10).  Never uses
        ``model_dump_json()`` — that method raises ``PydanticSerializationError``
        for tensor-containing fields (RESEARCH Pitfall 2).

        Parameters
        ----------
        report : EvalReport
            Populated evaluation report to serialise.  Must have only
            JSON-primitive values in ``params`` (int, float, str, bool, None)
            or ``json.dump`` will raise ``TypeError``.
        output_dir : str or Path
            Directory where ``eval_report.json`` is written.  Must already
            exist (created by ``run()`` via D-12).

        Returns
        -------
        Path
            Absolute path of the written file:
            ``Path(output_dir) / "eval_report.json"`` (D-11 fixed filename).

        Notes
        -----
        ``indent=2`` is used for human-readable output.
        The filename is always ``eval_report.json`` — callers cannot override
        this (D-11).
        """
        out_path = Path(output_dir) / "eval_report.json"  # D-11 fixed filename
        data = report.model_dump()  # D-10 — never model_dump_json()
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        return out_path
