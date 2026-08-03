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
Source frame = ``source[source_sorted_keys[0]]`` (source's first frame), target
frame = ``target[target_sorted_keys[-1]]`` (target's last frame).  This gives
maximum temporal span across two distinct trajectories — Phase 30 cross-trajectory
interpretation of Open Q3.
"""

import json
from pathlib import Path
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/metrics.py:53-67, eval/types.py:48-53.
from zreg.core.dataset import zRegPointCloud

import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.metrics import MetricsEngine
from eval.stages import AlignmentStage, LabelTransferStage
from eval.tracking import export_trajectory
from eval.types import AlignResult, EvalReport, LabelResult, StageMetrics
from eval.viz import plot_metrics, plot_trajectory

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
        self.factory: DataFactory | None = None  # WR-02: initialised in run(), guarded in _run_single

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
        **save_plots branch (D-09, D-10, D-12, FRAME-08, Phase 25):**
        When ``config.save_plots=True``, ``run()`` calls ``plot_trajectory``
        and ``plot_metrics`` from ``eval.viz`` after building a preliminary
        ``EvalReport``.  Because ``EvalReport`` is frozen (Pitfall 6), the
        final report is constructed via
        ``preliminary_report.model_copy(update={"plot_paths": plot_paths})``
        rather than re-constructing from scratch.

        Up to three files are produced when ``save_plots=True``:

        - ``alignment_trajectory.pdf/.png`` — written by ``plot_trajectory``
          only when ``result["align"] is not None``.
        - ``label_trajectory.pdf/.png`` — written by ``plot_trajectory``
          only when ``result["label"] is not None``.
        - ``metrics_summary.pdf`` — always written when ``save_plots=True``.

        All files are written to ``config.output_dir`` alongside
        ``eval_report.json``.  ``save_report`` is called AFTER ``plot_paths``
        is finalised so the persisted JSON reflects the final ``plot_paths``
        list.

        ``D-12``: ``Path(config.output_dir).mkdir(...)`` is the first line of
        ``run()`` — the viz parent directory is guaranteed to exist when the
        save_plots branch executes.
        """
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)  # D-12

        self.factory = DataFactory(self.config)
        source = self.factory.load_real()
        if self.config.pipeline_mode == "paired":
            target = self.factory.load_target()
        else:  # pipeline_mode == "synthetic" — Phase 31 MODE-02
            target = self.factory.generate_target(source, self.config.transform_spec)

        result = self._run_single(source, target, self.params)

        agg = self.engine.aggregate([result["metrics"]])
        # per_dataset expects dict[str, dict[str, float]] — inner dict must be
        # flat metric-name → float mapping.  Extract mean values from agg for
        # the single-dataset case (aggregate returns {metric: {mean, std, min, max}}).
        per_dataset_flat: dict[str, float] = {
            metric: stats["mean"] for metric, stats in agg.items()
        }

        # Step 1: Build a preliminary report (plot_paths=[], trajectory_paths=[] populated below).
        preliminary_report = EvalReport(
            params=dict(self.params),
            metrics=result["metrics"],
            aggregated_metrics=agg,
            per_dataset={"dataset": per_dataset_flat},
            plot_paths=[],
            trajectory_paths=[],
            sanity_flags=result["sanity_flags"],
        )

        # Step 2: Conditionally render plots and collect paths.
        output_dir_path = Path(self.config.output_dir)
        plot_paths: list[str] = []
        if self.config.save_plots:
            # D-10 (Phase 25): plot_trajectory handles both stages internally;
            # guard is inside plot_trajectory — returns [] when both are None.
            plot_paths.extend(
                plot_trajectory(
                    result["align"],
                    result["label"],
                    source,
                    self.config.label_names,
                    output_dir_path,
                    target=target,
                )
            )
            summary_path = output_dir_path / "metrics_summary.pdf"
            plot_paths.extend(plot_metrics(preliminary_report, summary_path))

        # Step 2b: Export trajectories unconditionally (D-13 / EXT-01).
        trajectory_paths = export_trajectory(result, source, self.config, output_dir_path, target=target)

        # Step 3: Build the final frozen report with both plot_paths and trajectory_paths.
        # EvalReport is frozen (Pitfall 6) — must use model_copy to update.
        # Single model_copy call handles both fields (D-14).
        report = preliminary_report.model_copy(
            update={"plot_paths": plot_paths, "trajectory_paths": trajectory_paths}
        )

        # Step 4: Persist JSON and return.
        self.save_report(report, self.config.output_dir)
        return report

    def _run_single(
        self,
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute stages and compute metrics for a single source/target pair.

        Implements the argument assembly recipe from 21-RESEARCH.md §_run_single,
        updated in Phase 30 to accept distinct source and target datasets.
        Stage execution is conditional on ``config.run_alignment`` /
        ``config.run_label_transfer``; skipped stages are zero-filled per D-04.

        Parameters
        ----------
        source : dict[int, zRegPointCloud]
            Source trajectory as returned by ``DataFactory.load_real()``, keyed
            by integer frame index.
        target : dict[int, zRegPointCloud]
            Target trajectory as returned by ``DataFactory.load_target()``, keyed
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

        **Source/target frame selection:** source_frame = ``source[source_sorted_keys[0]]``
        (source's first frame), target_frame = ``target[target_sorted_keys[-1]]``
        (target's last frame) — maximum temporal span across two distinct trajectories
        (Phase 30 cross-trajectory interpretation of Open Q3).

        **Zero-fill (D-04):** Skipped-stage metrics are zeroed via
        ``model_copy(update={...})`` because ``StageMetrics`` is a frozen
        pydantic model (RESEARCH Pitfall 6).
        """
        # WR-02: guard against _run_single being called before run() initialises factory
        if self.factory is None:
            raise RuntimeError("_run_single called before run() — factory not initialised")

        align_result = None
        label_result = None

        # --- Stage execution ---
        if self.config.run_alignment:
            align_result = AlignmentStage(self.config).run(source, target, params)
            stage_input = align_result.aligned_cloud
        else:
            stage_input = source  # D-05 — use source as fallback

        if self.config.run_label_transfer:
            label_result = LabelTransferStage(self.config).run(stage_input, target, params, align_result=align_result)

        # --- Argument assembly for compute_stage_metrics (8 positional args) ---
        # Pitfall 4 — local source_pos/target_pos avoid shadowing the source/target parameters
        source_sorted_keys = sorted(source.keys())
        target_sorted_keys = sorted(target.keys())
        source_frame = source[source_sorted_keys[0]]
        target_frame = target[target_sorted_keys[-1]]
        source_pos = source_frame["pos"]   # shape (N, 3)
        target_pos = target_frame["pos"]   # shape (M, 3)

        warp_path = align_result.warp_path if align_result else []
        transforms = []  # AlignResult has no transform objects — temporal_stability([]) returns 0.0

        # Ground truth — branches on pipeline_mode (Phase 31 MODE-03)
        if self.config.pipeline_mode == "synthetic":
            gt = self.factory.get_synthetic_ground_truth()
        else:
            gt = self.factory.get_ground_truth(source)  # {frame_key: id_tensor}
        y_true = gt[source_sorted_keys[-1]]
        if label_result is not None:
            # Label keys are TARGET frames per Plan 30-01 LabelTransferStage contract
            # Use last key actually present in transferred_labels (= last paired target
            # frame) — guards against KeyError when |source| < |target| (CR-01).
            transferred_keys = sorted(label_result.transferred_labels.keys())
            knn_target_key = transferred_keys[-1]
            y_pred = label_result.transferred_labels[knn_target_key]
        else:
            knn_target_key = target_sorted_keys[-1]
            y_pred = torch.zeros_like(y_true)  # zero-fill D-04

        # WR-01: truncate to min length when source and target have different point counts.
        # Phase 56 D-03 (GT-02) re-scoping: this positional-truncation fallback now only
        # matters for pipeline_mode="paired" heterogeneous real-data cases (Phase 32
        # HETERO-01), where no per-point correspondence can be computed between two
        # independently-loaded real datasets. For pipeline_mode="synthetic",
        # DataFactory.get_synthetic_ground_truth() already gathers y_true by the tracked
        # drop_points/sample_new_points correspondence index (Phase 56 Task 1), so y_true
        # and y_pred already match length by the time this line runs — this block is a
        # defensive no-op in that case, not the correctness mechanism. compute_f1 validates
        # shape equality strictly regardless of pipeline_mode.
        if y_true.shape[0] != y_pred.shape[0]:
            min_len = min(y_true.shape[0], y_pred.shape[0])
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        # KNN consistency needs points and labels from the same target frame.
        # When |source| < |target|, the last transferred frame differs from the
        # global last target frame, causing a cell-count mismatch.
        points_for_knn = target[knn_target_key]["pos"]
        labels_for_knn = (
            label_result.transferred_labels[knn_target_key]
            if label_result is not None
            else y_pred
        )

        metrics = self.engine.compute_stage_metrics(
            source_pos, target_pos, warp_path, transforms,
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
