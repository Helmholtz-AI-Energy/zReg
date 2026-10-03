"""LabelTransferStage: thin color-transfer wrapper for FRAME-06 evaluation framework.

This module implements ``LabelTransferStage(PipelineStage)``, a thin orchestration
layer over ``zreg.color_transfer.transfer_colors()``.  No kNN or distance logic is
reimplemented here — all numerical computation delegates to the existing
``zreg.color_transfer.*`` package via KNN_VOTING or CPD_WEIGHTED (Phase 44).

Hyperparam mapping:

+---------------+-------------------------------------------------------+
| params key    | purpose                                               |
+===============+=======================================================+
| k_neighbours  | k kwarg passed to transfer_colors knn_voting          |
+---------------+-------------------------------------------------------+
| dist_metric   | validated non-empty str; no-op in Phase 20 (D-06)    |
+---------------+-------------------------------------------------------+
| smoothing     | validated float >= 0.0; no-op in Phase 20 (D-04)     |
+---------------+-------------------------------------------------------+
| threshold     | validated float >= 0.0; no-op in Phase 20 (D-05)     |
+---------------+-------------------------------------------------------+
| method        | 'knn_voting' (default), 'cpd_weighted', 'pointnet2', |
|               | or 'egnn'; defaults to config.label_transfer_method   |
|               | when absent (Phase 44; pointnet2/egnn added Phase 48) |
+---------------+-------------------------------------------------------+

Notes
-----
**Sequential pairing:**
For k in range(min(len(source), len(target))), transfer labels from
``source[source_keys[k]]`` to ``target[target_keys[k]]``.  No frame-0
pass-through (D-02 + Phase 30).

**Boolean guard pattern:**
Bool exclusion guards are applied to all 4 params per WR-01 pattern from
Phase 19.  ``isinstance(x, bool)`` must be tested before ``isinstance(x, int)``
because ``bool`` is a subclass of ``int`` in Python.

**CPD-weighted label transfer (Phase 44):**
``method='cpd_weighted'`` reuses ``zreg.color_transfer``'s existing
CPD-posterior-weighted-average math, fixing two call-site bugs so
``zreg.color_transfer`` itself never needs to change:

- **pmat orientation (D-04, Phase 59 NUM-04):** ``EstepResult.pmat`` from
  ``expectation_step()`` is shaped ``(n_aligned_source, n_target)``, while
  the underlying label-transfer helper requires ``(n_receiver, n_provider)``
  and row-normalises it.  The orientation is chosen from
  ``config.label_source`` (never inferred from shapes):

  * ``"source"`` (default): provider = aligned source, receiver = target.
    The stage builds a transposed copy of the E-step result (via the
    namedtuple's ``_replace``), so labels flow aligned source -> target.
  * ``"target"``: provider = target, receiver = aligned source.  The
    posterior is used as stored; each aligned-source point gets the
    posterior-weighted vote of the target's labels (target -> aligned
    source).

  Zero-row policy (Phase 62 RD-1..RD-3, shared with the library through
  ``zreg.label_transfer.repair_pmat_rows``): a receiver point whose
  posterior row sums to zero (or is non-finite), e.g. through float
  underflow far from every provider point, gets the label of its nearest
  provider point instead of a NaN-derived label.  The frame and count are
  logged once and recorded in ``LabelResult.flags``.  Non-finite positions
  in any paired frame are rejected with ``ValueError`` before any transfer
  (``_check_alignment``, 62-REVIEW WR-05); the ``-1`` (no label) sentinel the
  repair assigns to a non-finite receiver position is defence in depth only
  and is not reachable through ``run()``.  A frame raises
  ``ValueError`` when *every* receiver row is bad or when more than
  ``MAX_PMAT_FALLBACK_FRACTION`` (0.5) of its rows would fall back.
- **categorical one-hot/argmax (D-05):** a literal weighted average of
  raw class indices is meaningless, so source labels are one-hot encoded
  before the call and the resulting soft scores are discretized back via
  ``argmax(dim=1)`` after.

**Learned label transfer (Phase 48):**
``method='pointnet2'`` and ``method='egnn'`` load a Phase 47 checkpoint
(via ``self.config.pointnet2_checkpoint_path`` / ``self.config.egnn_checkpoint_path``,
D-02) once per ``run()`` call, before the per-frame loop (Pattern 3 — never
reloaded inside the loop), then run inference per frame pair. The joint
cloud fed to the model MUST byte-for-byte mirror
``train_label_transfer.py:train_step``'s encoding (source-then-target
concatenation, one-hot over the first ``n_classes`` feature dims, unknown-
flag bit at the last column for target rows) or predictions are silently
meaningless. See ``_load_learned_model`` for the checkpoint-loading
contract (three ``ValueError`` guards: missing path config, file not found,
``model_class`` mismatch).
"""

import logging
import warnings
from pathlib import Path
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/metrics.py:53-67, eval/types.py:48-53.
from zreg.label_transfer import transfer_labels as transfer_colors, LabelTransferMethod as ColorTransferMethod
from zreg.label_transfer import repair_pmat_rows
from zreg.core.dataset import zRegPointCloud
from zreg.evaluation import chamfer
from zreg.models import PointNet2LabelTransfer, EGNNLabelTransfer

import torch  # consistent import order for downstream callers (macOS-ARM zreg-before-torch rule)

from eval.config import EvalConfig
from eval.stages.base import PipelineStage
from eval.types import LabelResult, AlignResult

__all__ = ["LabelTransferStage"]

ALIGNMENT_WARN_THRESHOLD: float = 1.0
_log = logging.getLogger(__name__)

# Phase 48: model-class dispatch, single source of truth (mirrors
# train_label_transfer.py:80 — the training script this phase's checkpoints
# come from).
MODEL_REGISTRY = {"pointnet2": PointNet2LabelTransfer, "egnn": EGNNLabelTransfer}


class LabelTransferStage(PipelineStage):
    """Concrete label-transfer stage wrapping ``transfer_colors``.

    Runs without ``AlignmentStage`` present (FRAME-06 gate 1).  All live
    knobs come from the ``params`` dict passed to ``run()``; ``config`` is
    stored but currently unused inside the stage (reserved for Phase 21
    EvaluationRunner integration).

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration.  Stored as ``self.config``.

    Notes
    -----
    Class-level ``REQUIRED_PARAMS`` enables test parametrization without
    instantiation (mirrors AlignmentStage pattern).
    """

    REQUIRED_PARAMS: tuple[str, ...] = (
        "k_neighbours",
        "dist_metric",
        "smoothing",
        "threshold",
    )
    OPTIONAL_PARAMS: tuple[str, ...] = (
        "method",  # defaults to config.label_transfer_method (Phase 44 knn_voting/
        # cpd_weighted; Phase 48 adds pointnet2/egnn)
    )
    VALID_METHODS: tuple = ("knn_voting", "cpd_weighted", "pointnet2", "egnn")

    def __init__(self, config: EvalConfig) -> None:
        """Store the evaluation configuration.

        Parameters
        ----------
        config : EvalConfig
            Validated evaluation configuration.  Stored as ``self.config``.
        """
        super().__init__(config)

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate LabelTransferStage hyperparameters.

        Checks that all required keys are present and that each value meets
        the type and range constraints.  Raises ``ValueError`` on the first
        failure encountered (raise-on-first-failure pattern).

        Optional parameters (not in REQUIRED_PARAMS) are populated from
        config defaults if missing (Phase 44: ``method`` defaults to
        ``config.label_transfer_method``).

        Parameters
        ----------
        params : dict[str, Any]
            Hyperparameter dict to validate.  Must contain all four keys in
            ``REQUIRED_PARAMS``; keys in ``OPTIONAL_PARAMS`` are populated
            from config if missing.

        Returns
        -------
        None
            Returns ``None`` implicitly on success.

        Raises
        ------
        ValueError
            If any required key is missing: ``"Missing required param: {key}"``.
            If ``k_neighbours`` is not an int >= 1 or is a bool:
                ``"k_neighbours must be int >= 1; got {value!r}"``.
            If ``dist_metric`` is not a non-empty string:
                ``"dist_metric must be non-empty str; got {value!r}"``.
            If ``smoothing`` is not a numeric >= 0.0 or is a bool:
                ``"smoothing must be float >= 0.0; got {value!r}"``.
            If ``threshold`` is not a numeric >= 0.0 or is a bool:
                ``"threshold must be float >= 0.0; got {value!r}"``.
            If ``method`` is not one of ``VALID_METHODS``:
                ``"method must be one of {VALID_METHODS}; got {value!r}"``.
        """
        for key in self.REQUIRED_PARAMS:
            if key not in params:
                raise ValueError(f"Missing required param: {key}")

        # Phase 44: populate optional params from config if missing
        if "method" not in params:
            params["method"] = self.config.label_transfer_method

        if not (
            isinstance(params["k_neighbours"], int)
            and not isinstance(params["k_neighbours"], bool)
            and params["k_neighbours"] >= 1
        ):
            raise ValueError(f"k_neighbours must be int >= 1; got {params['k_neighbours']!r}")

        if not (isinstance(params["dist_metric"], str) and params["dist_metric"]):
            raise ValueError(f"dist_metric must be non-empty str; got {params['dist_metric']!r}")

        if not (
            isinstance(params["smoothing"], (int, float))
            and not isinstance(params["smoothing"], bool)
            and params["smoothing"] >= 0.0
        ):
            raise ValueError(f"smoothing must be float >= 0.0; got {params['smoothing']!r}")

        if not (
            isinstance(params["threshold"], (int, float))
            and not isinstance(params["threshold"], bool)
            and params["threshold"] >= 0.0
        ):
            raise ValueError(f"threshold must be float >= 0.0; got {params['threshold']!r}")

        if params["method"] not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}; got {params['method']!r}")

    @staticmethod
    def _load_learned_model(method: str, checkpoint_path: str | None) -> torch.nn.Module:
        """Load a Phase 47 checkpoint and construct a ``.eval()``-mode model (Phase 48).

        Performs the full verified load sequence (48-RESEARCH.md Pattern 1):
        checkpoint-path presence, file existence, safe deserialization
        (``weights_only=True``), and ``model_class`` consistency, each guarded
        by a specific ``ValueError`` so failures are diagnosable without
        inspecting a traceback.

        Parameters
        ----------
        method : str
            ``"pointnet2"`` or ``"egnn"`` — selects ``MODEL_REGISTRY[method]``.
        checkpoint_path : str or None
            Path to a ``.pt`` checkpoint produced by ``train_label_transfer.py``'s
            ``save_checkpoint`` (Phase 47). ``None`` means the corresponding
            ``EvalConfig`` field (``{method}_checkpoint_path``) was never set.

        Returns
        -------
        torch.nn.Module
            A ``model_cls(**ckpt["hyperparams"])`` instance with
            ``load_state_dict`` applied, in ``.eval()`` mode.

        Raises
        ------
        ValueError
            If ``checkpoint_path`` is ``None``, if the file does not exist,
            if ``torch.load`` fails for any reason, or if the checkpoint's
            ``model_class`` does not match ``method``.

        Notes
        -----
        Uses ``torch.load(checkpoint_path, map_location="cpu", weights_only=True)``
        (45-DESIGN.md security mandate — safe deserialization; CPU-only
        inference convention). No extra ``.to("cpu")`` call is added since
        ``map_location="cpu"`` already places all tensors on CPU before
        ``load_state_dict`` copies into the already-CPU-constructed model.
        """
        if checkpoint_path is None:
            raise ValueError(
                f"method={method!r} requires config.{method}_checkpoint_path to be set "
                "(got None). Set it in your EvalConfig/YAML, or use a different method."
            )
        if not Path(checkpoint_path).exists():
            raise ValueError(f"method={method!r} checkpoint file not found: {checkpoint_path!r}")
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ValueError(
                f"method={method!r} checkpoint at {checkpoint_path!r} failed to load: {e}"
            ) from e
        if ckpt.get("model_class") != method:
            raise ValueError(
                f"method={method!r} checkpoint at {checkpoint_path!r} has "
                f"model_class={ckpt.get('model_class')!r}, expected {method!r}."
            )
        model_cls = MODEL_REGISTRY[method]
        model = model_cls(**ckpt["hyperparams"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    @staticmethod
    def _check_alignment(
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
    ) -> float:
        """Compute mean per-frame Chamfer distance between source and target.

        Uses sequential frame pairing (same order as ``run()``).  When source
        and target differ in length, only ``min(len(source), len(target))``
        pairs are evaluated.

        Returns 0.0 when there are no paired frames.

        Raises
        ------
        ValueError
            If a paired source or target frame has a non-finite position
            (62-REVIEW WR-05).  Label transfer, the Chamfer diagnostic and
            the downstream kNN consistency all require finite positions, so
            such a frame is rejected here with a frame-specific message.
        """
        source_keys = sorted(source.keys())
        target_keys = sorted(target.keys())
        n_pairs = min(len(source_keys), len(target_keys))

        total = 0.0
        counted = 0
        for k in range(n_pairs):
            src_pos = source[source_keys[k]]["pos"]
            tgt_pos = target[target_keys[k]]["pos"]
            for side, key, pos in (("source", source_keys[k], src_pos),
                                   ("target", target_keys[k], tgt_pos)):
                n_bad = int((~torch.isfinite(pos).all(dim=1)).sum().item()) if pos.numel() else 0
                if n_bad > 0:
                    raise ValueError(
                        f"LabelTransferStage: {side} frame {key} has {n_bad} point(s) with a "
                        "non-finite position; label transfer requires finite positions"
                    )
            if src_pos.shape[0] == 0 or tgt_pos.shape[0] == 0:
                continue  # skip empty frames — chamfer(empty) returns nan
            dist = chamfer(src_pos, tgt_pos)
            total += float(dist.item() if hasattr(dist, "item") else dist)
            counted += 1

        return total / counted if counted > 0 else 0.0

    def run(
        self,
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        params: dict[str, Any],
        align_result: AlignResult | None = None,
    ) -> LabelResult:
        """Run label transfer from ``source`` to ``target`` and return a ``LabelResult``.

        Calls ``self.validate_params(params)`` as the first line (D-08 guarantee).

        Parameters
        ----------
        source : dict[int, zRegPointCloud]
            Source trajectory keyed by integer frame index.  Labels are
            transferred FROM each source frame.
        target : dict[int, zRegPointCloud]
            Target trajectory keyed by integer frame index.  Labels are
            transferred TO each target frame.
        params : dict[str, Any]
            Must contain all four keys in ``REQUIRED_PARAMS``.  See
            ``validate_params`` for the full constraint list.  ``method``
            (in ``OPTIONAL_PARAMS``) defaults to ``config.label_transfer_method``
            when absent.
        align_result : AlignResult | None, optional
            Result of a prior ``AlignmentStage.run()`` call, required only
            when ``params["method"] == "cpd_weighted"`` (Phase 44).  Ignored
            for ``method == "knn_voting"``.  Defaults to ``None``.

        Returns
        -------
        LabelResult
            Pydantic-frozen result with:
            - ``transferred_labels``: per-frame label tensors, keyed by
              target frame index.
            - ``params_used``: shallow copy of ``params`` (Pitfall 7).
            - ``pre_transfer_alignment``: mean per-frame Chamfer distance
              between source and target computed before transfer.

        Raises
        ------
        ValueError
            If ``params["method"] == "cpd_weighted"`` and ``align_result``
            is ``None`` (D-08), or if ``align_result.estep_results`` has no
            entry for the current frame pair's target key (D-08) — this
            happens when ``alignment_method`` is not ``"cpd"`` or
            ``cpd_penalty`` is ``None``, since the CPD posterior is only
            captured for CPD-registered frames.  Also raised when every
            receiver point of a frame has an oriented CPD posterior row with
            zero or non-finite mass (Phase 59 D-05), or when more than
            ``MAX_PMAT_FALLBACK_FRACTION`` (0.5) of a frame's rows are bad
            (Phase 62 RD-3); isolated bad rows fall back to the nearest
            provider label and are recorded in ``LabelResult.flags``.  Also
            raised, before any transfer, when a paired source or target frame
            has a non-finite position (62-REVIEW WR-05).  See
            ``validate_params`` for the other ``ValueError`` cases.

        Notes
        -----
        ``validate_params(params)`` is called as the first line (D-08).

        **Sequential pairing:**
        For k in range(min(len(source), len(target))), transfer labels from
        ``source[source_keys[k]]`` to ``target[target_keys[k]]``.  No
        frame-0 pass-through (D-02 + Phase 30).

        ``source_colors`` must be ``unsqueeze(-1)`` to shape (N, 1); result
        is squeezed with ``[:, 0]`` to shape (M,).  D-12.  This applies to
        the ``knn_voting`` path only — ``cpd_weighted`` one-hot encodes
        instead (D-05).

        ``params_used=dict(params)`` is a shallow copy (Pitfall 7).
        ``smoothing``, ``threshold``, ``dist_metric`` are validated but
        no-op in Phase 20.  D-04/D-05/D-06.

        **CPD-weighted path (Phase 44, Phase 59 NUM-04):** see module
        docstring for the pmat orientation (D-04; transposed for the default
        ``label_source='source'``, as stored for ``label_source='target'``),
        the zero-mass row guard and the one-hot/argmax (D-05) handling.
        """
        self.validate_params(params)

        if not source:
            raise ValueError("source must be non-empty; got 0 frames")
        if not target:
            raise ValueError("target must be non-empty; got 0 frames")

        alignment_dist = self._check_alignment(source, target)

        if self.config.run_alignment:
            _log.info(
                "Pre-transfer alignment quality (mean Chamfer): %.4f "
                "(alignment stage was run upstream)",
                alignment_dist,
            )
        else:
            if alignment_dist > ALIGNMENT_WARN_THRESHOLD:
                warnings.warn(
                    f"LabelTransferStage received potentially misaligned input "
                    f"(mean Chamfer distance = {alignment_dist:.4f} > {ALIGNMENT_WARN_THRESHOLD}). "
                    "Consider running AlignmentStage first (set run_alignment=true in config).",
                    stacklevel=2,
                )
            else:
                _log.info(
                    "Pre-transfer alignment quality (mean Chamfer): %.4f "
                    "(alignment stage skipped — input appears pre-aligned)",
                    alignment_dist,
                )

        source_keys = sorted(source.keys())
        target_keys = sorted(target.keys())

        n_pairs = min(len(source_keys), len(target_keys))
        transferred: dict[int, torch.Tensor] = {}
        flags: list[str] = []

        # Phase 48: load the learned model ONCE per run() call, before the
        # per-frame loop (Pattern 3) — never reloaded per frame pair.
        learned_model: torch.nn.Module | None = None
        if params["method"] in ("pointnet2", "egnn"):
            checkpoint_path = (
                self.config.pointnet2_checkpoint_path
                if params["method"] == "pointnet2"
                else self.config.egnn_checkpoint_path
            )
            learned_model = self._load_learned_model(params["method"], checkpoint_path)
            # Phase 62 (Research Open Q2) / 62-REVIEW WR-08: the checkpoint is
            # loaded on CPU; move the model ONCE to the frames' device (first
            # source frame) instead of re-moving a shared module per frame.
            model_device = source[sorted(source.keys())[0]]["pos"].device
            learned_model.to(model_device)

        for k in range(n_pairs):
            sk = source_keys[k]
            tk = target_keys[k]
            src_frame = source[sk]
            tgt_frame = target[tk]
            n_src = src_frame["pos"].shape[0]
            if params["k_neighbours"] > n_src:
                raise ValueError(
                    f"k_neighbours={params['k_neighbours']} exceeds source frame "
                    f"{sk} point count ({n_src})"
                )
            labels_tensor = src_frame.get(self.config.label_field)
            if labels_tensor is None:
                raise ValueError(
                    f"Source frame {sk} has no '{self.config.label_field}' field. "
                    "Label transfer requires annotated data. Check config.label_field."
                )

            if params["method"] == "cpd_weighted":
                if align_result is None:
                    raise ValueError(
                        "method='cpd_weighted' requires align_result (got None). "
                        "Run AlignmentStage first, or use method='knn_voting'."
                    )
                if tk not in align_result.estep_results:
                    raise ValueError(
                        f"method='cpd_weighted' requires align_result.estep_results[{tk}], "
                        "which is missing. This happens when alignment_method is not 'cpd' "
                        "or cpd_penalty is None — CPD posterior is only captured for CPD-"
                        "registered frames."
                    )
                # The aligned cloud carries exactly the target's keys, so the
                # receiver key tk finds the posterior in both directions.
                estep_result = align_result.estep_results[tk]
                # AlignmentStage stores expectation_step(t_source=aligned_source,
                # target=target): pmat[m, n] is the posterior that target point n
                # was generated by aligned-source component m, shape
                # (n_aligned_source, n_target).  transfer_colors expects
                # (n_receiver, n_provider) and row-normalises it.
                # - label_source == "target" (Phase 59 NUM-04): provider = target,
                #   receiver = aligned source -> use pmat as stored; row m is the
                #   soft correspondence of aligned-source point m over target points.
                # - default ("source"): provider = aligned source, receiver =
                #   target -> transpose (D-04).
                # The orientation comes from the config, never from shapes (square
                # frames would be ambiguous).
                if self.config.label_source == "target":
                    oriented = estep_result
                else:
                    oriented = estep_result._replace(pmat=estep_result.pmat.T)
                # Phase 62 RD-1..RD-3 (59-REVIEW IN-09a/c): one policy for all
                # pmat consumers. The shared helper decides bad rows (zero or
                # non-finite mass, or a non-finite receiver position) and their
                # nearest-provider fallback, and raises when every row is bad or
                # more than MAX_PMAT_FALLBACK_FRACTION of them would fall back.
                # Provider = src_frame, receiver = tgt_frame, exactly as passed
                # to transfer_colors below.
                repair = repair_pmat_rows(
                    oriented.pmat, src_frame["pos"], tgt_frame["pos"], context=f"frame {tk}"
                )
                if repair.n_bad > 0:
                    # The only report for this frame (RD-1b): the library reuses
                    # the repair below and neither repairs again nor warns.
                    _log.warning(repair.message)
                    flags.append(repair.message)
                one_hot = torch.nn.functional.one_hot(labels_tensor.long()).float()
                soft_scores = transfer_colors(
                    src_frame["pos"],
                    tgt_frame["pos"],
                    method=ColorTransferMethod.CPD_WEIGHTED,
                    source_colors=one_hot,
                    pmat_repair=repair,
                )
                frame_labels = soft_scores.argmax(dim=1)
                if repair.n_bad > 0:
                    # RD-2: nearest provider label, or -1 (no label, defence in
                    # depth: _check_alignment rejects non-finite positions) for a
                    # receiver whose position is non-finite.
                    idx = repair.fallback_idx
                    provider_labels = labels_tensor.long().to(idx.device)
                    fallback = torch.where(
                        idx >= 0,
                        provider_labels[idx.clamp(min=0)],
                        torch.full_like(idx, -1),
                    )
                    frame_labels[repair.bad_rows] = fallback.to(frame_labels.dtype)
                transferred[tk] = frame_labels
            elif params["method"] in ("pointnet2", "egnn"):
                # Phase 48: joint-cloud construction MUST byte-for-byte mirror
                # train_label_transfer.py:train_step's encoding (48-RESEARCH.md
                # Pattern 2) — source-then-target concatenation, one-hot over
                # [:n_src, :n_classes], unknown-flag at [n_src:, -1] = 1.0.
                n_classes = learned_model.n_classes  # read from the model, NOT
                # the data (Anti-Patterns — data-derived n_classes could
                # silently under-size joint_feat).
                joint_pos = torch.cat([src_frame["pos"], tgt_frame["pos"]], dim=0)
                dev = joint_pos.device
                if dev != model_device:
                    raise ValueError(
                        f"frame pair ({sk}, {tk}) is on {dev} but the learned model was "
                        f"placed on {model_device} (device of the first source frame); "
                        "all frames must share one device"
                    )
                joint_feat = torch.zeros(joint_pos.shape[0], n_classes + 1, device=dev)
                joint_feat[:n_src, :n_classes] = torch.nn.functional.one_hot(
                    labels_tensor.to(dev).long(), num_classes=n_classes
                ).float()
                joint_feat[n_src:, -1] = 1.0  # "unknown" flag for target rows
                with torch.no_grad():
                    logits = learned_model(joint_pos, joint_feat)  # (n_joint, n_classes)
                transferred[tk] = logits[n_src:].argmax(dim=1)
            else:
                transferred[tk] = transfer_colors(
                    src_frame["pos"],
                    tgt_frame["pos"],
                    method=ColorTransferMethod.KNN_VOTING,
                    source_colors=labels_tensor.unsqueeze(-1),
                    k=params["k_neighbours"],
                )[:, 0]

        return LabelResult(
            transferred_labels=transferred,
            params_used=dict(params),
            pre_transfer_alignment=alignment_dist,
            flags=flags,
        )
