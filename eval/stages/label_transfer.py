"""LabelTransferStage: thin color-transfer wrapper for FRAME-06 evaluation framework.

This module implements ``LabelTransferStage(PipelineStage)``, a thin orchestration
layer over ``zreg.color_transfer.transfer_colors()``.  No kNN or distance logic is
reimplemented here — all numerical computation delegates to the existing
``zreg.color_transfer.*`` package via KNN_VOTING (FRAME-06 explicit constraint).

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
"""

from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/metrics.py:53-67, eval/types.py:48-53.
from zreg.color_transfer import transfer_colors, ColorTransferMethod
from zreg.dataset import zRegPointCloud

import torch  # noqa: F401 — ensures consistent import order for downstream callers

from eval.config import EvalConfig
from eval.stages.base import PipelineStage
from eval.types import LabelResult

__all__ = ["LabelTransferStage"]


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

        Parameters
        ----------
        params : dict[str, Any]
            Hyperparameter dict to validate.  Must contain all four keys in
            ``REQUIRED_PARAMS``.

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
        """
        for key in self.REQUIRED_PARAMS:
            if key not in params:
                raise ValueError(f"Missing required param: {key}")

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

    def run(
        self,
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        params: dict[str, Any],
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
            ``validate_params`` for the full constraint list.

        Returns
        -------
        LabelResult
            Pydantic-frozen result with:
            - ``transferred_labels``: per-frame label tensors, keyed by
              target frame index.
            - ``params_used``: shallow copy of ``params`` (Pitfall 7).

        Notes
        -----
        ``validate_params(params)`` is called as the first line (D-08).

        **Sequential pairing:**
        For k in range(min(len(source), len(target))), transfer labels from
        ``source[source_keys[k]]`` to ``target[target_keys[k]]``.  No
        frame-0 pass-through (D-02 + Phase 30).

        ``source_colors`` must be ``unsqueeze(-1)`` to shape (N, 1); result
        is squeezed with ``[:, 0]`` to shape (M,).  D-12.

        ``params_used=dict(params)`` is a shallow copy (Pitfall 7).
        ``smoothing``, ``threshold``, ``dist_metric`` are validated but
        no-op in Phase 20.  D-04/D-05/D-06.
        """
        self.validate_params(params)

        if not source:
            raise ValueError("source must be non-empty; got 0 frames")
        if not target:
            raise ValueError("target must be non-empty; got 0 frames")

        source_keys = sorted(source.keys())
        target_keys = sorted(target.keys())

        # Real data has multi-channel RGB colors (N, C); synthetic data has
        # single-channel class indices (N,). Use "id" for real, "color" for synthetic.
        # Guard against color=None (CR-02): if color is absent, fall back to "id".
        sample_color = source[source_keys[0]]["color"]
        if sample_color is None:
            label_key = "id"
        else:
            label_key = "id" if sample_color.dim() > 1 else "color"

        n_pairs = min(len(source_keys), len(target_keys))
        transferred: dict[int, torch.Tensor] = {}

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
            labels_tensor = src_frame[label_key]
            if labels_tensor is None:
                raise ValueError(
                    f"Source frame {sk} has no labels in field '{label_key}'. "
                    "Ensure the dataset has been annotated before label transfer."
                )
            transferred[tk] = transfer_colors(
                src_frame["pos"],
                tgt_frame["pos"],
                method=ColorTransferMethod.KNN_VOTING,
                source_colors=labels_tensor.unsqueeze(-1),
                k=params["k_neighbours"],
            )[:, 0]

        return LabelResult(transferred_labels=transferred, params_used=dict(params))
