"""AlignmentStage: thin DTW + CPD wrapper for FRAME-05 evaluation framework.

This module implements ``AlignmentStage(PipelineStage)``, a thin orchestration
layer over ``zreg.dtw.DynamicTimeWarping``.  No DTW or CPD logic is
reimplemented here — all numerical computation delegates to the existing
``zreg.dtw.*`` package (FRAME-05 explicit constraint).

Hyperparam mapping (D-07):

+------------------+--------------------------------------------+
| params key       | DynamicTimeWarping constructor arg         |
+==================+============================================+
| window_size      | window (Sakoe-Chiba band width)            |
+------------------+--------------------------------------------+
| dtw_dist_fn      | distance_metric (e.g. "euclidean", "swd")  |
+------------------+--------------------------------------------+
| cpd_penalty      | cpd_type ("rigid", "affine", "nonrigid",   |
|                  | or None)                                   |
+------------------+--------------------------------------------+
| step             | temporal stride applied to dataset keys    |
+------------------+--------------------------------------------+
| n_breakpoints    | upper-bound cap on n_changepoints count    |
+------------------+--------------------------------------------+

Notes
-----
**aligned_cloud semantics (Phase 33):**
``AlignResult.aligned_cloud`` is a ``dict[int, zRegPointCloud]`` keyed by
the full target keys.  Each frame is a deep copy of the corresponding
temporally-resampled source frame.  When ``cpd_penalty`` is set, each frame is
additionally spatially registered to its paired target frame via CPD
(``_build_aligned_cloud``).  The original ``source`` dict is never mutated.

**DTWResult.rotations quirk:**
When ``cpd_type=None``, ``DTWResult.rotations`` is ``[]`` (empty list), NOT
``None``, despite the annotation.  ``AlignmentStage`` does not reference
``result.rotations`` anywhere to avoid this pitfall (Pitfall 2 / 19-RESEARCH.md).
"""

from copy import deepcopy
from typing import Any

# zreg.* MUST precede torch on macOS-ARM (libomp SIGABRT).
# Enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/metrics.py:53-67, eval/types.py:48-53.
from zreg.cpd import RigidCPD, AffineCPD, NonRigidCPD
from zreg.dataset import zRegPointCloud
from zreg.dtw import DynamicTimeWarping
from zreg.types import StoredTransform
import zreg.utils as utils

import torch  # noqa: F401 — ensures consistent import order for downstream callers

from eval.config import EvalConfig
from eval.stages.base import PipelineStage
from eval.types import AlignResult

__all__ = ["AlignmentStage"]


class AlignmentStage(PipelineStage):
    """Concrete alignment stage wrapping ``DynamicTimeWarping``.

    Runs without ``LabelTransferStage`` present (FRAME-05 gate 1).  All live
    knobs come from the ``params`` dict passed to ``run()``; ``config`` is
    stored but currently unused inside the stage (reserved for Phase 21
    EvaluationRunner integration).

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration.  Stored as ``self.config``.

    Notes
    -----
    Class-level attributes ``REQUIRED_PARAMS`` and ``VALID_CPD`` are defined
    as class-body tuples so callers can introspect them without instantiation
    (used in ``tests/test_alignment_stage.py`` parametrize decorators).
    """

    REQUIRED_PARAMS: tuple[str, ...] = (
        "window_size",
        "step",
        "cpd_penalty",
        "dtw_dist_fn",
        "n_breakpoints",
    )
    VALID_CPD: tuple = (None, "rigid", "affine", "nonrigid")

    def __init__(self, config: EvalConfig) -> None:
        """Store the evaluation configuration.

        Parameters
        ----------
        config : EvalConfig
            Validated evaluation configuration.
        """
        super().__init__(config)

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate AlignmentStage hyperparameters per D-10.

        Checks that all required keys are present and that each value meets
        the type and range constraints.  Raises ``ValueError`` on the first
        failure encountered.

        Parameters
        ----------
        params : dict[str, Any]
            Hyperparameter dict to validate.  Must contain all five keys in
            ``REQUIRED_PARAMS``.

        Returns
        -------
        None
            Returns ``None`` implicitly on success.

        Raises
        ------
        ValueError
            If any required key is missing: ``"Missing required param: {key}"``.
            If ``window_size`` is not an int or is <= 0:
                ``"window_size must be int > 0"``.
            If ``step`` is not an int or is < 1:
                ``"step must be int >= 1"``.
            If ``cpd_penalty`` is not in ``VALID_CPD``:
                ``"cpd_penalty must be one of"``.
            If ``dtw_dist_fn`` is not a non-empty string:
                ``"dtw_dist_fn must be non-empty str"``.
            If ``n_breakpoints`` is not an int or is < 0:
                ``"n_breakpoints must be int >= 0"``.
        """
        for key in self.REQUIRED_PARAMS:
            if key not in params:
                raise ValueError(f"Missing required param: {key}")

        if not (isinstance(params["window_size"], int)
                and not isinstance(params["window_size"], bool)
                and params["window_size"] > 0):
            raise ValueError(f"window_size must be int > 0; got {params['window_size']!r}")

        if not (isinstance(params["step"], int)
                and not isinstance(params["step"], bool)
                and params["step"] >= 1):
            raise ValueError(f"step must be int >= 1; got {params['step']!r}")

        if params["cpd_penalty"] not in self.VALID_CPD:
            raise ValueError(
                f"cpd_penalty must be one of {self.VALID_CPD}; got {params['cpd_penalty']!r}"
            )

        if not (isinstance(params["dtw_dist_fn"], str) and params["dtw_dist_fn"]):
            raise ValueError(f"dtw_dist_fn must be non-empty str; got {params['dtw_dist_fn']!r}")

        if not (isinstance(params["n_breakpoints"], int)
                and not isinstance(params["n_breakpoints"], bool)
                and params["n_breakpoints"] >= 0):
            raise ValueError(f"n_breakpoints must be int >= 0; got {params['n_breakpoints']!r}")

    def run(
        self,
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        params: dict[str, Any],
    ) -> AlignResult:
        """Run DTW + CPD alignment from ``source`` to ``target`` and return an ``AlignResult``.

        Calls ``self.validate_params(params)`` as the first line (D-09 guarantee).

        Parameters
        ----------
        source : dict[int, zRegPointCloud]
            Source trajectory keyed by integer frame index.  Mirrors the
            shape returned by ``DataFactory.load_real()`` and
            ``DataFactory.generate_synthetic()``.  Never mutated by ``run()``.
        target : dict[int, zRegPointCloud]
            Target trajectory keyed by integer frame index.  DTW aligns
            ``source`` against ``target``; ``target`` is not returned in the
            result (D-06).
        params : dict[str, Any]
            Must contain all five keys in ``REQUIRED_PARAMS``.  See
            ``validate_params`` for the full constraint list.

        Returns
        -------
        AlignResult
            Pydantic-frozen result with:
            - ``aligned_cloud``: CPD-transformed (or DTW-resampled) source
              trajectory keyed by full target keys.  When ``cpd_penalty=None``,
              contains temporally-resampled deep-copy source frames.  When
              ``cpd_penalty`` is set, each frame is spatially registered to
              its paired target frame via CPD.
            - ``warp_path``: DTW optimal alignment path.
            - ``dtw_distance``: accumulated DTW cost.
            - ``n_changepoints``: diagonal/non-diagonal transition count,
              capped at ``params["n_breakpoints"]`` (D-06).
            - ``params_used``: shallow copy of ``params`` (Pitfall 7).

        Notes
        -----
        ``aligned_cloud`` is built by ``_build_aligned_cloud`` using the
        already-computed ``warp_path`` — DTW is not re-run.  The original
        ``source`` dict is never mutated; all frames in ``aligned_cloud``
        are deep copies.

        ``params_used`` is a shallow copy (``dict(params)``) to avoid Pitfall 7:
        mutating the original ``params`` dict after ``run()`` would otherwise
        silently corrupt the frozen ``AlignResult``.
        """
        self.validate_params(params)

        # Build strided sub-dicts symmetrically for source and target (Pitfall 1).
        # sorted() makes key order deterministic for non-contiguous key sets.
        source_sorted = sorted(source.keys())
        source_sub = {i: source[k] for i, k in enumerate(source_sorted[:: params["step"]])}

        target_sorted = sorted(target.keys())
        target_sub = {i: target[k] for i, k in enumerate(target_sorted[:: params["step"]])}

        result = DynamicTimeWarping(
            x=source_sub,
            y=target_sub,  # Phase 30: two-dataset paired alignment (D-06)
            distance_metric=params["dtw_dist_fn"],
            cpd_type=params["cpd_penalty"],
            window=params["window_size"],
            downsample_method=None,  # disable random downsampling — works regardless of color field presence
        ).compute()

        n_jumps = self._count_jumps(result.warping_path)
        n_changepoints = min(n_jumps, params["n_breakpoints"])  # D-06 cap

        aligned_cloud = self._build_aligned_cloud(
            source=source,
            target=target,
            source_sub=source_sub,
            target_sub=target_sub,
            warp_path=result.warping_path,
            cpd_penalty=params["cpd_penalty"],
            stored_transforms=result.stored_transforms,
        )

        return AlignResult(
            aligned_cloud=aligned_cloud,
            warp_path=result.warping_path,
            dtw_distance=result.distance,
            n_changepoints=n_changepoints,
            params_used=dict(params),  # shallow copy — Pitfall 7
        )

    @staticmethod
    def _count_jumps(warping_path: list[tuple[int, int]]) -> int:
        """Count diagonal-to-non-diagonal (and vice versa) transitions in a DTW path.

        A diagonal step advances both ``i`` and ``j`` by 1 simultaneously.  A
        non-diagonal step (``"step"`` kind) advances exactly one of them by 1.
        A "jump" is any transition from one kind to the other (D-04 / D-05).

        Parameters
        ----------
        warping_path : list[tuple[int, int]]
            DTW warping path from ``DTWResult.warping_path``, ordered from
            ``(0, 0)`` to ``(n-1, m-1)``.

        Returns
        -------
        int
            Raw jump count before the ``n_breakpoints`` cap.  The cap is
            applied in ``run()``, not here (D-06).

        Notes
        -----
        Diagonal step: both ``i`` and ``j`` advance by 1.
        Non-diagonal step: only one of ``i`` or ``j`` advances by 1.
        A jump is a diagonal-to-non-diagonal or non-diagonal-to-diagonal
        transition.  (D-04 / D-05.)

        Returns 0 for paths with fewer than 2 points (no steps possible).
        """
        if len(warping_path) < 2:
            return 0

        def _kind(prev: tuple[int, int], curr: tuple[int, int]) -> str:
            di = curr[0] - prev[0]
            dj = curr[1] - prev[1]
            return "diag" if (di == 1 and dj == 1) else "step"

        kinds = [_kind(warping_path[k - 1], warping_path[k]) for k in range(1, len(warping_path))]
        return sum(1 for k in range(1, len(kinds)) if kinds[k] != kinds[k - 1])

    @staticmethod
    def _build_aligned_cloud(
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        source_sub: dict[int, zRegPointCloud],
        target_sub: dict[int, zRegPointCloud],
        warp_path: list[tuple[int, int]],
        cpd_penalty: str | None,
        stored_transforms: dict[tuple[int, int], StoredTransform] | None = None,
    ) -> dict[int, zRegPointCloud]:
        """Build the CPD-transformed aligned source trajectory.

        For each target frame (full dataset), find the temporally corresponding
        source frame from the warp path and optionally apply CPD spatial
        registration.  Returns a dict keyed by full target keys.

        Parameters
        ----------
        source : dict[int, zRegPointCloud]
            Full source trajectory (all frames).
        target : dict[int, zRegPointCloud]
            Full target trajectory (all frames).
        source_sub : dict[int, zRegPointCloud]
            Strided source sub-dict (keys 0..N-1, values = original source frames).
        target_sub : dict[int, zRegPointCloud]
            Strided target sub-dict (keys 0..M-1, values = original target frames).
        warp_path : list[tuple[int, int]]
            DTW warping path as (source_sub_idx, target_sub_idx) pairs.
        cpd_penalty : str | None
            CPD type (``"rigid"``, ``"affine"``, ``"nonrigid"``) or ``None``
            for temporal-only alignment.
        stored_transforms : dict[tuple[int, int], StoredTransform] | None, optional
            Mapping from ``(src_sub_idx, tgt_sub_idx)`` to ``StoredTransform``,
            captured during Step 1 (pairwise distance computation).  When a key is
            present and ``cpd_penalty`` is not ``None``, the stored transform is
            reused (normalise → apply stored transform → denormalise) instead of
            running fresh CPD.  ``None`` is treated as an empty dict (D-08).

        Returns
        -------
        dict[int, zRegPointCloud]
            Aligned trajectory with same keys as target.  Each frame is a
            deep copy of the corresponding source frame, optionally
            spatially registered via CPD.
        """
        if stored_transforms is None:
            stored_transforms = {}
        # Build lookup: target_sub_idx -> first matching source_sub_idx
        tgt_to_src_sub: dict[int, int] = {}
        for src_sub_idx, tgt_sub_idx in warp_path:
            if tgt_sub_idx not in tgt_to_src_sub:
                tgt_to_src_sub[tgt_sub_idx] = src_sub_idx

        n_sub = len(target_sub)
        target_keys_sorted = sorted(target.keys())
        n_target_full = len(target_keys_sorted)
        # CR-02: compute fallback once — warp_path is guaranteed non-empty by DTW
        fallback_src_sub_idx = warp_path[0][0] if warp_path else 0

        aligned: dict[int, zRegPointCloud] = {}

        for pos, tk in enumerate(target_keys_sorted):
            # Nearest strided target index for this full-resolution position
            tgt_sub_idx = min(round(pos * n_sub / n_target_full), n_sub - 1)

            src_sub_idx = tgt_to_src_sub.get(tgt_sub_idx, fallback_src_sub_idx)

            # Retrieve and deep-copy the source frame (no in-place mutation)
            src_frame = deepcopy(source_sub[src_sub_idx])

            if cpd_penalty is not None:
                key = (src_sub_idx, tgt_sub_idx)
                # None-guard (CR-02): a StoredTransform with None normalisation params was
                # created when normalize=False in Step 1. Since no normalisation was applied
                # then, the reuse path must not be taken — fall through to fresh CPD instead.
                _st = stored_transforms.get(key)
                if _st is not None and _st.src_min is not None and _st.tgt_max is not None:
                    # REUSE PATH (D-09): normalise with Step-1 params → apply stored transform →
                    # denormalise into target coordinate space. Avoids re-running CPD from
                    # identity on raw unnormalised data (fixes 8× scale convergence failure).
                    src_norm, _ = utils.normalize_point_cloud(
                        src_frame["pos"], min_vals=_st.src_min, max_vals=_st.src_max
                    )
                    transformed = _st.transform.transform(src_norm)
                    src_frame["pos"] = utils.undo_normalize(
                        transformed, maxvals=_st.tgt_max, minvals=_st.tgt_min
                    )
                else:
                    # FALLBACK PATH (D-10): fresh CPD on raw data — covers edge frames outside
                    # the DTW window that were never computed in Step 1.
                    # CR-03: use full target frame for spatial registration, not strided sub-dict
                    tgt_frame = target[tk]
                    tf_params = {
                        "device": src_frame["pos"].device,
                        "dtype": src_frame["pos"].dtype,
                    }

                    if cpd_penalty == "nonrigid":
                        # NonRigidCPD does not accept tf_init_params (P5)
                        cpd_obj = NonRigidCPD(
                            source=src_frame["pos"],
                            use_color=False,
                            log_freq=-1,
                        )
                    elif cpd_penalty == "affine":
                        cpd_obj = AffineCPD(
                            source=src_frame["pos"],
                            use_color=False,
                            tf_init_params=tf_params,
                            log_freq=-1,
                        )
                    else:  # "rigid"
                        cpd_obj = RigidCPD(
                            source=src_frame["pos"],
                            use_color=False,
                            tf_init_params=tf_params,
                            log_freq=-1,
                        )

                    # CR-01: use registration() return value — NonRigidCPD does not set
                    # self.transformation (overrides maximization_step without super() call)
                    reg_result = cpd_obj.registration(tgt_frame["pos"], w=0.0, maxiter=1000, tol=1e-5)
                    src_frame["pos"] = reg_result.transformation.transform(src_frame["pos"])

            aligned[tk] = src_frame

        return aligned
