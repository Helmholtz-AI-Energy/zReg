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
from zreg.registration import ICPRegistration, SlicedWassersteinAligner
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
    OPTIONAL_PARAMS: tuple[str, ...] = (
        "alignment_method",  # Phase 39: defaults to config.alignment_method
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

        Optional parameters (not in REQUIRED_PARAMS) are populated from
        config defaults if missing (Phase 39: alignment_method defaults to
        config.alignment_method).

        Parameters
        ----------
        params : dict[str, Any]
            Hyperparameter dict to validate.  Must contain all keys in
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

        # Phase 39: populate optional params from config if missing
        if "alignment_method" not in params:
            params["alignment_method"] = self.config.alignment_method

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

        if params["alignment_method"] not in ("cpd", "icp", "swd"):
            raise ValueError(
                f"alignment_method must be 'cpd', 'icp', or 'swd'; got {params['alignment_method']!r}"
            )

        # Phase 40: validate swd_variant if present and alignment_method is 'swd'
        if "swd_variant" in params and params["alignment_method"] == "swd":
            if params["swd_variant"] not in ("swd", "aswd", "oswd", "gswd", "pswd", "maxswd"):
                raise ValueError(
                    f"swd_variant must be one of {{'swd', 'aswd', 'oswd', 'gswd', 'pswd', 'maxswd'}}; "
                    f"got {params['swd_variant']!r}"
                )

    def run(
        self,
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        params: dict[str, Any],
    ) -> AlignResult:
        """Run DTW + spatial registration (CPD or ICP) from ``source`` to ``target`` and return an ``AlignResult``.

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
            Must contain all keys in ``REQUIRED_PARAMS`` (including
            ``alignment_method``).  See ``validate_params`` for the full
            constraint list.

        Returns
        -------
        AlignResult
            Pydantic-frozen result with:
            - ``aligned_cloud``: spatially registered (or DTW-resampled) source
              trajectory keyed by full target keys.  When ``cpd_penalty=None``
              and ``alignment_method`` is not applicable, contains temporally-
              resampled deep-copy source frames.  When spatial registration is
              enabled, each frame is spatially registered to its paired target
              frame via the chosen method (CPD or ICP).
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
            alignment_method=params["alignment_method"],
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
        alignment_method: str = "cpd",
        stored_transforms: dict[tuple[int, int], StoredTransform] | None = None,
        **kwargs,  # Captures swd_num_iterations, swd_variant if passed (Phase 40)
    ) -> dict[int, zRegPointCloud]:
        """Build the spatially-registered aligned source trajectory.

        For each target frame (full dataset), find the temporally corresponding
        source frame from the warp path and optionally apply spatial registration
        (CPD or ICP).  Returns a dict keyed by full target keys.

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
        alignment_method : str, default "cpd"
            Registration method: ``"cpd"`` for Coherent Point Drift, ``"icp"``
            for Open3D ICP (point-to-point, rigid), or ``"swd"`` for Sliced Wasserstein
            Distance with variant selection (Phase 40).
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
            matched_source_frame = src_frame
            matched_target_frame = target[tk]

            if cpd_penalty is None and alignment_method == "cpd":
                # Temporal-only: no spatial registration
                aligned[tk] = matched_source_frame
            elif cpd_penalty is not None and alignment_method == "cpd":
                # CPD spatial registration
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
                        matched_source_frame["pos"], min_vals=_st.src_min, max_vals=_st.src_max
                    )
                    transformed = _st.transform.transform(src_norm)
                    matched_source_frame["pos"] = utils.undo_normalize(
                        transformed, maxvals=_st.tgt_max, minvals=_st.tgt_min
                    )
                else:
                    # FALLBACK PATH (D-10): fresh CPD on raw data — covers edge frames outside
                    # the DTW window that were never computed in Step 1.
                    tf_params = {
                        "device": matched_source_frame["pos"].device,
                        "dtype": matched_source_frame["pos"].dtype,
                    }

                    if cpd_penalty == "nonrigid":
                        # NonRigidCPD does not accept tf_init_params (P5)
                        cpd_obj = NonRigidCPD(
                            source=matched_source_frame["pos"],
                            use_color=False,
                            log_freq=-1,
                        )
                    elif cpd_penalty == "affine":
                        cpd_obj = AffineCPD(
                            source=matched_source_frame["pos"],
                            use_color=False,
                            tf_init_params=tf_params,
                            log_freq=-1,
                        )
                    else:  # "rigid"
                        cpd_obj = RigidCPD(
                            source=matched_source_frame["pos"],
                            use_color=False,
                            tf_init_params=tf_params,
                            log_freq=-1,
                        )

                    # CR-01: use registration() return value — NonRigidCPD does not set
                    # self.transformation (overrides maximization_step without super() call)
                    reg_result = cpd_obj.registration(matched_target_frame["pos"], w=0.0, maxiter=1000, tol=1e-5)
                    matched_source_frame["pos"] = reg_result.transformation.transform(matched_source_frame["pos"])

                aligned[tk] = matched_source_frame
            elif alignment_method == "icp":
                # ICP spatial registration
                icp = ICPRegistration()
                stored_transform = icp.register(
                    source=matched_source_frame,
                    target=matched_target_frame,
                )
                # Apply stored transform to frame (denormalised space)
                registered_frame = AlignmentStage._apply_stored_transform(
                    matched_source_frame,
                    stored_transform,
                )
                aligned[tk] = registered_frame
            elif alignment_method == "swd":
                # SWD spatial registration (Phase 40)
                swd_variant = kwargs.get("swd_variant", "aswd")  # Default from EvalConfig
                swd_num_iterations = kwargs.get("swd_num_iterations", 50)  # Default

                aligner = SlicedWassersteinAligner(
                    variant=swd_variant,
                    num_iterations=swd_num_iterations,
                )
                stored_transform = aligner.register(
                    source=matched_source_frame,
                    target=matched_target_frame,
                )
                registered_frame = AlignmentStage._apply_stored_transform(
                    matched_source_frame,
                    stored_transform,
                )
                aligned[tk] = registered_frame
            else:
                # Temporal-only fallback (shouldn't reach here with valid params)
                aligned[tk] = matched_source_frame

        return aligned

    @staticmethod
    def _apply_stored_transform(
        cloud: zRegPointCloud,
        stored_transform: StoredTransform,
    ) -> zRegPointCloud:
        """Apply a stored transformation matrix to a point cloud.

        Parameters
        ----------
        cloud : zRegPointCloud
            Point cloud to transform.
        stored_transform : StoredTransform
            Cached transformation (ICPTransformation wrapper with matrix
            in denormalised space).

        Returns
        -------
        zRegPointCloud
            Transformed cloud (deep copy with updated pos field).
        """
        result = deepcopy(cloud)
        # Extract matrix from ICPTransformation wrapper
        matrix = torch.tensor(
            stored_transform.transform.matrix,
            dtype=cloud["pos"].dtype,
            device=cloud["pos"].device,
        )
        # Transform: pos_new = (matrix @ [pos, 1]^T)[:3]
        ones = torch.ones(
            (cloud["pos"].shape[0], 1),
            dtype=cloud["pos"].dtype,
            device=cloud["pos"].device,
        )
        pos_homog = torch.cat([cloud["pos"], ones], dim=1)  # [N, 4]
        pos_transformed = (matrix @ pos_homog.T).T  # [N, 4]
        result["pos"] = pos_transformed[:, :3]
        return result
