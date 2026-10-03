"""MetricsEngine: stateless engine wrapping all six ``zreg.metrics.*`` primitives.

``MetricsEngine`` is the computational core of the evaluation framework
(FRAME-03).  It takes a validated ``EvalConfig`` and exposes four pure public
methods plus one convenience helper:

- ``normalize(StageMetrics) -> dict[str, float]`` — maps the six raw metrics
  to ``[0, 1]`` with correct direction (D-05: lower-is-better → ``1/(1+x)``;
  D-06: higher-is-better → pass-through).  Returned dict uses the canonical
  short-name key set per Pitfall 4 so it matches ``EvalConfig.metric_weights``.
- ``compute_score(StageMetrics) -> float`` — weighted scalar in ``[0, 1]``
  after auto-rescaling ``config.metric_weights`` by their sum (D-08).
  Raises ``ValueError`` on zero-sum weights (Pattern 3, ASVS V11).
- ``aggregate(list[StageMetrics]) -> dict[str, dict[str, float]]`` — per-field
  ``{mean, std, min, max}`` summaries keyed by the LONG raw field names per
  A7.  Empty list returns ``{}`` (Pitfall 7).
- ``sanity_check(*, align, label, metrics) -> list[str]`` — keyword-only
  inputs per A2.  Returns descriptive warning strings for five degenerate
  cases (Pitfall 3): empty cloud, single-frame dataset, all-same labels,
  all-sentinel labels, non-finite metric values.
- ``compute_stage_metrics(...)`` — optional helper (per A1) that runs all six
  primitives, packs into a ``StageMetrics`` instance, and returns it with
  ``normalized`` pre-populated via ``model_copy``.  ``chamfer`` and
  ``hausdorff`` are computed per-frame between ``aligned_cloud`` and
  ``target`` (every frame key shared by both) and then averaged — NOT on a
  single frame pair — via the private ``_frame_averaged_chamfer_hausdorff``
  helper.  ``.item()`` coercion (Pitfall 5) for scalar ``torch.Tensor``
  returns from ``chamfer``/``hausdorff``/``temporal_stability`` happens
  inside that helper and at the ``temporal_stability`` call site.

The engine is stateless aside from ``self.config``.  Construction performs no
I/O.  Every metric call is a one-way delegation to ``zreg.metrics.*`` — no
reimplementation (FRAME-03 explicit constraint).

Notes
-----
**Pitfall 4 — canonical short-name keys.**  ``normalize()`` MUST emit the
returned dict with the short-name keys (``"chamfer"``, ``"hausdorff"``,
``"path_smoothness"``, ``"temporal_stability"``, ``"f1"``,
``"knn_consistency"``), NOT the long ``StageMetrics`` field names
(``"chamfer_distance"``, ``"f1_score"``, ...).  ``EvalConfig.metric_weights``
uses the same short keys, so ``compute_score`` is a clean dot product.

**Pitfall 5 — ``.item()`` coercion for scalar torch.Tensor returns.**
``chamfer``, ``hausdorff``, and ``temporal_stability`` return scalar
``torch.Tensor`` objects.  ``path_smoothness``, ``compute_f1``, and
``knn_consistency`` return plain Python ``float``.  ``compute_stage_metrics``
calls ``.item()`` on the tensor returns before packing into ``StageMetrics``
(whose fields are typed ``float``) to avoid silent loss of precision or
``ValidationError`` from pydantic.

Interpreting normalized metrics
--------------------------------
The six raw metrics fall into two families that are normalized very
differently, and knowing which family a metric belongs to is necessary to
read the normalized value correctly.

**1. Distance-based metrics — chamfer, hausdorff, temporal_stability.**
These are unbounded, lower-is-better values measured in the *coordinate
units of the point cloud passed into the metric*, not in some fixed
percentage or physical unit.  ``normalize()`` maps them through the
saturating curve ``1 / (1 + x)``:

- ``x = 0`` → ``1.0`` (identical clouds / zero drift — the best possible
  score).
- ``x = 1`` (one coordinate unit of average error) → ``0.5``.
- As ``x → ∞``, the score → ``0`` but never reaches it.

Crucially, "one coordinate unit" is **not** an absolute, cross-dataset
constant — it is whatever scale the point cloud happens to be in when the
metric is computed.  In this pipeline that scale is normally set by
``EvalConfig.data_preprocessing`` (default ``method="standardize"``, i.e.
z-score normalization applied per-dimension in ``DataFactory.load_real`` /
``load_target`` — see ``eval/config.py:DataPreprocessingConfig``), so by
default ``x = 1`` roughly corresponds to *one standard deviation* of the
point spread in that trajectory, not to a fixed physical distance.  If
preprocessing is disabled (``data_preprocessing=None``) or set to a
different method, the same raw ``x`` value means something else again.
**Practical consequence:** a normalized chamfer score of 0.8 in one run and
0.8 in another run are only "equally good" if both runs used the same
preprocessing config on datasets of comparable spread — they are not
comparable across datasets with different scale/units, or across a
preprocessed vs. an unpreprocessed run.

**2. path_smoothness is a distance-based metric too, but not spatial.**
It is the variance of DTW-path curvature computed over *index* pairs
``(i, j)``, not over point coordinates, so it is dimensionless and
independent of point-cloud scale or ``data_preprocessing``.  Its magnitude
instead scales with the DTW path length / number of frames (a longer,
noisier warp path produces a larger raw value).  The same ``1 / (1 + x)``
saturating map is applied for consistency of range, but the "1 unit"
reference point here means "one unit of path curvature," unrelated to
spatial scale.

**3. Fraction-based metrics — f1, knn_consistency.**
These are already bounded in ``[0, 1]`` by construction (F1 is an
F-measure over label counts; kNN consistency is literally a fraction of
matching neighbours) — ``normalize()`` passes them through unchanged.
Unlike the distance-based family, these ARE directly comparable across
datasets and runs, since "1.0" always means the same thing (perfect label
agreement) regardless of point-cloud scale.

**What the normalized scores are for.**  Because of the scale-dependence
above, treat the six ``normalized`` values as being most reliable for:

- feeding ``compute_score``'s weighted sum for hyperparameter search /
  ranking trials *within* one dataset and preprocessing configuration;
- eyeballing which of the six metrics is comparatively weak or strong for
  a single run.

Do not use them for absolute claims across runs ("chamfer went from 0.6 to
0.8, so alignment improved by 0.2") without also checking the corresponding
raw values (``StageMetrics.chamfer_distance`` etc., or
``MetricsEngine.aggregate``'s per-field ``mean``/``std``) in their native
units — the raw values are what actually carries physical meaning.
"""

# stdlib first
import logging
import math
import statistics
from collections.abc import Mapping
from typing import NamedTuple

# zreg.metrics MUST precede import torch (libomp SIGABRT lesson from Phase 12;
# enforced in tests/conftest.py:20-24 and eval/data_factory.py:18-35).
# zreg.metrics transitively pulls in zreg.dataset and sklearn so it must
# precede torch on macOS-ARM.
from zreg.evaluation import (
    chamfer,
    compute_f1,
    hausdorff,
    knn_consistency,
    path_smoothness,
    temporal_stability,
)
from zreg.core.dataset import zRegPointCloud

# torch AFTER zreg.* imports
import torch

# local sibling modules last
from eval.config import EvalConfig
from eval.types import AlignResult, LabelResult, StageMetrics

__all__ = ["FrameAverage", "MetricsEngine"]

_log = logging.getLogger(__name__)


class FrameAverage(NamedTuple):
    """Frame-averaged chamfer/hausdorff plus frame-coverage information.

    Returned by ``MetricsEngine._frame_averaged_chamfer_hausdorff``.

    Parameters
    ----------
    chamfer : float
        Mean per-frame Chamfer distance over the scored frames, or
        ``+inf`` when no frame could be scored.
    hausdorff : float
        Mean per-frame Hausdorff distance over the scored frames, or
        ``+inf`` when no frame could be scored.
    n_scored : int
        Number of frames that contributed to the means.
    flags : list[str]
        Frame-coverage problems (each starting with ``"frame coverage:"``):
        non-dict inputs, no shared frame keys, partial key overlap, or
        skipped degenerate frames.  Empty for fully healthy input.
    """

    chamfer: float
    hausdorff: float
    n_scored: int
    flags: list[str]


class MetricsEngine:
    """Stateless engine wrapping all six ``zreg.metrics.*`` primitives.

    Constructed from a validated ``EvalConfig``.  No I/O is performed during
    construction.  Mirrors the engine-with-config shape of
    ``eval.data_factory.DataFactory`` from Phase 17.

    Public methods (FRAME-03):

    - ``normalize(metrics)`` — short-name canonical-key dict in ``[0, 1]``
    - ``compute_score(metrics)`` — auto-rescaled weighted scalar in ``[0, 1]``
    - ``aggregate(results)`` — per-metric mean/std/min/max across trials
    - ``sanity_check(*, align, label, metrics)`` — list of warning strings
    - ``compute_stage_metrics(...)`` — convenience helper that runs the six
      raw primitives and packs them into a fully-populated ``StageMetrics``

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration.  Only stored — no I/O is
        performed.  ``config.metric_weights`` is read by ``compute_score``.
    """

    def __init__(self, config: EvalConfig) -> None:
        """Store config.  No I/O is performed.

        Parameters
        ----------
        config : EvalConfig
            The validated experiment configuration produced by
            ``EvalConfig.from_yaml`` or direct construction.
        """
        self.config = config

    def normalize(self, metrics: StageMetrics) -> dict[str, float]:
        """Map the six raw ``StageMetrics`` fields to ``[0, 1]``.

        Lower-is-better metrics (chamfer, hausdorff, path_smoothness,
        temporal_stability) are mapped via ``1 / (1 + x)`` per D-05 — always
        in ``(0, 1]``, single-sample safe.  Higher-is-better metrics (f1,
        knn_consistency) are already in ``[0, 1]`` and are passed through
        unchanged per D-06.

        Parameters
        ----------
        metrics : StageMetrics
            Per-stage raw metric values.  The ``normalized`` field on the
            input is ignored; this method only reads the six raw float
            fields.

        Returns
        -------
        dict[str, float]
            Dict with exactly six entries keyed by the canonical short
            names (Pitfall 4): ``"chamfer"``, ``"hausdorff"``,
            ``"path_smoothness"``, ``"temporal_stability"``, ``"f1"``,
            ``"knn_consistency"``.  These keys match
            ``EvalConfig.metric_weights`` so ``compute_score`` is a clean
            dot product.

        Notes
        -----
        Canonical key mapping (Pitfall 4):

        - ``"chamfer"``             ``1 / (1 + chamfer_distance)``
        - ``"hausdorff"``           ``1 / (1 + hausdorff_distance)``
        - ``"path_smoothness"``     ``1 / (1 + path_smoothness)``
        - ``"temporal_stability"``  ``1 / (1 + temporal_stability)``
        - ``"f1"``                  ``f1_score`` (pass-through)
        - ``"knn_consistency"``     ``knn_consistency`` (pass-through)

        See the module docstring's "Interpreting normalized metrics" section
        for what each raw value is normalized *against* (point-cloud scale
        for chamfer/hausdorff/temporal_stability vs. DTW path index space for
        path_smoothness vs. already-bounded fractions for f1/knn_consistency)
        and why normalized scores from different datasets or preprocessing
        configs are not directly comparable for the distance-based metrics.
        """
        return {
            "chamfer": 1.0 / (1.0 + metrics.chamfer_distance),
            "hausdorff": 1.0 / (1.0 + metrics.hausdorff_distance),
            "path_smoothness": 1.0 / (1.0 + metrics.path_smoothness),
            "temporal_stability": 1.0 / (1.0 + metrics.temporal_stability),
            "f1": metrics.f1_score,
            "knn_consistency": metrics.knn_consistency,
        }

    def compute_score(self, metrics: StageMetrics) -> float:
        """Compute the auto-rescaled weighted score in ``[0, 1]``.

        Reads ``self.config.metric_weights``, rescales them by their sum so
        the user need not normalise to 1.0 exactly (D-08), and returns the
        dot product against ``metrics.normalized``.  Keys present in
        ``metric_weights`` but absent from ``metrics.normalized`` are
        silently skipped (Pattern 3 robustness against typos).

        Parameters
        ----------
        metrics : StageMetrics
            Per-stage metric values.  ``metrics.normalized`` MUST be
            pre-populated (typically via ``normalize`` then
            ``model_copy(update={"normalized": ...})``).

        Returns
        -------
        float
            Scalar score in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``sum(config.metric_weights.values()) == 0`` (Pattern 3 +
            ASVS V11 business-logic guard).
        """
        w = self.config.metric_weights
        norm = metrics.normalized
        total = sum(w[k] for k in w if k in norm)
        if total == 0:
            raise ValueError("metric_weights sum to zero")
        return sum(norm[k] * w[k] / total for k in w if k in norm)

    def aggregate(
        self, results: list[StageMetrics]
    ) -> dict[str, dict[str, float]]:
        """Aggregate per-metric ``{mean, std, min, max}`` across trials.

        Returns a nested dict keyed by the LONG raw ``StageMetrics`` field
        names per A7 (``chamfer_distance``, ``hausdorff_distance``,
        ``path_smoothness``, ``temporal_stability``, ``f1_score``,
        ``knn_consistency``) — NOT the short ``normalize`` keys.

        Parameters
        ----------
        results : list[StageMetrics]
            Per-trial raw metric values.  May be empty.

        Returns
        -------
        dict[str, dict[str, float]]
            For each of the six raw field names, a dict
            ``{"mean": ..., "std": ..., "min": ..., "max": ...}``.
            Returns ``{}`` on empty input (Pitfall 7).

        Notes
        -----
        - Empty list returns ``{}`` (Pitfall 7 — avoids
          ``StatisticsError`` from ``statistics.mean([])``).
        - Single-element list returns ``std=0.0`` for every metric
          (``statistics.pstdev`` requires len >= 2 to be meaningful;
          the explicit guard keeps the contract clean).
        """
        if not results:
            return {}
        field_names = [
            "chamfer_distance",
            "hausdorff_distance",
            "path_smoothness",
            "temporal_stability",
            "f1_score",
            "knn_consistency",
        ]
        out: dict[str, dict[str, float]] = {}
        for name in field_names:
            vals = [getattr(r, name) for r in results]
            out[name] = {
                "mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }
        return out

    def sanity_check(
        self,
        *,
        align: AlignResult | None = None,
        label: LabelResult | None = None,
        metrics: StageMetrics | None = None,
    ) -> list[str]:
        """Flag degenerate inputs with descriptive warning strings.

        Implements the five Pitfall 3 cases.  Returns an empty list when
        nothing is degenerate.  Inputs are keyword-only per A2 — every
        caller must be explicit about what it is checking.

        Parameters
        ----------
        align : AlignResult or None, keyword-only
            Optional alignment-stage output.  When present, checked for
            empty / single-frame ``aligned_cloud``.
        label : LabelResult or None, keyword-only
            Optional label-transfer-stage output.  When present, every
            frame's tensor is checked for all-same and all-sentinel labels.
        metrics : StageMetrics or None, keyword-only
            Optional metrics object.  When present, each of the six raw
            float fields is checked for non-finite values (NaN/Inf).

        Returns
        -------
        list[str]
            Descriptive flags.  Empty list when no degeneracies detected.

        Notes
        -----
        Implemented checks (Pitfall 3):

        1. ``align.aligned_cloud == {}`` → ``"empty aligned_cloud: 0 frames"``
        2. ``len(align.aligned_cloud) == 1`` → ``"single-frame dataset:
           DTW/temporal_stability undefined"``
        3. ``label`` tensor has only one unique value
           → ``"all-same labels in frame: F1 degenerate"`` (first occurrence)
        4. ``label`` tensor is all ``-1``
           → ``"all-sentinel labels: compute_f1 returns 0"`` (first occurrence)
        5. ``metrics`` field is NaN or Inf
           → ``"non-finite metric: {field}={value}"`` (per field)
        6. ``metrics.coverage_flags`` (frame-coverage problems recorded by
           ``compute_stage_metrics``) are appended verbatim; each starts
           with ``"frame coverage:"``.
        7. ``label.flags`` (label-transfer problems handled locally, e.g. a
           ``cpd_weighted`` zero-mass fallback) are appended verbatim.
        """
        flags: list[str] = []
        if align is not None:
            n_frames = len(align.aligned_cloud)
            if n_frames == 0:
                flags.append("empty aligned_cloud: 0 frames")
            elif n_frames == 1:
                flags.append(
                    "single-frame dataset: DTW/temporal_stability undefined"
                )
        if label is not None:
            # Early-exit on first occurrence of each degenerate case so the
            # returned list stays digestible even on large label dicts.
            same_flagged = False
            sentinel_flagged = False
            for tensor in label.transferred_labels.values():
                if tensor.numel() == 0:
                    continue
                if not sentinel_flagged and (tensor == -1).all().item():
                    flags.append(
                        "all-sentinel labels: compute_f1 returns 0"
                    )
                    sentinel_flagged = True
                    # All-sentinel implies all-same; do not also flag same.
                    same_flagged = True
                elif not same_flagged and tensor.unique().numel() <= 1:
                    flags.append("all-same labels in frame: F1 degenerate")
                    same_flagged = True
                if same_flagged and sentinel_flagged:
                    break
            flags.extend(label.flags)
        if metrics is not None:
            for name in (
                "chamfer_distance",
                "hausdorff_distance",
                "path_smoothness",
                "temporal_stability",
                "f1_score",
                "knn_consistency",
            ):
                value = getattr(metrics, name)
                if not math.isfinite(value):
                    flags.append(f"non-finite metric: {name}={value}")
            flags.extend(metrics.coverage_flags)
        return flags

    def _frame_averaged_chamfer_hausdorff(
        self,
        aligned_cloud: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
    ) -> FrameAverage:
        """Mean Chamfer and Hausdorff distance across every shared frame.

        Computes ``chamfer``/``hausdorff`` once per frame key present in
        *both* ``aligned_cloud`` and ``target`` and averages each metric
        across the frames that could be scored.  Every coverage problem is
        recorded as a flag instead of being silently absorbed (NUM-05).

        Parameters
        ----------
        aligned_cloud : dict[int, zRegPointCloud]
            Per-frame aligned source point clouds, e.g.
            ``AlignResult.aligned_cloud`` (or raw ``source`` when the
            alignment stage was skipped — the caller is responsible for that
            fallback).
        target : dict[int, zRegPointCloud]
            Per-frame target point clouds, keyed the same way.

        Returns
        -------
        FrameAverage
            ``chamfer`` / ``hausdorff``: means over the scored frames, or
            ``+inf`` when no frame could be scored (non-dict inputs, no
            shared keys, every shared frame degenerate).
            ``n_scored``: number of frames that contributed to the means.
            ``flags``: one ``"frame coverage: ..."`` string per problem —
            non-dict inputs, no shared frame keys, partial key overlap
            (also logged as a warning), or a skipped degenerate frame
            (empty / non-finite / otherwise rejected by ``chamfer`` or
            ``hausdorff``, or a non-finite result).

        Notes
        -----
        ``+inf`` rather than ``nan`` is the "no score" sentinel because
        ``StageMetrics.chamfer_distance`` / ``hausdorff_distance`` are
        ``Field(ge=0)`` and pydantic rejects ``nan``, whereas ``inf`` is
        accepted, reported by ``sanity_check`` as a non-finite metric and
        normalises to ``1 / (1 + inf) = 0.0`` (worst).  The former ``0.0``
        fallback normalised to ``1.0`` — a fake perfect score.
        """
        inf = float("inf")
        flags: list[str] = []
        if not isinstance(aligned_cloud, Mapping) or not isinstance(
            target, Mapping
        ):
            flags.append(
                "frame coverage: inputs are not per-frame dicts (got "
                f"{type(aligned_cloud).__name__}, {type(target).__name__}); "
                "chamfer/hausdorff set to inf"
            )
            return FrameAverage(inf, inf, 0, flags)

        aligned_keys = set(aligned_cloud)
        target_keys = set(target)
        shared = sorted(aligned_keys & target_keys)
        if not shared:
            flags.append(
                "frame coverage: no shared frame keys between aligned cloud "
                "and target; chamfer/hausdorff set to inf"
            )
            return FrameAverage(inf, inf, 0, flags)

        aligned_only = sorted(aligned_keys - target_keys)
        target_only = sorted(target_keys - aligned_keys)
        if aligned_only or target_only:
            msg = (
                f"frame coverage: partial overlap — scored {len(shared)} of "
                f"{len(aligned_keys | target_keys)} frames (aligned-only keys: "
                f"{aligned_only}, target-only keys: {target_only})"
            )
            flags.append(msg)
            _log.warning(msg)

        chamfer_vals: list[float] = []
        hausdorff_vals: list[float] = []
        for key in shared:
            src_pos = aligned_cloud[key]["pos"]
            tgt_pos = target[key]["pos"]
            try:
                c = chamfer(src_pos, tgt_pos).item()  # Pitfall 5: .item()
                h = hausdorff(src_pos, tgt_pos).item()  # Pitfall 5
            except ValueError as exc:
                flags.append(
                    f"frame coverage: skipped degenerate frame {key}: {exc}"
                )
                continue
            if not (math.isfinite(c) and math.isfinite(h)):
                flags.append(
                    f"frame coverage: skipped degenerate frame {key}: "
                    f"non-finite result (chamfer={c}, hausdorff={h})"
                )
                continue
            chamfer_vals.append(c)
            hausdorff_vals.append(h)

        if not chamfer_vals:
            return FrameAverage(inf, inf, 0, flags)
        return FrameAverage(
            statistics.mean(chamfer_vals),
            statistics.mean(hausdorff_vals),
            len(chamfer_vals),
            flags,
        )

    def compute_stage_metrics(
        self,
        aligned_cloud: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        warp_path: list[tuple[int, int]],
        transforms: list,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        points_for_knn: torch.Tensor,
        labels_for_knn: torch.Tensor,
        k_neighbours: int = 10,
    ) -> StageMetrics:
        """Run all six raw metrics and return a fully-populated ``StageMetrics``.

        Convenience wrapper (A1) for Phase 21 ``EvaluationRunner``.  Packs
        the six raw metric values into ``StageMetrics`` (whose fields are
        typed ``float``), then populates ``StageMetrics.normalized`` via
        ``self.normalize`` and ``model_copy``.

        Parameters
        ----------
        aligned_cloud : dict[int, zRegPointCloud]
            Per-frame aligned source point clouds — typically
            ``AlignResult.aligned_cloud`` (spatially registered per frame
            when CPD/ICP/SWD ran, temporally resampled only when
            ``cpd_penalty=None``), or the raw ``source`` dict when the
            alignment stage was skipped.  ``chamfer`` and ``hausdorff`` are
            computed once per frame key shared with ``target`` and then
            averaged (see ``_frame_averaged_chamfer_hausdorff``) — not on a
            single frame pair.
        target : dict[int, zRegPointCloud]
            Per-frame target point clouds, keyed the same way as
            ``aligned_cloud``.
        warp_path : list[tuple[int, int]]
            DTW alignment path passed to ``path_smoothness``.
        transforms : list
            Sequence of ``RigidTransformation`` / ``AffineTransformation``
            passed to ``temporal_stability``.
        y_true : torch.Tensor
            Ground-truth labels passed to ``compute_f1``.
        y_pred : torch.Tensor
            Predicted labels passed to ``compute_f1``.
        points_for_knn : torch.Tensor
            Point cloud passed to ``knn_consistency``.
        labels_for_knn : torch.Tensor
            Label tensor passed to ``knn_consistency``.
        k_neighbours : int, optional
            Number of neighbours for ``knn_consistency``.  Default ``10``.

        Returns
        -------
        StageMetrics
            A new ``StageMetrics`` instance with all six raw fields populated,
            ``normalized`` pre-computed via ``self.normalize`` and
            ``coverage_flags`` carrying the ``FrameAverage.flags`` of the
            chamfer/hausdorff computation.  When no frame could be scored,
            ``chamfer_distance`` / ``hausdorff_distance`` are ``+inf``
            (normalised ``0.0``), never ``0.0``.  When ``transforms`` is
            empty, ``temporal_stability`` is ``+inf`` (normalised ``0.0``)
            and a ``"metric unavailable: temporal_stability ..."`` flag is
            added (WR-06) instead of the former fake-perfect ``0.0``.

        Notes
        -----
        Called by ``EvaluationRunner._run_single`` (Phase 21) and
        ``HyperparamOptimizer._objective`` (Phase 22).
        """
        fa = self._frame_averaged_chamfer_hausdorff(aligned_cloud, target)
        flags = list(fa.flags)
        if len(transforms) == 0:
            # WR-06: temporal_stability([]) is 0.0, which normalises to 1.0
            # ("perfect") although nothing was measured.  Report it as
            # unavailable (+inf -> normalised 0.0) with an explicit flag.
            ts = float("inf")
            flags.append(
                "metric unavailable: temporal_stability (no per-frame transforms)"
            )
        else:
            ts = temporal_stability(transforms).item()  # Pitfall 5
        sm = StageMetrics(
            chamfer_distance=fa.chamfer,
            hausdorff_distance=fa.hausdorff,
            path_smoothness=path_smoothness(warp_path),  # already float
            temporal_stability=ts,
            f1_score=compute_f1(y_true, y_pred),  # already float
            knn_consistency=knn_consistency(
                points_for_knn, labels_for_knn, k=k_neighbours
            ),
            coverage_flags=flags,
        )
        return sm.model_copy(update={"normalized": self.normalize(sm)})
