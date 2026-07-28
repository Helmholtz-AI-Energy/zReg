"""Tests for eval.types result models and eval.metrics.MetricsEngine.

Six test classes cover Phase 18 deliverables:

- TestResultTypesImportable — FRAME-04 gate 1: all 6 types importable from
                              eval.types (populated in Plan 18-01).
- TestStageMetricsFrozen    — D-02: pydantic frozen attribute reassignment
                              raises pydantic.ValidationError; AlignResult
                              and LabelResult accept arbitrary types
                              (zRegPointCloud, torch.Tensor).  Populated in
                              Plan 18-01.
- TestNormalize             — FRAME-03 / D-05 / D-06 / Pitfall 4: short-name
                              key set; 1/(1+x) for lower-is-better;
                              pass-through for f1/knn_consistency.  Populated
                              in Plan 18-02.
- TestComputeScore          — FRAME-03 / D-07 / D-08 / Pitfall 4:
                              auto-rescaled weights; scalar in [0,1];
                              zero-sum raises; missing-key robustness.
                              Populated in Plan 18-02.
- TestSanityCheck           — FRAME-03 + Pitfall 3: 5 degenerate-input
                              cases (empty cloud, single-frame, all-same
                              labels, all-sentinel labels, non-finite
                              metric) plus a clean-baseline negative
                              control.  Populated in Plan 18-02.
- TestAggregate             — FRAME-03 + Pitfall 7 + A7: empty list → {};
                              single-element std=0; known mean/std/min/max;
                              LONG raw field-name keys; four-stat shape.
                              Populated in Plan 18-02.
"""

import pydantic
import pytest

from eval.config import EvalConfig

# zreg.dataset MUST be imported before torch on macOS ARM to avoid libomp SIGABRT
from zreg.core.dataset import zRegPointCloud

import torch

from eval.metrics import MetricsEngine
from eval.types import (
    AlignResult,
    EvalReport,  # noqa: F401  (re-exported here for downstream-phase import surface)
    LabelResult,
    SearchResult,  # noqa: F401
    StageMetrics,
    Trial,  # noqa: F401
)


# ---------------------------------------------------------------------------
# TestResultTypesImportable — FRAME-04 gate 1 (Plan 18-01)
# ---------------------------------------------------------------------------


class TestResultTypesImportable:
    """FRAME-04 gate 1: all 6 result types importable from eval.types."""

    def test_all_six_importable(self):
        """All 6 result models (AlignResult, LabelResult, StageMetrics, Trial, SearchResult, EvalReport) load."""
        for cls in [AlignResult, LabelResult, StageMetrics, Trial, SearchResult, EvalReport]:
            assert cls.__name__  # import at top of file already proved loadability


# ---------------------------------------------------------------------------
# TestStageMetricsFrozen — D-02 frozen + arbitrary_types_allowed (Plan 18-01)
# ---------------------------------------------------------------------------


class TestStageMetricsFrozen:
    """D-02: result models are frozen and accept arbitrary tensor / zRegPointCloud types."""

    def test_attribute_assignment_raises(self):
        """D-02: reassigning a field on a constructed StageMetrics raises pydantic.ValidationError."""
        sm = StageMetrics(
            chamfer_distance=0.5,
            hausdorff_distance=0.3,
            path_smoothness=0.1,
            temporal_stability=0.2,
            f1_score=0.8,
            knn_consistency=0.9,
        )
        with pytest.raises(pydantic.ValidationError, match="frozen"):
            sm.chamfer_distance = 0.0

    def test_arbitrary_types_torch_tensor_accepted(self):
        """D-02: LabelResult accepts dict[int, torch.Tensor] without ValidationError."""
        lr = LabelResult(transferred_labels={0: torch.zeros(5)}, params_used={})
        assert isinstance(lr.transferred_labels[0], torch.Tensor)

    def test_arbitrary_types_zregpointcloud_accepted(self):
        """D-02: AlignResult accepts dict[int, zRegPointCloud] without ValidationError."""
        ar = AlignResult(
            aligned_cloud={0: zRegPointCloud()},
            warp_path=[(0, 0)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        assert isinstance(ar.aligned_cloud[0], zRegPointCloud)


# ---------------------------------------------------------------------------
# TestNormalize — FRAME-03 / D-05 / D-06 / Pitfall 4 (STUB — Plan 18-02)
# ---------------------------------------------------------------------------


class TestNormalize:
    """FRAME-03 / D-05 / D-06: MetricsEngine.normalize maps to [0,1] with correct direction."""

    def _engine(self) -> MetricsEngine:
        """Build a MetricsEngine with default config (D-07 default weights)."""
        return MetricsEngine(EvalConfig(data_path="x"))

    def test_chamfer_zero_maps_to_one(self):
        """D-05: lower-is-better metrics at 0 map to 1/(1+0) == 1.0."""
        sm = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.0,
            knn_consistency=0.0,
        )
        norm = self._engine().normalize(sm)
        assert norm["chamfer"] == 1.0
        assert norm["hausdorff"] == 1.0

    def test_f1_passes_through(self):
        """D-06: higher-is-better metrics (f1, knn_consistency) are unchanged."""
        sm = StageMetrics(
            chamfer_distance=10.0,
            hausdorff_distance=10.0,
            path_smoothness=10.0,
            temporal_stability=10.0,
            f1_score=0.75,
            knn_consistency=0.42,
        )
        norm = self._engine().normalize(sm)
        assert norm["f1"] == 0.75
        assert norm["knn_consistency"] == 0.42

    def test_chamfer_one_maps_to_half(self):
        """D-05 sanity point: 1/(1+1) == 0.5."""
        sm = StageMetrics(
            chamfer_distance=1.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.0,
            knn_consistency=0.0,
        )
        assert abs(self._engine().normalize(sm)["chamfer"] - 0.5) < 1e-9

    def test_canonical_short_keys(self):
        """Pitfall 4: normalize returns the canonical short-name key set."""
        sm = StageMetrics(
            chamfer_distance=0.1,
            hausdorff_distance=0.2,
            path_smoothness=0.3,
            temporal_stability=0.4,
            f1_score=0.5,
            knn_consistency=0.6,
        )
        norm = self._engine().normalize(sm)
        assert set(norm.keys()) == {
            "chamfer",
            "hausdorff",
            "path_smoothness",
            "temporal_stability",
            "f1",
            "knn_consistency",
        }

    def test_lower_is_better_in_unit_interval(self):
        """D-05: even very large lower-is-better values stay in (0, 1]."""
        sm = StageMetrics(
            chamfer_distance=1e6,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.0,
            knn_consistency=0.0,
        )
        norm = self._engine().normalize(sm)
        assert 0.0 < norm["chamfer"] <= 1.0


# ---------------------------------------------------------------------------
# TestComputeScore — FRAME-03 / D-07 / D-08 / Pitfall 4 (STUB — Plan 18-02)
# ---------------------------------------------------------------------------


class TestComputeScore:
    """FRAME-03 / D-07 / D-08: weighted score scalar in [0,1]; weights auto-rescale."""

    def test_returns_float_in_unit_interval(self):
        """D-08: compute_score returns a Python float in [0, 1] with default weights."""
        eng = MetricsEngine(EvalConfig(data_path="x"))
        sm = StageMetrics(
            chamfer_distance=1.0,
            hausdorff_distance=1.0,
            path_smoothness=1.0,
            temporal_stability=1.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        sm = sm.model_copy(update={"normalized": eng.normalize(sm)})
        score = eng.compute_score(sm)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_weights_auto_rescale(self):
        """D-08: weights need not sum to 1.0 — engine rescales by their sum."""
        eng = MetricsEngine(
            EvalConfig(
                data_path="x",
                metric_weights={
                    "chamfer": 70,
                    "f1": 30,
                    "hausdorff": 0,
                    "path_smoothness": 0,
                    "temporal_stability": 0,
                    "knn_consistency": 0,
                },
            )
        )
        sm = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=1.0,
            knn_consistency=0.0,
        )
        sm = sm.model_copy(update={"normalized": eng.normalize(sm)})
        # After normalize: chamfer=1.0 (1/(1+0)), f1=1.0 (pass-through).
        # After auto-rescale: 70/100 * 1.0 + 30/100 * 1.0 = 1.0.
        assert abs(eng.compute_score(sm) - 1.0) < 1e-9

    def test_zero_sum_weights_raises(self):
        """Pattern 3 + ASVS V11: zero-sum weights raise ValueError before any division."""
        eng = MetricsEngine(
            EvalConfig(
                data_path="x",
                metric_weights={
                    "chamfer": 0,
                    "hausdorff": 0,
                    "path_smoothness": 0,
                    "temporal_stability": 0,
                    "f1": 0,
                    "knn_consistency": 0,
                },
            )
        )
        sm = StageMetrics(
            chamfer_distance=1.0,
            hausdorff_distance=1.0,
            path_smoothness=1.0,
            temporal_stability=1.0,
            f1_score=0.5,
            knn_consistency=0.5,
        )
        sm = sm.model_copy(update={"normalized": eng.normalize(sm)})
        with pytest.raises(ValueError, match="zero"):
            eng.compute_score(sm)

    def test_unknown_normalized_key_is_skipped(self):
        """Pattern 3 robustness: weight keys absent from metrics.normalized are silently skipped."""
        eng = MetricsEngine(EvalConfig(data_path="x"))
        sm = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=1.0,
            knn_consistency=1.0,
        )
        full_norm = eng.normalize(sm)
        # Remove one canonical key to simulate a missing-key scenario.
        partial_norm = {k: v for k, v in full_norm.items() if k != "chamfer"}
        sm_partial = sm.model_copy(update={"normalized": partial_norm})
        # Must not raise KeyError; score re-normalised over present keys only → [0, 1].
        score = eng.compute_score(sm_partial)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestSanityCheck — FRAME-03 + Pitfall 3 (STUB — Plan 18-02)
# ---------------------------------------------------------------------------


class TestSanityCheck:
    """FRAME-03 + Pitfall 3: MetricsEngine.sanity_check flags degenerate inputs."""

    def _engine(self) -> MetricsEngine:
        """Build a MetricsEngine with default config."""
        return MetricsEngine(EvalConfig(data_path="x"))

    def test_empty_cloud_flagged(self):
        """Pitfall 3 case 1: aligned_cloud == {} is flagged as empty."""
        ar = AlignResult(
            aligned_cloud={},
            warp_path=[],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        flags = self._engine().sanity_check(align=ar)
        assert any("empty" in f.lower() for f in flags)

    def test_single_frame_dataset_flagged(self):
        """Pitfall 3 case 2: len(aligned_cloud) == 1 flagged as single-frame."""
        ar = AlignResult(
            aligned_cloud={0: zRegPointCloud()},
            warp_path=[(0, 0)],
            dtw_distance=0.0,
            n_changepoints=0,
            params_used={},
        )
        flags = self._engine().sanity_check(align=ar)
        assert any("single-frame" in f or "single frame" in f for f in flags)

    def test_all_same_labels_flagged(self):
        """Pitfall 3 case 3: a frame whose labels are all the same value is flagged."""
        lr = LabelResult(
            transferred_labels={0: torch.ones(10, dtype=torch.long) * 5},
            params_used={},
        )
        flags = self._engine().sanity_check(label=lr)
        assert any("all-same" in f or "same labels" in f for f in flags)

    def test_all_sentinel_labels_flagged(self):
        """Pitfall 3 case 4: a frame whose labels are all -1 (sentinel) is flagged."""
        lr = LabelResult(
            transferred_labels={0: torch.full((5,), -1, dtype=torch.long)},
            params_used={},
        )
        flags = self._engine().sanity_check(label=lr)
        assert any("sentinel" in f for f in flags)

    def test_empty_tensor_not_flagged_as_sentinel(self):
        """CR-03: empty label tensor must not trigger the all-sentinel flag (vacuous truth)."""
        lr = LabelResult(
            transferred_labels={0: torch.tensor([], dtype=torch.long)},
            params_used={},
        )
        flags = self._engine().sanity_check(label=lr)
        assert not any("sentinel" in f for f in flags)

    def test_non_finite_metric_flagged(self):
        """Pitfall 3 case 5: NaN/Inf in a raw StageMetrics field is flagged."""
        sm = StageMetrics.model_construct(
            chamfer_distance=float("nan"),
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.0,
            knn_consistency=0.0,
        )
        flags = self._engine().sanity_check(metrics=sm)
        assert any("non-finite" in f for f in flags)

    def test_clean_inputs_no_flags(self):
        """Negative control: healthy multi-frame align + varied labels + finite metrics → []."""
        ar = AlignResult(
            aligned_cloud={0: zRegPointCloud(), 1: zRegPointCloud()},
            warp_path=[(0, 0), (1, 1)],
            dtw_distance=0.5,
            n_changepoints=0,
            params_used={},
        )
        lr = LabelResult(
            transferred_labels={0: torch.tensor([0, 1, 2, 3], dtype=torch.long)},
            params_used={},
        )
        sm = StageMetrics(
            chamfer_distance=0.1,
            hausdorff_distance=0.2,
            path_smoothness=0.05,
            temporal_stability=0.01,
            f1_score=0.9,
            knn_consistency=0.8,
        )
        assert self._engine().sanity_check(align=ar, label=lr, metrics=sm) == []


# ---------------------------------------------------------------------------
# TestAggregate — FRAME-03 + Pitfall 7 (STUB — Plan 18-02)
# ---------------------------------------------------------------------------


class TestAggregate:
    """FRAME-03 + Pitfall 7 + A7: per-metric mean/std/min/max, LONG raw field-name keys."""

    def _engine(self) -> MetricsEngine:
        """Build a MetricsEngine with default config."""
        return MetricsEngine(EvalConfig(data_path="x"))

    def _sm(self, chamfer: float, **overrides) -> StageMetrics:
        """Build a StageMetrics with one custom chamfer + overridable other fields."""
        defaults = dict(
            chamfer_distance=chamfer,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.0,
            knn_consistency=0.0,
        )
        defaults.update(overrides)
        return StageMetrics(**defaults)

    def test_empty_list_returns_empty_dict(self):
        """Pitfall 7: aggregate([]) returns {} (no StatisticsError)."""
        assert self._engine().aggregate([]) == {}

    def test_single_element_std_is_zero(self):
        """Single-element list: std == 0.0 for every field; mean == min == max."""
        sm = self._sm(chamfer=0.5)
        agg = self._engine().aggregate([sm])
        for name in (
            "chamfer_distance",
            "hausdorff_distance",
            "path_smoothness",
            "temporal_stability",
            "f1_score",
            "knn_consistency",
        ):
            assert agg[name]["std"] == 0.0

    def test_known_values_mean_std_min_max(self):
        """Three-element fixture: mean/min/max exact, std matches statistics.pstdev."""
        import statistics

        sm1 = self._sm(chamfer=1.0)
        sm2 = self._sm(chamfer=2.0)
        sm3 = self._sm(chamfer=3.0)
        agg = self._engine().aggregate([sm1, sm2, sm3])
        assert abs(agg["chamfer_distance"]["mean"] - 2.0) < 1e-9
        assert agg["chamfer_distance"]["min"] == 1.0
        assert agg["chamfer_distance"]["max"] == 3.0
        expected_std = statistics.pstdev([1.0, 2.0, 3.0])
        assert abs(agg["chamfer_distance"]["std"] - expected_std) < 1e-9

    def test_long_name_keys(self):
        """A7: aggregate keys are LONG raw StageMetrics field names, not short normalize keys."""
        sm = self._sm(chamfer=0.5)
        agg = self._engine().aggregate([sm])
        assert set(agg.keys()) == {
            "chamfer_distance",
            "hausdorff_distance",
            "path_smoothness",
            "temporal_stability",
            "f1_score",
            "knn_consistency",
        }

    def test_each_field_has_four_stats(self):
        """Every field's inner dict has exactly the four expected stat keys."""
        sm = self._sm(chamfer=0.5)
        agg = self._engine().aggregate([sm])
        for name in agg:
            assert set(agg[name].keys()) == {"mean", "std", "min", "max"}


# ---------------------------------------------------------------------------
# TestEvalReportTrajectoryPaths — Phase 24 EXT-01 Task 1 (RED gate)
# ---------------------------------------------------------------------------


class TestEvalReportTrajectoryPaths:
    """Phase 24 EXT-01: EvalReport.trajectory_paths field existence and ordering."""

    def _make_report(self) -> EvalReport:
        """Minimal EvalReport with all required fields."""
        sm = StageMetrics(
            chamfer_distance=0.0,
            hausdorff_distance=0.0,
            path_smoothness=0.0,
            temporal_stability=0.0,
            f1_score=0.0,
            knn_consistency=0.0,
        )
        return EvalReport(
            params={},
            metrics=sm,
            aggregated_metrics={},
            per_dataset={},
        )

    def test_trajectory_paths_defaults_to_empty_list(self):
        """EvalReport.trajectory_paths defaults to [] when not provided."""
        report = self._make_report()
        assert report.trajectory_paths == []

    def test_trajectory_paths_field_order_after_plot_paths(self):
        """trajectory_paths appears after plot_paths and before sanity_flags in model_fields."""
        fields = list(EvalReport.model_fields.keys())
        assert "trajectory_paths" in fields, "trajectory_paths field not found in EvalReport"
        assert fields.index("trajectory_paths") < fields.index("sanity_flags"), (
            f"trajectory_paths not before sanity_flags; fields: {fields}"
        )

    def test_trajectory_paths_field_order_before_sanity_flags(self):
        """trajectory_paths appears before sanity_flags in model_fields."""
        fields = list(EvalReport.model_fields.keys())
        assert fields.index("trajectory_paths") < fields.index("sanity_flags"), (
            f"trajectory_paths not before sanity_flags; fields: {fields}"
        )
