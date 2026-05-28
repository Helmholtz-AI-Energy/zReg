"""Tests for eval.types result models and (future) eval.metrics.MetricsEngine.

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
                              pass-through for f1/knn_consistency.  Stubbed
                              here; populated in Plan 18-02.
- TestComputeScore          — FRAME-03 / D-07 / D-08 / Pitfall 4:
                              auto-rescaled weights; scalar in [0,1];
                              zero-sum raises.  Stubbed here; populated in
                              Plan 18-02.
- TestSanityCheck           — FRAME-03 + Pitfall 3: 5 degenerate-input
                              cases (empty cloud, single-frame, all-same
                              labels, all-sentinel labels, non-finite
                              metric).  Stubbed here; populated in
                              Plan 18-02.
- TestAggregate             — FRAME-03 + Pitfall 7: empty list → {};
                              single-element std=0; known mean/std/min/max.
                              Stubbed here; populated in Plan 18-02.

This file does NOT import ``eval.metrics`` — that module is added in Plan
18-02.  Importing it now would break test collection.  Stubbed test bodies
use ``pytest.skip("populated in Plan 18-02")`` so the suite stays green.
"""

import pydantic
import pytest

from eval.config import EvalConfig  # noqa: F401  (Plan 18-02 stubs will use this)

# zreg.dataset MUST be imported before torch on macOS ARM to avoid libomp SIGABRT
from zreg.dataset import zRegPointCloud

import torch

from eval.types import (
    AlignResult,
    EvalReport,  # noqa: F401  (re-exported here for Plan 18-02 import surface)
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
    """FRAME-03 / D-05 / D-06: MetricsEngine.normalize maps to [0,1] with correct direction.

    Stubbed for Plan 18-01 (eval.metrics does not yet exist).  Plan 18-02 will
    populate these three subtests:

    - test_chamfer_zero_maps_to_one    — 1/(1+0) == 1.0 (D-05, lower-is-better)
    - test_f1_passes_through           — D-06: f1 / knn_consistency unchanged
    - test_chamfer_one_maps_to_half    — 1/(1+1) == 0.5 (D-05 sanity point)

    Canonical short-name key set per Pitfall 4: "chamfer", "hausdorff",
    "path_smoothness", "temporal_stability", "f1", "knn_consistency".
    """

    def test_placeholder(self):
        """Stub — populated in Plan 18-02 once eval.metrics.MetricsEngine exists."""
        pytest.skip("populated in Plan 18-02")


# ---------------------------------------------------------------------------
# TestComputeScore — FRAME-03 / D-07 / D-08 / Pitfall 4 (STUB — Plan 18-02)
# ---------------------------------------------------------------------------


class TestComputeScore:
    """FRAME-03 / D-07 / D-08: weighted score scalar in [0,1]; weights auto-rescale.

    Stubbed for Plan 18-01.  Plan 18-02 will populate three subtests:

    - test_returns_float_in_unit_interval — score is a float and lies in [0, 1]
    - test_weights_auto_rescale            — D-08: arbitrary positive weights normalized by sum
    - test_zero_sum_weights_raises         — guard: sum(weights) == 0 raises ValueError
    """

    def test_placeholder(self):
        """Stub — populated in Plan 18-02 once eval.metrics.MetricsEngine exists."""
        pytest.skip("populated in Plan 18-02")


# ---------------------------------------------------------------------------
# TestSanityCheck — FRAME-03 + Pitfall 3 (STUB — Plan 18-02)
# ---------------------------------------------------------------------------


class TestSanityCheck:
    """FRAME-03: MetricsEngine.sanity_check flags degenerate inputs.

    Stubbed for Plan 18-01.  Plan 18-02 will populate the 5 degenerate-input
    cases from Pitfall 3:

    - test_empty_cloud_flagged             — aligned_cloud == {} or every frame empty
    - test_single_frame_dataset_flagged    — len(aligned_cloud) <= 1
    - test_all_same_labels_flagged         — transferred_labels.unique().numel() <= 1
    - test_all_sentinel_labels_flagged     — every label == -1
    - test_non_finite_metric_flagged       — any StageMetrics field is NaN or Inf
    """

    def test_placeholder(self):
        """Stub — populated in Plan 18-02 once eval.metrics.MetricsEngine exists."""
        pytest.skip("populated in Plan 18-02")


# ---------------------------------------------------------------------------
# TestAggregate — FRAME-03 + Pitfall 7 (STUB — Plan 18-02)
# ---------------------------------------------------------------------------


class TestAggregate:
    """FRAME-03: MetricsEngine.aggregate returns mean/std/min/max per metric.

    Stubbed for Plan 18-01.  Plan 18-02 will populate three subtests:

    - test_empty_list_returns_empty_dict   — Pitfall 7: aggregate([]) -> {}
    - test_single_element_std_is_zero      — len == 1: std == 0.0, min == mean == max
    - test_known_values_mean_std_min_max   — fixture with known list, exact assertions
    """

    def test_placeholder(self):
        """Stub — populated in Plan 18-02 once eval.metrics.MetricsEngine exists."""
        pytest.skip("populated in Plan 18-02")
