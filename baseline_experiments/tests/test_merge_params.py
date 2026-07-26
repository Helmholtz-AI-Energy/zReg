"""Tests for merge_params.py: merge_two, merge_selfcal_params,
merge_groundtruth_params, merge_combined_params.
"""

from __future__ import annotations

import pytest

# conftest.py inserts baseline_experiments/scripts into sys.path.
from merge_params import (
    merge_combined_params,
    merge_groundtruth_params,
    merge_selfcal_params,
    merge_two,
)


# ── merge_two ─────────────────────────────────────────────────────────────────


class TestMergeTwo:
    def test_numeric_float_average(self):
        result = merge_two({"a": 4.0}, {"a": 6.0})
        assert result == {"a": 5.0}

    def test_numeric_int_rounds_to_int(self):
        result = merge_two({"a": 3}, {"a": 4})
        assert result == {"a": 4}  # round((3+4)/2) = round(3.5) = 4
        assert isinstance(result["a"], int)

    def test_numeric_int_exact_half(self):
        result = merge_two({"a": 4}, {"a": 6})
        assert result == {"a": 5}
        assert isinstance(result["a"], int)

    def test_numeric_mixed_int_float_stays_float(self):
        # one int, one float → avg is float, not rounded
        result = merge_two({"a": 4}, {"a": 6.0})
        assert result["a"] == 5.0
        assert isinstance(result["a"], float)

    def test_equal_string_kept(self):
        result = merge_two({"cpd": "rigid"}, {"cpd": "rigid"})
        assert result == {"cpd": "rigid"}

    def test_equal_none_kept(self):
        result = merge_two({"cpd": None}, {"cpd": None})
        assert result == {"cpd": None}

    def test_conflict_falls_back_to_default(self):
        result = merge_two({"cpd": "rigid"}, {"cpd": "affine"}, defaults={"cpd": "rigid"})
        assert result == {"cpd": "rigid"}

    def test_conflict_raises_without_default(self):
        with pytest.raises(ValueError, match="conflict"):
            merge_two({"cpd": "rigid"}, {"cpd": "affine"})

    def test_key_only_in_a(self):
        result = merge_two({"a": 1}, {})
        assert result == {"a": 1}

    def test_key_only_in_b(self):
        result = merge_two({}, {"b": 2})
        assert result == {"b": 2}

    def test_disjoint_dicts_union(self):
        result = merge_two({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_defaults_not_unioned_into_key_set(self):
        # defaults keys absent from both a and b must NOT appear in output
        result = merge_two({"a": 1}, {"a": 3}, defaults={"a": 10, "extra": 99})
        assert "extra" not in result

    def test_bool_not_treated_as_numeric_conflict(self):
        # bool subclasses int in Python but merge_two must NOT average booleans
        with pytest.raises(ValueError, match="conflict"):
            merge_two({"flag": True}, {"flag": False})

    def test_bool_equal_kept(self):
        result = merge_two({"flag": True}, {"flag": True})
        assert result == {"flag": True}

    def test_empty_both(self):
        assert merge_two({}, {}) == {}


# ── merge_selfcal_params ──────────────────────────────────────────────────────


class TestMergeSelfcalParams:
    def test_alignment_keys_averaged_across_kobitski_and_shah(self):
        kobitski = {"window_size": 5, "cpd_penalty": "rigid"}
        shah_align = {"window_size": 15, "cpd_penalty": "rigid"}
        shah_lt = {"k_neighbours": 3}
        defaults = {"window_size": 10, "cpd_penalty": "rigid", "k_neighbours": 5}
        result = merge_selfcal_params(kobitski, shah_align, shah_lt, defaults)
        assert result["window_size"] == 10.0
        assert result["cpd_penalty"] == "rigid"

    def test_label_transfer_key_from_shah_lt(self):
        kobitski = {"window_size": 10}
        shah_align = {"window_size": 10}
        shah_lt = {"k_neighbours": 3}
        defaults = {"window_size": 10, "k_neighbours": 5}
        result = merge_selfcal_params(kobitski, shah_align, shah_lt, defaults)
        assert result["k_neighbours"] == 3

    def test_shah_lt_alignment_key_contributes_to_average(self):
        # If shah_lt also contains an alignment key it participates in the
        # second merge_two pass, shifting the average further.
        # Use floats so intermediate results are not subject to int rounding.
        kobitski = {"window_size": 10.0}
        shah_align = {"window_size": 20.0}
        shah_lt = {"window_size": 30.0, "k_neighbours": 3.0}
        defaults = {"window_size": 10.0, "k_neighbours": 5.0}
        result = merge_selfcal_params(kobitski, shah_align, shah_lt, defaults)
        # pass 1: (10.0+20.0)/2 = 15.0; pass 2: (15.0+30.0)/2 = 22.5
        assert result["window_size"] == 22.5

    def test_defaults_fill_missing_keys(self):
        kobitski = {"window_size": 10}
        shah_align = {"window_size": 10}
        shah_lt = {}
        defaults = {"window_size": 10, "k_neighbours": 7, "step": 8}
        result = merge_selfcal_params(kobitski, shah_align, shah_lt, defaults)
        assert result["k_neighbours"] == 7
        assert result["step"] == 8

    def test_all_sources_empty_returns_defaults(self):
        defaults = {"window_size": 10, "k_neighbours": 5}
        result = merge_selfcal_params({}, {}, {}, defaults)
        assert result == defaults


# ── merge_groundtruth_params ──────────────────────────────────────────────────


class TestMergeGroundtruthParams:
    def test_kobitski_and_shah_alignment_averaged(self):
        # Use floats so intermediate results are not subject to int rounding.
        kobitski = {"window_size": 5.0}
        shah = {"window_size": 15.0, "k_neighbours": 4.0}
        defaults = {"window_size": 10.0, "k_neighbours": 5.0}
        result = merge_groundtruth_params(kobitski, shah, defaults)
        # shah passed as both shah_alignment and shah_label_transfer.
        # pass 1: merge(kobitski, shah): window=(5.0+15.0)/2=10.0, k_neighbours=4.0
        # pass 2: merge(result, shah): window=(10.0+15.0)/2=12.5, k_neighbours=(4.0+4.0)/2=4.0
        assert result["window_size"] == 12.5
        assert result["k_neighbours"] == 4.0

    def test_shah_only_lt_key_passes_through_unchanged(self):
        # A key that exists only in shah_both (not in kobitski) is taken directly.
        kobitski = {"window_size": 10}
        shah = {"window_size": 10, "k_neighbours": 3}
        defaults = {"window_size": 10, "k_neighbours": 5}
        result = merge_groundtruth_params(kobitski, shah, defaults)
        # k_neighbours only in shah → 3 from pass 1; then (3+3)/2=3 in pass 2.
        assert result["k_neighbours"] == 3

    def test_defaults_fill_missing_keys(self):
        kobitski = {}
        shah = {}
        defaults = {"window_size": 10, "step": 8}
        result = merge_groundtruth_params(kobitski, shah, defaults)
        assert result == defaults


# ── merge_combined_params ─────────────────────────────────────────────────────


class TestMergeCombinedParams:
    def _run(self, kobitski_sc, shah_sc_align, shah_sc_lt,
             kobitski_gt, shah_gt, defaults):
        return merge_combined_params(
            kobitski_sc, shah_sc_align, shah_sc_lt,
            kobitski_gt, shah_gt, defaults,
        )

    def test_numeric_params_averaged_across_both_regimes(self):
        # selfcal sources all agree on window_size=20 → selfcal_merged=20
        # gt sources all agree on window_size=10 → gt_merged=10
        # combined: (20+10)/2 = 15
        kobitski_sc = {"window_size": 20}
        shah_sc_align = {"window_size": 20}
        shah_sc_lt = {"window_size": 20, "k_neighbours": 3}
        kobitski_gt = {"window_size": 10}
        shah_gt = {"window_size": 10, "k_neighbours": 7}
        defaults = {"window_size": 10, "k_neighbours": 5}
        result = self._run(kobitski_sc, shah_sc_align, shah_sc_lt,
                           kobitski_gt, shah_gt, defaults)
        assert result["window_size"] == 15.0

    def test_lt_params_averaged_across_both_regimes(self):
        # selfcal: shah_sc_lt k_neighbours=3 → selfcal_merged k_neighbours=3
        # gt: shah_gt k_neighbours=7 → gt_merged k_neighbours=7
        # combined: (3+7)/2 = 5.0
        kobitski_sc = {}
        shah_sc_align = {}
        shah_sc_lt = {"k_neighbours": 3}
        kobitski_gt = {}
        shah_gt = {"k_neighbours": 7}
        defaults = {"k_neighbours": 5}
        result = self._run(kobitski_sc, shah_sc_align, shah_sc_lt,
                           kobitski_gt, shah_gt, defaults)
        assert result["k_neighbours"] == 5.0

    def test_categorical_agreement_passes_through(self):
        kobitski_sc = {"cpd_penalty": "rigid"}
        shah_sc_align = {"cpd_penalty": "rigid"}
        shah_sc_lt = {"cpd_penalty": "rigid"}
        kobitski_gt = {"cpd_penalty": "rigid"}
        shah_gt = {"cpd_penalty": "rigid"}
        defaults = {"cpd_penalty": "rigid"}
        result = self._run(kobitski_sc, shah_sc_align, shah_sc_lt,
                           kobitski_gt, shah_gt, defaults)
        assert result["cpd_penalty"] == "rigid"

    def test_categorical_conflict_across_regimes_falls_back_to_default(self):
        # selfcal converges to "euclidean", gt converges to "cosine"
        kobitski_sc = {"dtw_dist_fn": "euclidean"}
        shah_sc_align = {"dtw_dist_fn": "euclidean"}
        shah_sc_lt = {"dtw_dist_fn": "euclidean"}
        kobitski_gt = {"dtw_dist_fn": "cosine"}
        shah_gt = {"dtw_dist_fn": "cosine"}
        defaults = {"dtw_dist_fn": "euclidean"}
        result = self._run(kobitski_sc, shah_sc_align, shah_sc_lt,
                           kobitski_gt, shah_gt, defaults)
        assert result["dtw_dist_fn"] == "euclidean"

    def test_defaults_fill_uncalibrated_keys(self):
        defaults = {"window_size": 10, "step": 8, "k_neighbours": 5}
        result = self._run({}, {}, {}, {}, {}, defaults)
        assert result["step"] == 8

    def test_defaults_only_keys_unchanged_by_averaging(self):
        # A key that appears only in defaults (not calibrated by any HPO run)
        # should remain at its default value, not be affected by averaging.
        defaults = {"step": 8, "window_size": 10}
        kobitski_sc = {"window_size": 10}
        shah_sc_align = {"window_size": 10}
        shah_sc_lt = {}
        kobitski_gt = {"window_size": 10}
        shah_gt = {}
        result = self._run(kobitski_sc, shah_sc_align, shah_sc_lt,
                           kobitski_gt, shah_gt, defaults)
        assert result["step"] == 8
