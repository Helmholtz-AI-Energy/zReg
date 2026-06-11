---
phase: 27-datafactory-geometric-augmentation-methods
verified: 2026-06-11T15:30:00Z
status: passed
score: 14/14 must-haves verified
overrides_applied: 0
re_verification: null
gaps: []
deferred: []
human_verification: []
---

# Phase 27: DataFactory Geometric Augmentation Methods Verification Report

**Phase Goal:** Add four standalone augmentation methods to DataFactory in eval/data_factory.py: scale(dataset, factor), rotate(dataset, rotation_matrix), drop_points(dataset, fraction, seed), and sample_new_points(dataset, n_extra, seed). Extend augment() dict dispatch to recognise new keys: "scale_factor", "dropout_fraction", "rotation_deg" (+ optional "rotation_axis"), "n_new_points" — applied in a fixed order after existing noise/outlier steps. All methods have immutable contract (input never mutated). Tests green.
**Verified:** 2026-06-11T15:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                            | Status     | Evidence                                                                                              |
|----|--------------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------------------|
| 1  | DataFactory.scale(dataset, factor) multiplies every frame's pos by factor and returns a new dict | VERIFIED | eval/data_factory.py line 312: `def scale`; `pos=pc["pos"] * factor` at line 339; TestScale 4 tests PASS |
| 2  | DataFactory.rotate(dataset, R) delegates to apply_rigid and returns a deep-copied dict           | VERIFIED | eval/data_factory.py line 347: `def rotate`; `return apply_rigid(dataset, tf)` at line 375; RigidTransformation wrapping at lines 370-374; TestRotate 4 tests PASS |
| 3  | DataFactory.drop_points(dataset, fraction, seed) keeps round(n*(1-fraction)) points per frame, preserving id/color/fps-idx | VERIFIED | eval/data_factory.py line 377: `def drop_points`; `torch.manual_seed(seed)` line 409; `idx = torch.randperm(...)[:keep].sort().values` line 414; all fields indexed at lines 415-420; TestDropPoints 5 tests PASS |
| 4  | DataFactory.sample_new_points(dataset, n_extra, seed) appends n_extra uniform-in-bbox points per frame | VERIFIED | eval/data_factory.py line 423: `def sample_new_points`; `bbox_min` at line 462; `_extend` helper at lines 468-474; fps-idx extended at line 481; TestSampleNewPoints 5 tests PASS |
| 5  | All four methods return new dicts; input dataset is never mutated                                | VERIFIED | scale builds `result: dict[int, zRegPointCloud] = {}`; drop_points and sample_new_points build new dicts; rotate uses apply_rigid which internally deep-copies; TestScale::test_input_not_mutated, TestDropPoints::test_input_not_mutated, TestSampleNewPoints::test_input_not_mutated all PASS |
| 6  | TestScale, TestRotate, TestDropPoints, TestSampleNewPoints test classes pass                     | VERIFIED | pytest output: 26 passed (all 4 new test classes); individual class results all PASS |
| 7  | augment({'scale_factor': 1.2}) calls self.scale and returns pos multiplied by 1.2               | VERIFIED | eval/data_factory.py lines 204-205: `if "scale_factor" in params: result = self.scale(result, params["scale_factor"])`; TestAugmentExtended::test_scale_factor_key PASS |
| 8  | augment({'rotation_deg': 90}) builds a rotation matrix via Rodrigues and calls self.rotate       | VERIFIED | eval/data_factory.py lines 207-220: Rodrigues formula inline; `result = self.rotate(result, R)` at line 220; TestAugmentExtended::test_rotation_deg_zero_noop PASS |
| 9  | augment({'dropout_fraction': 0.3}) calls self.drop_points and reduces point count by ~30%       | VERIFIED | eval/data_factory.py lines 222-223: `if "dropout_fraction" in params: result = self.drop_points(...)`; TestAugmentExtended::test_dropout_fraction_key PASS |
| 10 | augment({'n_new_points': 10}) calls self.sample_new_points and increases point count by 10 per frame | VERIFIED | eval/data_factory.py lines 225-226: `if "n_new_points" in params: result = self.sample_new_points(...)`; TestAugmentExtended::test_n_new_points_key PASS |
| 11 | augment({'scale_factor': 1.2, 'dropout_fraction': 0.1}) chains scale then dropout in that order | VERIFIED | Dispatch order in code: scale_factor (step 3) before dropout_fraction (step 5); TestAugmentExtended::test_scale_then_dropout_chain PASS |
| 12 | augment({'sigma': 0.01, 'n_outliers': 5}) still works correctly (existing steps unaffected)     | VERIFIED | Steps 1 and 2 unchanged; TestAugmentExtended::test_existing_sigma_unaffected and test_existing_outliers_unaffected PASS |
| 13 | augment({}) is still a no-op (existing behavior preserved)                                       | VERIFIED | TestAugment::test_empty_params_noop PASS; pre-existing tests unaffected |
| 14 | TestAugmentExtended test class passes                                                            | VERIFIED | pytest: 8/8 tests in TestAugmentExtended PASS |

**Score:** 14/14 truths verified

### Required Artifacts

| Artifact                        | Expected                                                             | Status     | Details                                                                                          |
|---------------------------------|----------------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------|
| `eval/data_factory.py`          | DataFactory with scale, rotate, drop_points, sample_new_points       | VERIFIED   | All four methods defined at lines 312, 347, 377, 423                                            |
| `eval/data_factory.py`          | rotate method delegating to apply_rigid                              | VERIFIED   | `return apply_rigid(dataset, tf)` at line 375 with RigidTransformation wrapper                  |
| `eval/data_factory.py`          | drop_points method using torch.randperm                              | VERIFIED   | `torch.randperm(n, device=pc["pos"].device)[:keep].sort().values` at line 414                   |
| `eval/data_factory.py`          | sample_new_points method with per-frame bbox sampling                | VERIFIED   | `bbox_min = pos.min(dim=0).values` at line 462; `new_pts = bbox_min + rand * (bbox_max - bbox_min)` |
| `eval/data_factory.py`          | augment() with 6-step dispatch containing "scale_factor"             | VERIFIED   | `scale_factor` dispatch at lines 204-205; docstring lists all 6 keys with dispatch order        |
| `eval/data_factory.py`          | Rodrigues rotation formula inline in augment()                       | VERIFIED   | Lines 207-220; references "Rodrigues' formula" in comment at line 214                           |
| `tests/test_data_factory.py`    | Unit test classes TestScale, TestRotate, TestDropPoints, TestSampleNewPoints | VERIFIED | All four classes found at lines 326, 385, 445, 511                                   |
| `tests/test_data_factory.py`    | TestAugmentExtended integration test class                           | VERIFIED   | Class found at line 583 with 8 test methods                                                     |

### Key Link Verification

| From                               | To                 | Via                                                              | Status   | Details                                                               |
|------------------------------------|--------------------|------------------------------------------------------------------|----------|-----------------------------------------------------------------------|
| DataFactory.rotate                 | apply_rigid        | RigidTransformation(rot=rotation_matrix, t=zeros(3), scale=1.0) | WIRED    | Lines 370-375: tf constructed and passed to apply_rigid               |
| DataFactory.drop_points            | torch.randperm     | torch.randperm(n, device=pc["pos"].device)[:keep].sort().values  | WIRED    | Line 414 matches pattern exactly                                      |
| DataFactory.sample_new_points      | per-frame bbox     | pos.min(dim=0).values / pos.max(dim=0).values                   | WIRED    | Lines 462-463: bbox_min and bbox_max computed per frame               |
| augment() scale_factor branch      | self.scale         | result = self.scale(result, params['scale_factor'])              | WIRED    | Line 205 matches pattern exactly                                      |
| augment() rotation_deg branch      | self.rotate        | Rodrigues formula → R tensor → self.rotate(result, R)           | WIRED    | Lines 207-220: Rodrigues, then `result = self.rotate(result, R)`     |
| augment() dropout_fraction branch  | self.drop_points   | result = self.drop_points(result, params['dropout_fraction'])    | WIRED    | Line 223 matches pattern exactly                                      |
| augment() n_new_points branch      | self.sample_new_points | result = self.sample_new_points(result, params['n_new_points']) | WIRED  | Line 226 matches pattern exactly                                      |

### Data-Flow Trace (Level 4)

The four new methods are pure tensor transforms with no dynamic data sources (no DB queries, no fetches). All data flows from the caller-supplied `dataset` dict through deterministic in-memory tensor operations. Level 4 tracing is not applicable to pure computational transforms.

### Behavioral Spot-Checks

| Behavior                         | Command                                                              | Result          | Status |
|----------------------------------|----------------------------------------------------------------------|-----------------|--------|
| Import DataFactory cleanly       | `python -c "from eval.data_factory import DataFactory; print('OK')"` | `import OK`     | PASS   |
| TestScale (4 tests)              | pytest TestScale -x -q                                               | 4 passed        | PASS   |
| TestRotate (4 tests)             | pytest TestRotate -x -q                                              | 4 passed        | PASS   |
| TestDropPoints (5 tests)         | pytest TestDropPoints -x -q                                          | 5 passed        | PASS   |
| TestSampleNewPoints (5 tests)    | pytest TestSampleNewPoints -x -q                                     | 5 passed        | PASS   |
| TestAugmentExtended (8 tests)    | pytest TestAugmentExtended -x -q                                     | 8 passed        | PASS   |
| Full test suite                  | pytest tests/ -q                                                     | 826 passed, 18 skipped, 0 errors | PASS |

### Probe Execution

No probe scripts declared in PLAN frontmatter. No `scripts/*/tests/probe-*.sh` found for this phase. Step 7c: SKIPPED (no declared probes).

### Requirements Coverage

| Requirement | Source Plan | Description                                                                    | Status    | Evidence                                                        |
|-------------|-------------|--------------------------------------------------------------------------------|-----------|-----------------------------------------------------------------|
| DF-01       | 27-01, 27-02 | Four standalone augmentation methods + augment() dispatch for new keys        | SATISFIED | All four methods implemented; augment() 6-step dispatch verified; 26 new tests green |

**Note:** DF-01 is referenced in ROADMAP.md (Phase 27 Requirements field) but is not listed in REQUIREMENTS.md. The requirement ID appears only in the ROADMAP phase entry. This is a pre-existing documentation gap in the project — the REQUIREMENTS.md file covers up through EXT-02 at its current extent, and newer requirement IDs (DF-01, DF-02, VIZ-01, EXT-03) from later phases are only in ROADMAP. This gap is informational; the requirement itself is fully satisfied.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

Scanned `eval/data_factory.py` and `tests/test_data_factory.py` for: `TBD`, `FIXME`, `XXX` (blockers), `TODO`, `HACK`, `PLACEHOLDER` (warnings), `return null/{}`, hardcoded empty data. No markers or stub patterns detected.

The one note from SUMMARY 27-01 — that `grep -c "torch.manual_seed" eval/data_factory.py` returns 3 not 2 because the docstring of `sample_new_points` contains the string — was investigated. The docstring occurrence is at line 433 and is genuinely documentation prose. The two actual code calls are at lines 409 (`drop_points`) and 458 (`sample_new_points`). This is not a stub; the acceptance criterion is satisfied in substance.

### Human Verification Required

None. All observable behaviors are verifiable programmatically. The test suite provides complete coverage of the four methods' contracts including immutability, rotation correctness, bbox containment, sentinel fill, and multi-key composition ordering.

### Gaps Summary

No gaps. All 14 must-have truths verified, all 8 artifacts verified, all 7 key links wired, 826 tests pass with 0 failures, no debt markers in modified files.

---

_Verified: 2026-06-11T15:30:00Z_
_Verifier: Claude (gsd-verifier)_
