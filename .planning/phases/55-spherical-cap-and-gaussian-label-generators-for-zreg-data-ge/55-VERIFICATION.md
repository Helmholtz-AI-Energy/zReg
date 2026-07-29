---
phase: 55-spherical-cap-and-gaussian-label-generators-for-zreg-data-ge
verified: 2026-07-30T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 55: Spherical-Cap and Gaussian Label Generators Verification Report

**Phase Goal:** zreg.data_generation.labels exposes assign_cap_labels (hard spherical-cap boundary) and assign_gaussian_labels (angle-dependent Bernoulli labels), both following the immutable deep-copy contract and exported from the package.
**Verified:** 2026-07-30
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | assign_cap_labels sets pc['label'] to label_inside for points within theta_deg of the pole and label_outside otherwise, per frame | VERIFIED | `torch.where(theta < theta_deg, label_inside, label_outside)` in labels.py:190; test_cap_pole_point_gets_label_inside and test_cap_equator_point_gets_label_outside pass |
| 2 | assign_gaussian_labels sets pc['label'] via a Bernoulli draw whose success probability exp(-theta^2/(2*sigma^2)) decreases with angular distance from the pole | VERIFIED | labels.py:251-253 computes `prob = torch.exp(-(theta**2) / (2.0 * sigma_deg**2))` then `torch.bernoulli(prob)`; test_gaussian_probability_decreases_with_angle passes (13/13 tests pass) |
| 3 | Both functions return a new dict and never mutate the input trajectory (deep-copy contract, D-03) | VERIFIED | Both functions call `copy.deepcopy(trajectory)` as first statement (labels.py:184, 245); test_cap_does_not_mutate_input and test_gaussian_does_not_mutate_input confirm input label is None after call |
| 4 | Both functions store pc['label'] as torch.long tensors of shape (N,) and handle empty frames (N==0) without special-casing | VERIFIED | `.to(torch.long)` applied at labels.py:191 and 254; no `if N==0` branch; test_cap_empty_frame and test_gaussian_empty_frame confirm (0,) torch.long output |
| 5 | Both functions normalise the pole internally so a non-unit-length pole yields identical results to its normalised form | VERIFIED | `_angular_distance_deg` normalises via `pole_vec / pole_vec.norm()` (labels.py:133); test_cap_pole_normalisation and test_gaussian_pole_normalisation confirm torch.equal results for pole=[0,0,1] vs pole=[0,0,5] |
| 6 | assign_cap_labels and assign_gaussian_labels are importable from zreg.data_generation | VERIFIED | Import command exits 0; `__init__.py` line 25: `from .labels import generate_labels, remove_labels, assign_cap_labels, assign_gaussian_labels`; both appear in `__all__` |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/zreg/data_generation/labels.py` | assign_cap_labels and assign_gaussian_labels public functions | VERIFIED | 2 function definitions confirmed (`grep -c` returned 2); private helper `_angular_distance_deg` present; `__all__` extended to include both names |
| `src/zreg/data_generation/__init__.py` | package-level export of both new functions | VERIFIED | Both functions on the `from .labels import` line and in `__all__`; module docstring updated to list both |
| `tests/test_generators.py` | test coverage for both new functions | VERIFIED | `TestSphericalLabelGenerators` class with 13 tests; all pass (0 deselected match failures) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/zreg/data_generation/__init__.py` | `src/zreg/data_generation/labels.py` | `from .labels import` | WIRED | Line 25 of `__init__.py` explicitly imports both functions |
| `tests/test_generators.py` | `src/zreg/data_generation/labels.py` | import + call | WIRED | Lines 27-28 import both; `TestSphericalLabelGenerators` calls both in 13 distinct tests |

### Data-Flow Trace (Level 4)

Not applicable. Both functions are pure computation utilities (no external data sources, DB queries, or API calls). Data flows from input `trajectory` arg through deep-copy, angular-distance computation, and label assignment entirely within the function body.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Both functions importable from package | `python -c "from zreg.data_generation import assign_cap_labels, assign_gaussian_labels; import inspect; print(inspect.signature(assign_cap_labels)); print(inspect.signature(assign_gaussian_labels))"` | Signatures printed, exit 0 | PASS |
| 2 function definitions in labels.py | `grep -c "def assign_cap_labels\|def assign_gaussian_labels" labels.py` | `2` | PASS |
| assign_cap_labels in __init__.py import and __all__ | `grep "assign_cap_labels" __init__.py` | 3 matching lines (docstring, import line, __all__ entry) | PASS |
| 13 new tests pass | `pytest test_generators.py -k "cap or gaussian_label or spherical or Spherical" -q` | `13 passed, 52 deselected in 0.74s` | PASS |

### Probe Execution

No probes declared in PLAN.md. Step 7c skipped.

### Requirements Coverage

No requirement IDs declared for this phase (requirements: [] in PLAN frontmatter).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns found |

Scan of `src/zreg/data_generation/labels.py` found zero instances of TBD, FIXME, XXX, TODO, HACK, PLACEHOLDER, or stub return patterns. The implementation is fully substantive.

### Human Verification Required

None. All must-haves are verifiable programmatically and all checks passed.

### Gaps Summary

No gaps. All 6 must-have truths are verified, all 3 required artifacts exist and are substantive and wired, both key links are confirmed, 13 tests pass with 0 regressions, and no anti-patterns or debt markers were found.

---

_Verified: 2026-07-30_
_Verifier: Claude (gsd-verifier)_
