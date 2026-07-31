---
phase: 56-configurable-multi-label-region-based-labeling-rework-genera
plan: 01
subsystem: data-generation
tags: [pydantic, torch, labels, logsumexp, synthetic-data]

# Dependency graph
requires:
  - phase: 55-spherical-cap-and-gaussian-label-generators-for-zreg-data-ge
    provides: "_angular_distance_deg helper and the cone-shape geometry (assign_cap_labels/assign_gaussian_labels) this plan generalizes"
provides:
  - "LabelComponentSpec/LabelSpec pydantic contract types for per-shape region components and multi-component labels"
  - "_component_score(pos, component, mode) — per-shape (voronoi/blob/cone) scoring function"
  - "_label_scores(pos, label_spec, mode) — weighted logsumexp mixture aggregation across a label's components"
affects: [56-02-assignment-modes-and-orchestrator, 56-03-config-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns: ["pydantic ConfigDict(extra=forbid) nested spec models", "logsumexp weighted mixture scoring"]

key-files:
  created: []
  modified: [src/zreg/data_generation/labels.py]

key-decisions:
  - "weight resolved as a relative multiplier into log-space (log(weight) added to each component's score before logsumexp), not a normalized/softmax prior — left as Claude's Discretion per 56-CONTEXT.md"
  - "Voronoi deterministic score is unscaled -dist_sq (no division), since squaring/scaling both preserve argmax ordering for single-component 'nearest center wins' semantics"

patterns-established:
  - "Per-shape score functions return higher-is-better log-space quantities so shapes can be freely mixed via logsumexp regardless of their native scale (Euclidean vs angular)"

requirements-completed: [D-08, D-09, D-10, D-11]

# Metrics
duration: ~15min
completed: 2026-07-31
---

# Phase 56 Plan 01: Label Component/Spec Contracts and Scoring Math Summary

**Added `LabelComponentSpec`/`LabelSpec` pydantic models plus `_component_score`/`_label_scores` scoring primitives to `labels.py`, generalizing the Phase 55 cone geometry and Voronoi nearest-center logic into a shape-agnostic, mixture-capable scoring layer.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-07-31T11:49:43+02:00 (immediately after plan-checker fix commit)
- **Completed:** 2026-07-31T12:03:40+02:00
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- `LabelComponentSpec` and `LabelSpec` pydantic models define the vocabulary for region components (voronoi/blob/cone) and multi-component labels, with `extra="forbid"` and validators for `center` length, `weight > 0`, and shape-conditional `sigma > 0`.
- `_component_score` implements all three shapes' per-point scoring math: voronoi (deterministic unscaled negative squared distance; probabilistic negative squared distance over temperature), blob (Euclidean isotropic Gaussian log-density), cone (angular Gaussian log-density reusing `_angular_distance_deg`, generalized from a single hardcoded pole to per-component pole/sigma).
- `_label_scores` combines a label's component scores into one per-point score via `torch.logsumexp(score + log(weight), dim=1)`, the weighted mixture aggregation from D-11.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define LabelComponentSpec and LabelSpec pydantic models** - `6d04bfa` (feat)
2. **Task 2: Implement per-shape scoring and mixture aggregation** - `18f40b0` (feat)

## Files Created/Modified
- `src/zreg/data_generation/labels.py` - added `LabelComponentSpec`, `LabelSpec`, `_component_score`, `_label_scores`; `generate_labels`/`assign_cap_labels`/`assign_gaussian_labels` untouched (Plan 56-02 rewrites/deletes them)

## Decisions Made
- `weight` is treated as a relative log-space multiplier (`+ math.log(c.weight)` before `logsumexp`), not required to sum to 1 across a label's components — matches D-11's "mixing coefficient/prior" language without forcing softmax normalization, per Claude's Discretion in 56-CONTEXT.md.
- Voronoi's deterministic-mode score returns unscaled `-dist_sq` rather than a distance (not squared) or a normalized probability — this exactly preserves `argmin`-equivalent ordering for "nearest center wins" while staying in the same log-space units as blob/cone scores for potential future mixing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's own Task 2 verify script contains a symmetric-distance test bug (not an implementation bug)**
- **Found during:** Task 2 verification
- **Issue:** The plan's inline verify script asserts `_component_score(pos_far, voronoi, 'deterministic').item() > _component_score(pos_near, voronoi, 'deterministic').item()` with `pos_far=[10,0,0]`, `pos_near=[0,0,0]`, and `voronoi.center=[5,0,0]` — both points are exactly 5 units from the center (equidistant), so the assertion `>` fails for any correct implementation (scores are equal, not strictly greater). This is a bug in the plan's verify script's test data, not in `_component_score`.
- **Fix:** Verified the actual invariant ("nearer center scores higher") with genuinely asymmetric points instead (`center=[5,0,0]`, near point `[4,0,0]` at distance 1, far point `[10,0,0]` at distance 5) — confirmed `_component_score` returns a strictly higher score for the nearer point, proving the implementation is correct. Did not modify `56-01-PLAN.md` (plan documents are not rewritten by the executor); all other assertions in the plan's verify scripts (Task 1 in full, Task 2's blob/cone/probabilistic-voronoi/mixture/empty-frame checks) ran unmodified and passed.
- **Files modified:** none (verification-only; production code unchanged)
- **Verification:** Ran the corrected assertion inline; also ran the full existing regression suite (`tests/test_generators.py`, 65 tests) with no regressions.
- **Committed in:** N/A (test-script-only observation, not a code change)

---

**Total deviations:** 1 auto-fixed (1 bug, in the plan's own verify script data, not in implementation)
**Impact on plan:** No scope creep — `_component_score`/`_label_scores` implementation matches the plan's action block and acceptance criteria exactly; only the ad-hoc verify script's specific numeric test case needed substituting with a non-degenerate example to actually exercise the invariant it claims to test.

## Issues Encountered
- Local environment requires `KMP_DUPLICATE_LIB_OK=TRUE` to avoid an `OMP: Error #15` libomp double-init crash when importing `torch` via the bare `python -c` verify commands on this machine — set for all verification runs in this plan; this is a pre-existing local environment quirk, not something introduced by this plan's code.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `LabelComponentSpec`, `LabelSpec`, `_component_score`, `_label_scores` are ready for Plan 56-02's `_assign_deterministic`/`_assign_probabilistic` and the reworked `generate_labels()` orchestrator to consume directly.
- Neither new model nor scoring function is yet exported via `__all__` or `zreg/data_generation/__init__.py` — this wiring is explicitly deferred to Plan 56-02 per this plan's scope boundary.
- No blockers.

---
*Phase: 56-configurable-multi-label-region-based-labeling-rework-genera*
*Completed: 2026-07-31*

## Self-Check: PASSED

- FOUND: src/zreg/data_generation/labels.py
- FOUND: .planning/phases/56-configurable-multi-label-region-based-labeling-rework-genera/56-01-SUMMARY.md
- FOUND: 6d04bfa (Task 1 commit)
- FOUND: 18f40b0 (Task 2 commit)
