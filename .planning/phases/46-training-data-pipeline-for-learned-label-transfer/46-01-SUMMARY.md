---
phase: 46-training-data-pipeline-for-learned-label-transfer
plan: 01
subsystem: data
tags: [numpy, torch, synthetic-data-generation, rejection-sampling]

# Dependency graph
requires: []
provides:
  - sample_ball(n_points, seed=42, radius=1.0) -> torch.Tensor single-frame solid-ball sampler
  - sample_bowl(n_points, seed=42, radius=1.0, d_ratio=0.5) -> torch.Tensor single-frame lower-hemisphere-shell sampler
  - both exported from zreg.generators package
affects: [46-02, 46-03, 47]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "numpy-native RNG seed contract for geometry samplers (np.random.default_rng(seed)) as a documented deviation from generate_trajectory's torch.manual_seed contract"

key-files:
  created: []
  modified:
    - src/zreg/generators/generators.py
    - src/zreg/generators/__init__.py
    - tests/test_generators.py

key-decisions:
  - "Ported sample_ball_shell/in_bowl/sample_bowl_frame math from scripts/generate_datasets.py verbatim, simplified to a single fixed radius (R_inner=0.0 for ball, no per-frame growth) since Phase 46 only needs one frame per seed (D-01/D-02)"
  - "Both samplers use numpy's default_rng instead of torch.manual_seed because the geometry math (inverse-CDF, rejection sampling) is numpy-native; the seed=None nondeterministic semantic is preserved and documented as an explicit deviation from generate_trajectory's contract"
  - "Renamed test methods to include the literal 'sample_ball'/'sample_bowl' substring so the plan's -k filter verify commands actually select the new tests (the original method names like test_shape_and_dtype did not match -k sample_ball since TestSampleBall has no underscore)"

patterns-established:
  - "Single-frame, non-growth geometry primitives added alongside generate_trajectory in zreg/generators/generators.py, returning bare (n_points, 3) float32 tensors (no zRegPointCloud wrapping, no labels/ids) — caller composes labels/wrapping downstream"

requirements-completed: [D-04]

# Metrics
duration: 10min
completed: 2026-07-11
---

# Phase 46 Plan 01: Bowl/Ball Geometry Samplers Summary

**Ported bowl/ball rejection-sampling geometry from `scripts/generate_datasets.py` into reusable `sample_ball()`/`sample_bowl()` single-frame samplers in `zreg.generators`, resolving D-04's shapeless-Gaussian-blob problem.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-07-11T10:05:02Z
- **Completed:** 2026-07-11T10:12:58Z
- **Tasks:** 2 completed
- **Files modified:** 3

## Accomplishments
- `sample_ball(n_points, seed=42, radius=1.0)` — volume-correct uniform sampling inside a solid sphere, ported from `sample_ball_shell` (R_inner=0.0)
- `sample_bowl(n_points, seed=42, radius=1.0, d_ratio=0.5)` — rejection-sampled lower-hemisphere bowl shell, ported from `in_bowl`/`sample_bowl_frame`
- Both exported from `zreg.generators` package (`from zreg.generators import sample_ball, sample_bowl`)
- 19 new tests (`TestSampleBall` x6, `TestSampleBowl` x7... actually 13 total across both classes) covering shape/dtype, geometry containment, seed reproducibility/variety, and ValueError guards

## Task Commits

Each task followed the TDD RED → GREEN cycle with a separate commit per gate:

1. **Task 1: sample_ball** — `e74bbba` (test: RED), `271b322` (feat: GREEN)
2. **Task 2: sample_bowl + package export** — `5117299` (test: RED), `d1fdbdd` (feat: GREEN)

**Plan metadata:** (this commit, following SUMMARY.md write)

## Files Created/Modified
- `src/zreg/generators/generators.py` - added `sample_ball()`, `_sample_ball_shell()`, `sample_bowl()`, `_in_bowl()`; updated module docstring and `__all__`
- `src/zreg/generators/__init__.py` - imports and re-exports `sample_ball`, `sample_bowl`
- `tests/test_generators.py` - added `TestSampleBall` (6 tests) and `TestSampleBowl` (7 tests, including the package-export check)

## Decisions Made
- Simplified the ported math to a single fixed radius per call (no growth-over-frames machinery) — matches D-01/D-02's single-frame-per-seed design; the full growth-model code stays untouched in `scripts/generate_datasets.py`.
- Kept the numpy-native RNG contract (`np.random.default_rng(seed)`) rather than forcing the geometry math through `torch.manual_seed`, since rejection/inverse-CDF sampling is naturally numpy-based; documented this explicitly in both function docstrings and the module docstring so future readers don't mistake it for an inconsistency.
- `id` is never populated by either sampler (`id=None` is the caller's responsibility, consistent with `generate_trajectory`'s existing contract) — this is a structural mitigation for the `label`-vs-`id` confusion flagged in 45-DESIGN.md/46-RESEARCH.md Pitfall 4, since there's no `id` field in this phase's own output to accidentally read.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Renamed test methods so plan's `-k` verify filters actually select the new tests**
- **Found during:** Task 1, immediately after writing the RED test class `TestSampleBall` with method names like `test_shape_and_dtype`
- **Issue:** The plan's acceptance criterion `python -m pytest tests/test_generators.py -k sample_ball -x -q` relies on pytest's `-k` substring filter matching against test node IDs. `TestSampleBall` (no underscore) does not contain the literal substring `sample_ball`, and generic method names like `test_shape_and_dtype` don't either — so `-k sample_ball` silently selected 0 tests and reported success without ever running them (pytest exits 0 on an empty `-k` selection given this project's `addopts` config).
- **Fix:** Renamed all `TestSampleBall`/`TestSampleBowl` test methods to include the literal `sample_ball`/`sample_bowl` substring (e.g. `test_sample_ball_shape_and_dtype`), matching the same naming convention already used in `TestGeneratorsCoverageGaps`. Re-ran the exact plan verify commands to confirm they now select and pass the intended tests (6 for Task 1, 13 for Task 2).
- **Files modified:** `tests/test_generators.py`
- **Verification:** `pytest tests/test_generators.py -k sample_ball -x -q` now collects 6 tests (was 0); `pytest tests/test_generators.py -k "sample_bowl or sample_ball" -x -q` now collects 13 (was 0 due to import error pre-implementation, then would have silently collected 0 post-implementation without this fix)
- **Committed in:** `271b322` (Task 1 feat commit, includes the rename) and `5117299` (Task 2 test commit, new methods named correctly from the start)

---

**Total deviations:** 1 auto-fixed (1 blocking - test naming so verify commands are meaningful, not silently vacuous)
**Impact on plan:** No scope creep — purely a test-naming fix so the plan's own acceptance criteria commands actually exercise the code they claim to verify.

## Issues Encountered
- `python -c "..."` one-liner acceptance checks initially failed with `ImportError` because the environment's editable `zReg` pip install resolves to the main repo's `src/` (not this worktree's `src/`), while pytest correctly prioritizes the worktree's own `src/` via `tests/conftest.py`'s `sys.path.insert(0, ...)`. Resolved by running the one-liner checks with `PYTHONPATH=src` prefixed, matching what conftest.py does for the test suite. This is an environment/tooling quirk, not a code issue — no source changes were needed.

## Next Phase Readiness
- `sample_ball()`/`sample_bowl()` are ready for Plan 46-03's training-triple pipeline: `generate_labels()` can wrap either sampler's output in a `zRegPointCloud` and assign Voronoi categorical labels, then `DataFactory.generate_target()` builds the corresponding target per Pattern 1 in `46-RESEARCH.md`.
- No blockers. Full regression suite green: 1218 passed, 18 skipped, 1 xpassed (was 1205 passed at Phase 45 close; +13 new tests, 0 regressions).

---
*Phase: 46-training-data-pipeline-for-learned-label-transfer*
*Completed: 2026-07-11*

## Self-Check: PASSED

All created/modified files found on disk; all 4 task commit hashes (e74bbba, 271b322, 5117299, d1fdbdd) found in git log.
