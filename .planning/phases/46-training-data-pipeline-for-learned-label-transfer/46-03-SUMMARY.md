---
phase: 46-training-data-pipeline-for-learned-label-transfer
plan: 03
subsystem: data
tags: [torch, pydantic, synthetic-data-generation, training-data, seed-split]

# Dependency graph
requires:
  - phase: 46-01
    provides: sample_ball()/sample_bowl() single-frame samplers in zreg.generators
  - phase: 46-02
    provides: TrainingTriple frozen pydantic model in eval/types.py
provides:
  - "split_seeds(n_train, n_val, base_seed=0) module function in eval/data_factory.py — disjoint-by-construction train/val seed ranges (D-03)"
  - "DataFactory.generate_training_triple(seed, n_classes=6, shape=None, n_points=None) -> TrainingTriple"
  - "DataFactory.generate_training_set(seeds, n_classes=6) -> list[TrainingTriple]"
  - "augment() per-seed 'augment_seed' RNG override threaded to all four stochastic sub-calls (Pitfall 3 resolved)"
  - "tests/test_data_factory_training_triples.py — Wave-0 test file, 7 classes, 22 tests"
affects: [47]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "New DataFactory methods that intentionally do NOT use the self._synthetic_dataset/_synthetic_target/_source_dataset singleton-cache pattern (Pattern 2, Pitfall 2) — documented explicitly in generate_training_triple's docstring so a future reader doesn't 'fix' it into caching"
    - "Seed-level train/val split via disjoint-by-construction integer ranges (Pattern 3), deliberately distinct from prepare_split()'s frame-level random.sample mechanism"

key-files:
  created: []
  modified:
    - eval/data_factory.py
    - tests/test_data_factory_training_triples.py

key-decisions:
  - "Geometry alternates by seed parity (even=ball, odd=bowl) when shape is not explicitly given, per the plan's documented rule — callers/tests derive geometry from seed % 2 rather than an internal attribute"
  - "transform_spec for generate_training_triple excludes 'n_new_points' — points it appends get a -1 label sentinel, invalid for supervised target labels (46-RESEARCH.md Anti-Patterns)"
  - "augment_seed (from Task 1) is threaded into every training triple's transform_spec as the seed itself, so noise/dropout diversity is real per-seed variation, not just per-seed-scaled magnitude"
  - "Added 3 extra tests (explicit n_points/shape override, unknown-shape ValueError) beyond the plan's literal behavior list to close a coverage gap discovered during full-suite verification and restore this project's established 100%-coverage-on-eval/zreg convention"

patterns-established:
  - "generate_training_set(seeds) as the batching companion to split_seeds() — Phase 47's training loop calls generate_training_set(train_seeds) / generate_training_set(val_seeds)"

requirements-completed: [D-01, D-02, D-03, D-04]

# Metrics
duration: ~35min (across two sessions, split by a session-limit interruption between Task 1 and Task 2)
completed: 2026-07-11
---

# Phase 46 Plan 3: Seed-Driven Training-Triple Pipeline Summary

**`DataFactory.generate_training_triple(seed)` composes `sample_ball`/`sample_bowl` + `generate_labels` + `generate_target` into a reproducible, seed-varied `(labeled source, transformed target)` pair; `split_seeds()` gives disjoint train/val seed ranges; `augment()` now threads a genuine per-seed RNG override.**

## Performance

- **Duration:** ~35 min total active work, split across two sessions (a prior session completed Task 1 and stopped mid-Task-2 at a session limit; this session resumed from the committed Task 1 state, committed Task 2's already-written RED tests, implemented GREEN, and finalized Task 3).
- **Started:** 2026-07-11 (Task 1, prior session)
- **Completed:** 2026-07-11T18:36:13+02:00
- **Tasks:** 3 completed
- **Files modified:** 2

## Accomplishments
- `augment()` forwards a caller-supplied `"augment_seed"` (default 42) to `add_gaussian_noise`, `add_outliers`, `drop_points`, and `sample_new_points` — different seeds now produce genuinely different noise/dropout patterns (not just magnitude-scaled); omitting the key preserves byte-identical legacy behavior.
- `split_seeds(n_train, n_val, base_seed=0)` — disjoint-by-construction train/val seed ranges (D-03), with `ValueError` guards on `n_train < 1` / `n_val < 0`.
- `DataFactory.generate_training_triple(seed, n_classes=6, shape=None, n_points=None)` — produces one `TrainingTriple` per seed: a 100-300-point labeled ball-or-bowl source cloud (D-01) and a rigid+scale+dropout-transformed, correctly-labeled target (labels propagate for free through `generate_target`). Same seed is bitwise-reproducible; different seeds are genuinely distinct (D-02); geometry alternates ball/bowl by seed parity (D-04). Deliberately bypasses the `self._synthetic_dataset` singleton cache (Pitfall 2).
- `DataFactory.generate_training_set(seeds, n_classes=6)` — batches `generate_training_triple` over an iterable (e.g. a `split_seeds()` range) into a `list[TrainingTriple]`.
- `tests/test_data_factory_training_triples.py` — the Wave-0 test file mandated by `45-DESIGN.md`, with all 7 required classes (`TestAugmentSeedThreading`, `TestSeedSplit`, `TestPointCountRegime`, `TestSeedReproducibilityAndVariety`, `TestLabelVsIdDiscipline`, `TestNoFalseCaching`, `TestGeometryCoverage`) and 22 tests total.

## Task Commits

Each task followed the TDD RED → GREEN cycle with a separate commit per gate:

1. **Task 1: Thread augment_seed through augment()** — `973e0fd` (test: RED), `91bd49f` (feat: GREEN) — completed in a prior session.
2. **Task 2: split_seeds() / generate_training_triple() / generate_training_set()** — `2117f31` (test: RED, tests were already written in the prior session but uncommitted at the session-limit cutoff), `772159d` (feat: GREEN, implemented this session).
3. **Task 3: Finalize the Wave-0 test file** — the five remaining mandated classes (`TestSeedSplit`/`TestPointCountRegime`/`TestSeedReproducibilityAndVariety` from Task 2, plus `TestLabelVsIdDiscipline`/`TestNoFalseCaching`/`TestGeometryCoverage`) were all already authored in the same uncommitted file alongside Task 2's tests — no separate authoring step was needed. This session added 3 further tests (`bc0dc74`, test) to close a coverage gap found during full-suite verification.

**Plan metadata:** (this commit, following SUMMARY.md write)

## Files Created/Modified
- `eval/data_factory.py` — added module-level `split_seeds()` (added to `__all__`); added `DataFactory.generate_training_triple()` and `DataFactory.generate_training_set()`; `augment()` now reads `augment_seed = params.get("augment_seed", 42)` and forwards it to all four stochastic sub-calls; new imports `sample_ball`, `sample_bowl` from `zreg.generators`, `TrainingTriple` from `eval.types`, `generate_labels` promoted from an unused `# noqa: F401` import to an active one.
- `tests/test_data_factory_training_triples.py` — new Wave-0 test file: `TestAugmentSeedThreading` (4 tests), `TestSeedSplit` (4 tests), `TestPointCountRegime` (4 tests, including the 2 override tests + 1 ValueError test added this session), `TestSeedReproducibilityAndVariety` (3 tests), `TestLabelVsIdDiscipline` (2 tests), `TestNoFalseCaching` (1 test), `TestGeometryCoverage` (1 test) — 22 tests total (was 16 at the RED/GREEN gate; +3 coverage-closing tests, +... see below).

## Decisions Made
- Followed the plan's exact recommended skeleton from `46-RESEARCH.md`'s "Recommended DataFactory new method skeleton" and Pattern 1/2/3 verbatim — no architectural deviation.
- Geometry selection rule (`seed % 2 == 0` → ball, else bowl) is documented in the method docstring so tests can assert against the documented rule rather than reading an internal attribute, per the plan's explicit instruction.
- Excluded `"n_new_points"` from the training triple's `transform_spec`, matching 46-RESEARCH.md's Anti-Patterns section (new points get a `-1` label sentinel, invalid as a supervised target).
- Threaded `augment_seed=seed` into every triple's `transform_spec` so Task 1's per-seed RNG threading actually contributes real per-seed diversity to the training data, not just deterministic geometric-parameter variation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added 3 tests to close a coverage gap on `generate_training_triple`'s shape-dispatch branch**
- **Found during:** Task 3, full-suite regression verification (`python -m pytest -q`)
- **Issue:** After Task 2's GREEN commit, `python -m pytest -q --cov-report=term-missing` showed `eval/data_factory.py` at 99% (238 statements, 1 missed; 86 branches, 3 missed) — specifically the `else: raise ValueError(...)` branch for an unrecognised `shape` value, and the two branch-partial paths where `n_points`/`shape` are explicitly supplied (not `None`) were never exercised, since every existing test called `generate_training_triple` with defaults only. This project's established convention (per `.planning/STATE.md` and every prior phase's SUMMARY) is 100% coverage on `zreg`/`eval`; leaving an unguarded/untested `ValueError` branch and two untested override code paths on a newly-added public method is a correctness/regression-safety gap for Phase 47's downstream training loop, which is expected to call this method with explicit `shape`/`n_points` overrides.
- **Fix:** Added `test_explicit_n_points_override`, `test_explicit_shape_override`, and `test_unknown_shape_raises_value_error` to `TestPointCountRegime` in `tests/test_data_factory_training_triples.py`.
- **Files modified:** `tests/test_data_factory_training_triples.py`
- **Verification:** `python -m pytest -q` — full suite coverage returned to 100% (`TOTAL 3773 0 1154 0 100%`); 1248 passed, 18 skipped, 1 xpassed (was 99% / 1245 passed before the fix).
- **Committed in:** `bc0dc74`

---

**Total deviations:** 1 auto-fixed (1 missing-critical — test coverage for an existing correctness guard, not new production code).
**Impact on plan:** No scope creep — pure test-coverage closure restoring an established project-wide convention; no production code was changed by this deviation.

## Issues Encountered
- The editable `zReg` pip install resolves to the main repo's `src/` rather than this worktree's `src/` when running bare `python -c "..."` one-liners (same environment quirk documented in `46-01-SUMMARY.md`). Resolved identically: prefixed acceptance-criteria one-liner checks with `PYTHONPATH=src`. `pytest` itself is unaffected (its own `sys.path` insertion via `tests/conftest.py` already prioritizes the worktree's `src/`). No source changes were needed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `split_seeds()`, `DataFactory.generate_training_triple()`, and `DataFactory.generate_training_set()` are ready for Phase 47's training loop: `train_seeds, val_seeds = split_seeds(200, 50)`; `train_set = factory.generate_training_set(train_seeds)`; `val_set = factory.generate_training_set(val_seeds)`.
- Phase 46 (all 3 plans: bowl/ball samplers, `TrainingTriple` model, seed-driven triple pipeline) is now fully complete.
- Full regression suite green: 1248 passed, 18 skipped, 1 xpassed, 100% coverage on `zreg`/`eval` (was 1229 at 46-02 close; +19 net new tests across this plan's 3 tasks: 4 augment_seed tests from Task 1, 15 split_seeds/generate_training_triple/generate_training_set/label-discipline/no-caching/geometry tests from Tasks 2-3).
- No blockers.

## TDD Gate Compliance

Both `tdd="true"` tasks followed the RED → GREEN gate sequence, verified in `git log`:
- Task 1: `973e0fd` (test, RED) → `91bd49f` (feat, GREEN).
- Task 2: `2117f31` (test, RED) → `772159d` (feat, GREEN).
Task 3 (plain `type="auto"`, no `tdd` attribute) required no separate RED/GREEN gate; its test-authoring obligation was satisfied by the tests already written alongside Task 2, finalized with a small coverage-closing `test` commit (`bc0dc74`).

---
*Phase: 46-training-data-pipeline-for-learned-label-transfer*
*Completed: 2026-07-11*

## Self-Check: PASSED
