---
phase: 56-configurable-multi-label-region-based-labeling-rework-genera
plan: 02
subsystem: data-generation
tags: [pydantic, torch, labels, softmax, multinomial, synthetic-data]

# Dependency graph
requires:
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 01)
    provides: "LabelComponentSpec/LabelSpec pydantic models and _component_score/_label_scores scoring primitives"
provides:
  - "_assign_deterministic/_assign_probabilistic assignment-mode functions"
  - "generate_labels() rewritten as the config-driven orchestrator (n_labels simple path + label_specs full path)"
  - "D-07 fix: region/component centers resolved exactly once per call, before the per-frame loop"
  - "Dead-clean removal of assign_cap_labels/assign_gaussian_labels and their 13 tests"
affects: [56-03-config-wiring, 56-04-comprehensive-tests, 56-05]

# Tech tracking
tech-stack:
  added: []
  patterns: ["torch.multinomial categorical sampling for probabilistic label assignment", "resolve-once-before-loop pattern for per-trajectory-stable RNG draws"]

key-files:
  created: []
  modified: [src/zreg/data_generation/labels.py, src/zreg/data_generation/__init__.py, tests/test_generators.py, eval/data_factory.py, eval/runners/optimizer.py, tests/test_benchmark_runner.py, tests/test_optimizer.py, tests/test_trajectory_export.py, tests/test_viz.py, tests/test_label_transfer_stage.py, tests/test_eval_runner.py]

key-decisions:
  - "n_labels path centers are drawn via a plain torch.randn(n_labels, 3) (no dtype/device pin) since LabelComponentSpec.center stores python floats via .tolist() and per-frame dtype/device casting happens inside _component_score"
  - "n_labels + mode='probabilistic' is rejected with a ValueError rather than silently defaulting a temperature - voronoi has no natural probability scale without an explicit temperature (D-10), and the simple path exposes no way to set one"
  - "DataFactory.generate_training_triple's own n_classes parameter name is left unchanged in this plan - only its internal generate_labels(...) call was updated to n_labels=n_classes; the public rename of n_classes on DataFactory/optimizer.py's tier docstrings is explicitly Plan 56-03's scope per D-04/D-13"

patterns-established:
  - "Region/component centers for the n_labels path are drawn once immediately after torch.manual_seed(seed), before the per-frame loop, and wrapped into resolved_specs = [LabelSpec(...)] so both the simple and full paths converge on the same per-frame scoring loop"

requirements-completed: [D-01, D-04, D-05, D-06, D-07, D-12]

# Metrics
duration: ~17min
completed: 2026-07-31
---

# Phase 56 Plan 02: Assignment Modes and generate_labels() Orchestrator Rework Summary

**Rewrote `generate_labels()` as the single config-driven entry point (n_labels/label_specs paths, deterministic/probabilistic modes) with the D-07 per-frame-center-redraw bug fixed, and deleted `assign_cap_labels`/`assign_gaussian_labels` (13 tests) as dead-clean removal.**

## Performance

- **Duration:** ~17 min
- **Started:** 2026-07-31T12:03:40+02:00 (immediately after Plan 56-01 completed)
- **Completed:** 2026-07-31T12:20:12+02:00
- **Tasks:** 2
- **Files modified:** 11 (3 in-scope per plan frontmatter + 8 call-site fixes from an auto-fixed regression, see Deviations)

## Accomplishments
- `_assign_deterministic` (argmax over per-label mixture scores) and `_assign_probabilistic` (softmax + `torch.multinomial` categorical sample) implemented per D-12.
- `generate_labels()` fully rewritten: supports `n_labels` (simple, auto-random Voronoi, drop-in for the old `n_classes` behaviour) and `label_specs` (full, arbitrary shapes/mixtures/label IDs) paths, validates mutual exclusivity and the `n_labels`+`mode="probabilistic"` combination, and resolves all region/component centers exactly once before the per-frame loop — the D-07 bug (centers redrawn every frame) is fixed.
- `assign_cap_labels`, `assign_gaussian_labels`, and their 13 tests (`TestSphericalLabelGenerators`) deleted entirely from `labels.py` and `tests/test_generators.py` — no back-compat shims, per D-06. `_angular_distance_deg` retained and confirmed still used by `_component_score`'s cone branch.
- `__all__` in both `labels.py` and `data_generation/__init__.py` updated to `LabelComponentSpec`, `LabelSpec`, `generate_labels`, `remove_labels`.
- Full test suite recovered from 35 failed/104 errors (caused by the `n_classes`→`n_labels` rename breaking every direct caller) to 1350 passed/22 skipped/1 xpassed via a Rule-1 auto-fix sweep of every `generate_labels(..., n_classes=...)` call site outside this plan's declared file scope.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement assignment modes and rewrite generate_labels() orchestrator** - `149d58b` (feat)
2. **Task 2: Delete assign_cap_labels/assign_gaussian_labels and their tests (D-06)** - `b43da79` (feat)
3. **Deviation fix: update generate_labels() call sites for n_classes->n_labels rename** - `e8bdc24` (fix)

## Files Created/Modified
- `src/zreg/data_generation/labels.py` - added `_assign_deterministic`/`_assign_probabilistic`; rewrote `generate_labels()`; deleted `assign_cap_labels`/`assign_gaussian_labels`; updated module `__all__` and docstring
- `src/zreg/data_generation/__init__.py` - updated exports (`LabelComponentSpec`, `LabelSpec`, `generate_labels`, `remove_labels`); updated package docstring's `labels` bullet
- `tests/test_generators.py` - renamed `n_classes=`→`n_labels=` in `TestLabelUtilities` and `TestGeneratorsCoverageGaps`; renamed `test_generate_labels_invalid_n_classes`→`test_generate_labels_invalid_n_labels`; deleted `TestSphericalLabelGenerators` (13 tests) and its now-unused import
- `eval/data_factory.py` - internal `generate_labels(base, n_labels=n_classes, seed=seed)` call fixed inside `generate_training_triple` (public `n_classes` param name unchanged, deferred to Plan 56-03)
- `eval/runners/optimizer.py` - sanity-tier `generate_labels(traj, n_labels=4, seed=42)` call fixed
- `tests/test_benchmark_runner.py`, `tests/test_optimizer.py`, `tests/test_trajectory_export.py`, `tests/test_viz.py`, `tests/test_label_transfer_stage.py`, `tests/test_eval_runner.py` - renamed `n_classes=`→`n_labels=` at every direct `generate_labels()` call site and matching docstring/comment references

## Decisions Made
- The `n_labels` path draws `torch.randn(n_labels, 3)` without pinning `dtype`/`device` — the resulting centers are converted to python floats via `.tolist()` before being stored on `LabelComponentSpec.center`, and `_component_score` re-casts to `pos.dtype`/`pos.device` per frame, so no precision/device mismatch is introduced.
- `n_labels` combined with `mode="probabilistic"` raises `ValueError` naming both parameters rather than silently picking a default temperature — matches D-10's framing that voronoi has no natural probability scale without an explicit temperature, and the simple path intentionally exposes no way to set one.
- Only `generate_labels()`'s own call sites were fixed in this plan's deviation sweep; `DataFactory.generate_training_triple`'s and `DataFactory.generate_training_set`'s own `n_classes` parameter names were left untouched, since renaming those public-facing params is explicitly Plan 56-03's scope (D-04/D-13, config wiring).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `n_classes`→`n_labels` rename broke every direct `generate_labels()` caller outside this plan's declared file scope**
- **Found during:** Post-Task-2 full-suite verification (`python -m pytest -q --no-cov`)
- **Issue:** Task 1's rename of `generate_labels()`'s `n_classes` parameter to `n_labels` is a breaking API change. `56-CONTEXT.md`'s canonical_refs only documented two production call sites needing rewiring (`eval/data_factory.py`, `eval/runners/optimizer.py`), both explicitly assigned to Plan 56-03. However, running the full suite revealed 8 additional files with **direct** `generate_labels(..., n_classes=...)` calls that `56-CONTEXT.md` had not enumerated: `eval/data_factory.py` (internal call inside `generate_training_triple`), `eval/runners/optimizer.py`, and six test files (`test_benchmark_runner.py`, `test_optimizer.py`, `test_trajectory_export.py`, `test_viz.py`, `test_label_transfer_stage.py`, `test_eval_runner.py`). This produced 35 failed + 104 errors in the full suite immediately after Task 1's commit.
- **Fix:** Renamed every direct `generate_labels(..., n_classes=...)` call to `generate_labels(..., n_labels=...)` across all 8 files (kwarg only, matching docstring/comment references updated too). Deliberately left `DataFactory.generate_training_triple`'s/`generate_training_set`'s own `n_classes` parameter names untouched (`eval/data_factory.py`'s internal call became `generate_labels(base, n_labels=n_classes, seed=seed)` — the public param name is unchanged), since that public-facing rename is Plan 56-03's declared scope (D-04/D-13). Also left every unrelated `n_classes` usage alone (PointNet++/eGNN model hyperparameters in `test_label_transfer_stage.py`, `test_zreg_models_*.py`, etc.) — confirmed via `grep -n "generate_labels("` that only true call sites were touched.
- **Files modified:** `eval/data_factory.py`, `eval/runners/optimizer.py`, `tests/test_benchmark_runner.py`, `tests/test_optimizer.py`, `tests/test_trajectory_export.py`, `tests/test_viz.py`, `tests/test_label_transfer_stage.py`, `tests/test_eval_runner.py`
- **Verification:** Full suite went from 35 failed/104 errors to 1350 passed/22 skipped/1 xpassed. Remaining 2 failures (`test_icp_registration.py`) confirmed unrelated (see Issues Encountered).
- **Committed in:** `e8bdc24`
- **Overlap note for Plan 56-05:** After making this fix, discovered `ROADMAP.md` already lists a dedicated `56-05-PLAN.md` ("Fix rename-cascade breakage across the remaining existing test suite (9 files)") whose Task 1 targets exactly the same 5 test files (`test_optimizer.py`, `test_viz.py`, `test_trajectory_export.py`, `test_label_transfer_stage.py`, `test_eval_runner.py`) for the identical `generate_labels(..., n_classes=...)`→`n_labels=` rename. This plan's fix makes Plan 56-05's Task 1 a no-op (its verify grep will simply find nothing left to change — non-conflicting). Plan 56-05's Task 2 remains fully applicable and untouched: `test_benchmark_runner.py`'s `generate_training_set(seeds, n_classes=N_CLASSES)` call (line ~307, a different function whose signature this plan did not touch) plus `test_data_factory_training_triples.py`, `test_zreg_models_pointnet2.py`, and `test_train_label_transfer.py`'s `generate_training_triple`/`generate_training_set` call sites are all still pending and unaffected by this plan.

---

**Total deviations:** 1 auto-fixed (1 bug, cross-file blast radius from Task 1's intentional rename)
**Impact on plan:** No scope creep in intent — the fix is a mechanical kwarg rename directly necessitated by Task 1's own change, matching D-04's stated intent ("rename n_classes → n_labels everywhere in the label-generation surface"). File-scope footprint is larger than the plan's declared `files_modified` (which only listed 3 files), but every touched file is a pure rename with no behavioural change beyond restoring passing tests. Plan 56-03's own scope (renaming `DataFactory.generate_training_triple`'s public `n_classes` param, and any remaining config-wiring work) is untouched and unaffected.

## Issues Encountered
- **`tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery` and `test_icp_rotation_recovery` fail only in full-suite runs, not in isolation.** Confirmed unrelated to this plan (ICP registration code was never touched): `pytest tests/test_icp_registration.py` alone passes 13/13, and `pytest tests/test_icp_registration.py::...test_icp_rotation_recovery tests/test_generators.py` together passes 53/53. This is pre-existing test-order-dependent flakiness (likely RNG/Open3D global state leakage from an unrelated test file earlier in full-suite order). Logged to `deferred-items.md` in the phase directory; not fixed (out of scope).
- Local environment requires `KMP_DUPLICATE_LIB_OK=TRUE` for `python -c` verification snippets (same pre-existing environment quirk noted in Plan 56-01's summary), and the ad-hoc smoke-check script needed `sys.path.insert(0, 'src')` since the installed editable `zreg` package points at a different worktree — `pytest`'s `conftest.py` already handles this via `sys.path.insert(0, str(Path(__file__).parent / "src"))`, so `pytest`-based verification was unaffected.
- Discovered mid-execution that an earlier `cd /Users/valeriekieslinger/Documents/Hiwi/BA/zReg && ...` command had drifted the Bash tool's cwd out of the assigned worktree into the main repo (the #3097 cwd-drift hazard) — caught before any commit was made in the wrong location; all subsequent commands were re-run from the correct worktree path (`.../.claude/worktrees/generate-labels-vision-969cb0`) and verified via `git rev-parse --show-toplevel`/`--git-dir` before proceeding.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `generate_labels(trajectory, n_labels=..., label_specs=..., mode=..., seed=...)` is the single, config-driven entry point ready for Plan 56-03's `EvalConfig.label_generation` wiring.
- `assign_cap_labels`/`assign_gaussian_labels` are fully gone from the codebase (function, export, tests) — confirmed via `grep -rn "assign_cap_labels\|assign_gaussian_labels" src/ tests/ eval/` returning no matches.
- D-07 is fixed and spot-checked with a 2-frame identical-`pos` smoke test; the comprehensive regression test proving temporal correlation across a full trajectory is explicitly deferred to Plan 56-04 per this plan's own scope.
- No blockers for Plan 56-03 (config wiring) or Plan 56-04 (comprehensive tests).

---
*Phase: 56-configurable-multi-label-region-based-labeling-rework-genera*
*Completed: 2026-07-31*

## Self-Check: PASSED

- FOUND: src/zreg/data_generation/labels.py
- FOUND: src/zreg/data_generation/__init__.py
- FOUND: tests/test_generators.py
- FOUND: eval/data_factory.py
- FOUND: eval/runners/optimizer.py
- FOUND: tests/test_benchmark_runner.py
- FOUND: tests/test_optimizer.py
- FOUND: tests/test_trajectory_export.py
- FOUND: tests/test_viz.py
- FOUND: tests/test_label_transfer_stage.py
- FOUND: tests/test_eval_runner.py
- FOUND: .planning/phases/56-configurable-multi-label-region-based-labeling-rework-genera/deferred-items.md
- FOUND: 149d58b (Task 1 commit)
- FOUND: b43da79 (Task 2 commit)
- FOUND: e8bdc24 (deviation-fix commit)
