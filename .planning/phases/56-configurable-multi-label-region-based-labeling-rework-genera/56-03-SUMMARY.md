---
phase: 56-configurable-multi-label-region-based-labeling-rework-genera
plan: 03
subsystem: eval-config-wiring
tags: [pydantic, config, torch, labels, n_classes-rename, synthetic-data]

# Dependency graph
requires:
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 01)
    provides: "LabelComponentSpec/LabelSpec pydantic models"
  - phase: 56-configurable-multi-label-region-based-labeling-rework-genera (Plan 02)
    provides: "generate_labels() as the config-driven orchestrator (n_labels/label_specs paths)"
provides:
  - "EvalConfig.label_generation field (LabelGenerationConfig model wrapping n_labels/label_specs/mode/seed)"
  - "DataFactory.generate_training_triple/generate_training_set fully renamed to n_labels with three-way precedence"
  - "optimizer.py sanity tier config-aware label generation"
  - "configs/label_generation_example.yaml demonstrating the new config surface"
affects: [56-04-comprehensive-tests, 56-05-remaining-test-suite-fixes]

# Tech tracking
tech-stack:
  added: []
  patterns: ["sentinel int | None = None precedence chain (explicit arg > config sub-model > hardcoded fallback)"]

key-files:
  created: [configs/label_generation_example.yaml]
  modified: [eval/config.py, eval/data_factory.py, eval/runners/optimizer.py, eval/runners/benchmark_runner.py, train_label_transfer.py]

key-decisions:
  - "eval/config.py now imports zreg.core.dataset (as a noqa import-order guard) before zreg.data_generation, preserving the scipy-before-torch macOS-ARM libomp SIGABRT convention even though eval/config.py itself does not directly use zRegPointCloud"
  - "generate_training_triple/generate_training_set use n_labels: int | None = None as a sentinel (not a concrete int=6 default) because a concrete default is indistinguishable from an explicit caller override of the same value and cannot support the three-way precedence rule"
  - "benchmark_runner.py's run_leakage_guard and train_label_transfer.py's --n-classes CLI flag/hyperparams keep their own n_classes-named public surface unchanged (out of D-04 scope per 56-CONTEXT.md); only their internal forwarding kwarg into the renamed DataFactory methods was updated"

requirements-completed: [D-04, D-13]

# Metrics
duration: ~20min
completed: 2026-07-31
---

# Phase 56 Plan 03: Config Wiring and n_classes -> n_labels Rename Cascade Summary

**Added `EvalConfig.label_generation` (a `LabelGenerationConfig` pydantic model) so scenario YAML can declare label specs declaratively, and completed the `n_classes` -> `n_labels` rename cascade through `DataFactory.generate_training_triple`/`generate_training_set`, `optimizer.py`'s sanity tier, and every internal forwarding call site.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-31 (immediately after Plan 56-02 completed)
- **Completed:** 2026-07-31
- **Tasks:** 3
- **Files modified:** 5 (4 modified + 1 new YAML), exactly matching the plan's declared `files_modified`

## Accomplishments
- `LabelGenerationConfig(BaseModel)` added to `eval/config.py`: `n_labels: int | None = None`, `label_specs: list[LabelSpec] | None = None`, `mode: Literal["deterministic", "probabilistic"] = "deterministic"`, `seed: int | None = 42`, with a `model_validator(mode="after")` enforcing mutual exclusivity between `n_labels`/`label_specs` — mirrors `generate_labels()`'s own validation so a bad config fails fast at `EvalConfig` construction time.
- `EvalConfig.label_generation: LabelGenerationConfig | None = None` added after `label_names`, defaulting to `None` — fully backward compatible with every existing YAML config. Added to `__all__` and given a matching docstring `Attributes` entry.
- `eval/config.py` now imports `zreg.core.dataset` (import-order guard, `# noqa: F401`) before `zreg.data_generation` — preserves the scipy-before-torch macOS-ARM libomp SIGABRT convention now that `eval/config.py` is no longer zreg/torch-free.
- `DataFactory.generate_training_triple`'s `n_classes: int = 6` parameter renamed to `n_labels: int | None = None` (sentinel, not a concrete default) implementing the exact three-way precedence from the plan: (1) explicit caller `n_labels` wins, (2) else `self.config.label_generation` is consulted (forwarding `n_labels`/`label_specs`/`mode`, with `seed` still driven by the per-triple `seed` argument), (3) else the hardcoded `n_labels=6` fallback (today's behaviour, unchanged).
- `DataFactory.generate_training_set`'s `n_classes: int = 6` parameter renamed to `n_labels: int | None = None`, forwarding unchanged (including `None`) to `generate_training_triple` so its own three-way precedence resolves the value.
- `eval/runners/optimizer.py`'s sanity-tier `_tier_dataset` call renamed `n_classes=4` -> consults `self.config.label_generation` when set (forwarding `n_labels`/`label_specs`/`mode`/`seed`), falling back to the hardcoded `generate_labels(traj, n_labels=4, seed=42)` otherwise. Module docstring's "Sanity tier labels (Pitfall 5)" note and the `_tier_dataset` docstring both updated to describe the config-aware fallback.
- `eval/runners/benchmark_runner.py`'s `run_leakage_guard` internal call updated to `DataFactory(self.config).generate_training_set(held_out_seeds, n_labels=n_classes)` — its own public `n_classes` parameter name and docstring are unchanged (shared naming with model hyperparameters elsewhere, out of D-04 scope).
- `train_label_transfer.py`'s `main()` internal call updated to `factory.generate_training_set(seeds, n_labels=args.n_classes)` — `args.n_classes`, the `--n-classes` CLI flag, and `hyperparams["n_classes"]` are all unchanged.
- `configs/label_generation_example.yaml` created: an illustrative sanity-tier scenario config demonstrating `label_generation` with a two-label `label_specs` spec (label 0 = a single blob component, label 1 = a single cone component), `mode: deterministic`, reusing the Kobitski tracklets path from `configs/synthetic_mode.yaml`.
- `grep -n "n_classes" eval/data_factory.py` returns no matches — fully renamed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add LabelGenerationConfig and EvalConfig.label_generation field** - `f7a2fec` (feat)
2. **Task 2: Rename n_classes to n_labels through DataFactory and wire config.label_generation into optimizer.py + DataFactory** - `c4eab36` (feat)
3. **Task 3: Fix internal rename-cascade call sites and add an example scenario YAML** - `470ea72` (feat)

## Files Created/Modified
- `eval/config.py` - added `LabelGenerationConfig` model + `EvalConfig.label_generation` field + import-order-guard imports + `__all__`/docstring updates
- `eval/data_factory.py` - `generate_training_triple`/`generate_training_set` renamed `n_classes` -> `n_labels` (sentinel `int | None = None`) with three-way precedence resolving against `self.config.label_generation`
- `eval/runners/optimizer.py` - sanity-tier `_tier_dataset` call now consults `self.config.label_generation`, falling back to hardcoded `n_labels=4, seed=42`; module docstring and method docstring updated
- `eval/runners/benchmark_runner.py` - `run_leakage_guard`'s internal `generate_training_set` call forwards `n_labels=n_classes` (public param name unchanged)
- `train_label_transfer.py` - `main()`'s `generate_training_set` call forwards `n_labels=args.n_classes` (CLI flag/hyperparams unchanged)
- `configs/label_generation_example.yaml` (new) - illustrative scenario YAML demonstrating `label_generation`

## Decisions Made
- `eval/config.py` imports `zreg.core.dataset` (unused directly, `# noqa: F401`) purely as an import-order guard before `zreg.data_generation` — preserves the project-wide scipy-before-torch convention now that `eval/config.py` is no longer zreg/torch-free (T-56-05 mitigation from the plan's threat model).
- `n_labels: int | None = None` is a sentinel, not a concrete `int = 6` default, on both `generate_training_triple` and `generate_training_set` — required to distinguish "no explicit override" from "caller explicitly asked for the same value as the fallback," which the three-way precedence rule depends on.
- `optimizer.py`'s sanity tier forwards `self.config.label_generation.seed` (not the per-call hardcoded `42`) when `label_generation` is set, so a scenario config's `seed` field is fully respected end-to-end.
- `benchmark_runner.py`/`train_label_transfer.py`'s own public `n_classes`-named surfaces (parameter name, docstring, CLI flag, model hyperparameter dict key) were deliberately left unchanged — renaming those is explicitly out of this phase's D-04 scope per `56-CONTEXT.md` (shared naming with model hyperparameters elsewhere in the codebase).

## Deviations from Plan

None - plan executed exactly as written. All three tasks' automated verify commands passed with the exact expected outcome (Task 2's verify command surfaces pre-existing `n_classes=` kwarg `TypeError`s in test files, explicitly anticipated by the plan's own acceptance criteria and deferred to Plan 56-05).

## Issues Encountered
- Full-suite run (`python -m pytest -q --no-cov`) shows 17 failures: 14 are pre-existing `generate_training_triple`/`generate_training_set` test-call sites still using `n_classes=` kwargs (`tests/test_data_factory_training_triples.py` x2, `tests/test_benchmark_runner.py` x4, `tests/test_train_label_transfer.py` x6, `tests/test_zreg_models_pointnet2.py` x1) — every one of these is a `TypeError: ... got an unexpected keyword argument 'n_classes'`, explicitly anticipated by this plan's Task 2 acceptance criteria and already scoped to Plan 56-05's Task 2 (per Plan 56-02's SUMMARY overlap note). The remaining 2 failures (`tests/test_icp_registration.py::test_icp_translation_recovery`/`test_icp_rotation_recovery`) are the pre-existing full-suite-order flake already logged in `deferred-items.md` from Plan 56-02, unrelated to this plan.
- Ad-hoc `python -c` verification commands required `sys.path.insert(0, 'src')` (or `PYTHONPATH=src`) since the installed editable `zreg` package resolves to a stale worktree (`agent-ae89a5d4b06d40fca`) rather than this one — same pre-existing environment quirk documented in Plans 56-01/56-02's summaries. `pytest`'s `conftest.py` already handles this via its own `sys.path.insert`, so pytest-based verification was unaffected.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `EvalConfig.label_generation` is ready for Plan 56-04's comprehensive `LabelGenerationConfig` test coverage.
- `DataFactory.generate_training_triple`/`generate_training_set` and `optimizer.py`'s sanity tier fully consult `config.label_generation` with correct three-way/config-or-fallback precedence — ready for Plan 56-04's config-wiring regression tests.
- Plan 56-05's Task 2 (fixing `n_classes=` kwargs in `tests/test_data_factory_training_triples.py`, `tests/test_benchmark_runner.py`, `tests/test_train_label_transfer.py`, `tests/test_zreg_models_pointnet2.py`) remains fully applicable and unaffected by this plan — those test files still call the OLD parameter name and must be updated to `n_labels=`.
- No blockers for Plan 56-04 (comprehensive tests) or Plan 56-05 (remaining test-suite fixes).

---
*Phase: 56-configurable-multi-label-region-based-labeling-rework-genera*
*Completed: 2026-07-31*

## Self-Check: PASSED

- FOUND: eval/config.py
- FOUND: eval/data_factory.py
- FOUND: eval/runners/optimizer.py
- FOUND: eval/runners/benchmark_runner.py
- FOUND: train_label_transfer.py
- FOUND: configs/label_generation_example.yaml
- FOUND: f7a2fec (Task 1 commit)
- FOUND: c4eab36 (Task 2 commit)
- FOUND: 470ea72 (Task 3 commit)
