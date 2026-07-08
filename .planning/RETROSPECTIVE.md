# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Consolidation

**Shipped:** 2026-04-09
**Phases:** 5 | **Plans:** 13 | **Tasks:** 18

### What Was Built

- `_validate_tensors()` wired into all 6 public API entry points — NaN/inf and device-mismatch detection with informative errors
- Fixed 5 bugs: MaxSWD gradient flow, RigidCPD scale formula, GSWD degree default, DTW backtrace dead-ends, homogeneous coordinate NaN/inf
- CPD convergence diagnostics (`n_iters`, `sigma2_history`) and sigma2 safety clamping with warnings
- ASWD tempfile auto-cleanup, MPI row-only allgather, color transfer enum dispatch, downsampling debug logging
- 275 passing regression tests covering all fixed subsystems across 3 test plans

### What Worked

- **Foundation-first phase order**: Shipping validation utilities (Phase 1) before any fixes meant all subsequent phases could use consistent error checking. No rework needed.
- **TDD per plan**: Writing failing tests before implementation caught contract mismatches early (e.g., MaxSWD in-place vs. copy semantics, pmat transpose ambiguity).
- **Parallel phase execution**: Phases 2 and 4 ran independently after Phase 1 with no conflicts — good dependency modeling paid off.
- **Deferring tests to Phase 5**: Tests verified the actual fixed implementations, not intermediate state. Clean regression baseline.
- **Code review for ambiguous changes**: w-clamping dtype selection (`torch.finfo(dtype).eps` vs. hardcoded), pmat strict validation vs. auto-transpose — these were caught before implementation by reading paper/existing patterns.

### What Was Inefficient

- **VALIDATION.md stale for Phases 01-02**: Templates were created at plan time but never updated after execution. VERIFICATION.md independently confirmed all truths, but Nyquist docs are misleading for future readers. Should update VALIDATION.md immediately after plan execution.
- **Phases 03-05 had no VALIDATION.md**: Nyquist pre-planning was skipped under time pressure. All verification happened post-hoc in VERIFICATION.md. Functional but documentation debt accumulated.
- **Plan 05-03 delivered 5 of 8 planned test methods**: 3 coverage tests were skipped (all `must_haves.truths` satisfied). Slightly over-scoped the plan initially.

### Patterns Established

- `_validate_tensors(*tensors, names, check_finite, check_device)` — single validation call point for all public API entry points
- `torch.clamp(x, min=torch.finfo(x.dtype).eps)` — dtype-adaptive epsilon clamping (used for sigma2 and w-coordinate)
- Pre-normalize data at call site, pass normalized data to kernel (`rbf_kernel` now pure)
- `try/except ValueError` around enum normalization to accept both string and enum inputs
- `defaults=(None, None)` on namedtuple for optional diagnostic fields (backwards compat)
- TDD cycle: write failing tests → implement → verify tests pass → write SUMMARY.md

### Key Lessons

1. **Model downstream consumers before API changes**: The `rbf_kernel` purity change required updating 3 call sites. Grep for usages before changing function contracts.
2. **Strict validation > silent correction**: The pmat "looks transposed" hint is more valuable than auto-transposing — users need to know their data is wrong, not have it silently fixed.
3. **In-place updates preserve optimizer state**: MaxSWD projection update must use `.data` assignment to preserve Adam's parameter reference. `param = new_value` breaks the optimizer.
4. **namedtuple defaults= for non-breaking extension**: Adding optional diagnostic fields to MstepResult with `defaults=(None, None)` kept all 9 existing 3-arg call sites valid.
5. **Subprocess for import-time env vars**: `ZREG_LOG_LEVEL` is read at import time, so tests using `os.environ` patching in-process don't work. Subprocess isolation is required.

### Cost Observations

- Model: claude-sonnet-4-6 (quality profile)
- Sessions: 1 intensive session (2026-04-09)
- Notable: Full 23-requirement consolidation milestone completed in a single day — well-scoped planning with clear dependency ordering made execution predictable

---

## Milestone: v1.1 — Code Quality & Refactoring

**Shipped:** 2026-05-13
**Phases:** 7 (6–11.1, including 1 inserted) | **Plans:** 14 | **Timeline:** 30 days

### What Was Built

- Full Python 3.12 migration — PEP 604/673/585 applied across all modules
- CPD restructured into package with abstract base, 4 variants, 10-export API
- DTW restructured into package with `DynamicTimeWarping`, `DTWResult`, `compose_constraints`
- Distance & Transform modules got explicit `__all__`, consistent interfaces, numpy-style docstrings
- Silent failures fixed: print→log.debug, lazy import guards, swallowed exceptions surfaced
- 391 passing tests at 94% coverage — no regressions from any refactoring phase
- `DistanceMetric` Protocol promoted to public API; callable pass-through enables custom metric functions

### What Worked

- **Package-first restructuring**: Moving flat modules to packages with `__init__.py` re-exports preserved all backwards-compatible import paths. Zero consumer breakage.
- **Inserted phases for audit gaps**: When the milestone audit found DTW-02 deferred, inserting Phase 11.1 cleanly resolved it without disrupting the rest of the milestone. The gap-→insert-→close pattern works well.
- **VERIFICATION.md as audit target**: Even without VALIDATION.md, having VERIFICATION.md for each phase made the milestone audit fast and accurate. Verification docs are the load-bearing artifact.
- **Phase 11 as coverage closer**: Deferring coverage gap fixes to a dedicated phase kept earlier restructuring phases focused. Phase 11 had clear, measurable targets (85% → 93% for downsampling).
- **Callable pass-through before string dispatch**: The minimal, non-breaking pattern for DTW-02 — rather than a full Protocol enforcement at runtime.

### What Was Inefficient

- **No VALIDATION.md for any v1.1 phase**: Same pattern as v1.0 — VERIFICATION.md is used but Nyquist docs skipped. Accumulated 7 phases of this debt. Should be standardized.
- **DTW-02 deferred twice**: First deferred from Phase 8 to Phase 9, then not addressed in Phase 9, then found by audit, then inserted as Phase 11.1. Each deferral added context overhead. Better to close small gaps inline.
- **Retroactive Phase 7 VERIFICATION.md**: Had to create it during audit (2026-05-13) because it was missing from execution. Writing VERIFICATION.md immediately after plan execution prevents this.
- **STATE.md progress bar stale**: Progress bar said 93% (6/7 phases) even after Phase 11.1 was inserted and completed — visual state was off.

### Patterns Established

- Package restructuring with backwards-compatible `__init__.py` re-exports — zero-breakage migration pattern
- `TODO(deferred)` markers for preserved dead code with explicit v1.2+ planning note
- Callable pass-through: `if callable(dist): early-continue` before string dispatch in sanitize functions
- Milestone audit → insert phase → re-audit cycle for gap closure

### Key Lessons

1. **Write VERIFICATION.md immediately after plan execution**, not at audit time. Retroactive docs are a red flag in audit.
2. **DTW-02 style deferrals compound**: Each deferral required re-reading context. Small interface consistency work should be done in the originating phase, or explicitly backlogged with a ticket.
3. **Audit before milestone close catches real gaps**: The DTW-02 gap was genuinely unaddressed (not just undocumented). The audit-to-insert-to-close pattern worked cleanly.
4. **Protocol-first API design**: Exporting `DistanceMetric` Protocol before any consumer uses it is the right order — it documents the contract; consumers adopt it when they need it.

### Cost Observations

- Model: claude-sonnet-4-6 (quality profile)
- Sessions: ~8–10 sessions across 30 days
- Notable: Restructuring phases (7, 8, 9) ran faster than code quality phases (10, 11) — clear structural targets are easier to execute than diffuse quality improvements

---

## Milestone: v1.2 — Evaluation Framework & Debt Resolution

**Shipped:** 2026-06-26
**Phases:** 27 (12–38) | **Plans:** 55 | **Timeline:** 44 days

### What Was Built

- Complete config-driven evaluation framework at repo-root `eval/` — `EvalConfig` + `DataFactory`, `MetricsEngine` + 6 frozen result types, isolated `AlignmentStage`/`LabelTransferStage`, `EvaluationRunner` + `viz.py`
- 3-tier `HyperparamOptimizer` with Optuna and MPI-parallel Propulate backends; Grid/Random strategies; `run_eval.py` CLI + scenario configs
- Dual-mode (paired source↔target + synthetic transform-spec with GT-aware HPO) and heterogeneous cross-format paired evaluation
- CPD-aligned trajectory output + stored-transform reuse (fixed 8× scale convergence) + pre-transfer alignment guard
- Trajectory export for LaTeX/pgfplots + per-frame visualisation refactor + viz unification
- Carry-forward debt closure + codebase-wide `color`→`label` rename + metrics/generators/tracking foundation
- 976 passing tests (from 391 at v1.1 close) — +585 tests

### What Worked

- **Wrap, don't reimplement**: Every stage delegates to existing `zreg.*` (DTW, CPD, color_transfer, metrics). The framework added orchestration, not duplicate algorithms — kept the surface area testable and the core library authoritative.
- **Isolated, standalone-runnable stages**: `PipelineStage` ABC with `validate_params` made Alignment and LabelTransfer independently testable before wiring into `EvaluationRunner`. Each phase shipped a green test class.
- **Frozen result types as contracts**: Defining `eval/types.py` early (Phase 18) gave every downstream phase a stable schema; `model_copy(update=...)` handled the few mutation points.
- **Audit-before-close caught a live regression**: The v1.2 audit ran the suite against HEAD and found an uncommitted (then committed) D-11 synthetic-mode sanity-isolation break in `optimizer.py` — fixed at close before tagging. Running tests during audit, not just reading VERIFICATION.md, paid off.
- **Architecture pivots handled as new phases**: The self-alignment→paired→synthetic→heterogeneous progression (Phases 30–32) and the aligned-cloud rework (33→35) were sequenced as discrete phases rather than retrofitted, keeping each change verifiable.

### What Was Inefficient

- **Out-of-order phase execution**: Phases ran 30,31,32,33,34,35 then 36,37,38 — and 28 was briefly mis-marked "Not started" in the Progress table while complete on disk. Roadmap bookkeeping drifted from reality; the milestone header still said "Phases 12–34" at close (actual 12–38).
- **EXT-01/02/03 never entered REQUIREMENTS.md**: Phases 24/25/26 delivered and verified EXT requirements that lived only in ROADMAP/STATE. The traceability table was incomplete until the archive added them at close.
- **Empty `requirements-completed` frontmatter**: Most SUMMARY.md files left this blank, so the 3-source audit cross-check fell back to VERIFICATION evidence tables. Metadata hygiene lagged.
- **VALIDATION.md coverage inconsistent**: 14 of 27 phases have none — Nyquist pre-planning was applied sporadically (same pattern flagged in v1.0/v1.1).
- **A WIP edit got committed mid-close**: The synthetic-optimizer change was committed as a `feat` between audit and completion, re-introducing a test failure the audit had already isolated. Committing flagged WIP without re-running the named test cost a round-trip.

### Patterns Established

- `PipelineStage` ABC: `run(source, target, params)` + `validate_params(params)` — two-input stages, never self-align
- Wrap existing `zreg.*` in `eval/` orchestration; never reimplement algorithms
- Frozen pydantic v2 result types; mutate via `model_copy(update=...)`
- Error-at-use-time (loaders/generators raise at call time, not construction)
- Stored CPD transform reuse: normalise with pinned params → apply stored transform → denormalise
- Tier-gated side effects in HPO: sanity tier uses isolated scratch factory; dev/full use shared `_synthetic_target`
- Optional heavy deps (Propulate) as lazy-imported extras with install-instruction ImportError

### Key Lessons

1. **Run the suite during milestone audit, not just read verifications.** VERIFICATION.md said "passed"; the live suite found a real D-11 regression. Tests are the load-bearing audit signal.
2. **Don't commit flagged WIP without re-running the named failing test.** The audit pinpointed the exact test; committing the edit anyway re-broke it.
3. **Register requirements when the phase is created, not at archive.** EXT-01/02/03 being absent from the traceability table made coverage accounting ambiguous for the whole milestone.
4. **Keep the Progress table in sync as phases land out of order.** Numeric-order assumptions broke down; the table needs updating per-completion, not per-numeric-sequence.
5. **Pivots are cheaper as new phases than retrofits.** The paired/synthetic/heterogeneous and aligned-cloud reworks stayed verifiable because each was its own phase with its own gate.

### Cost Observations

- Model profile: quality (Opus/Sonnet mix), `mode: yolo`
- Sessions: many across 44 days; ~353 commits
- Notable: largest milestone to date (27 phases / 55 plans / +16.5k LOC) — modular phase boundaries and per-phase test gates kept it tractable despite out-of-order execution

---

## Milestone: v1.4 — Trajectory Alignment & Optimization Enhancements

**Shipped:** 2026-07-08
**Phases:** 5 (39–43) | **Plans:** 11 | **Timeline:** 2 days execution (2026-06-29 → 2026-06-30); gaps closed 2026-07-08

### What Was Built

- `ICPRegistration` (Open3D point-to-point) as `alignment_method: icp` in AlignmentStage dispatcher; StoredTransform pattern
- `SlicedWassersteinAligner` with 5 SWD variants via gradient descent + SO(3) SVD projection; `alignment_method: swd` + `swd_variant` YAML
- `src/zreg/preprocessing.py` — `compute_pca_rotation` (det=+1 SVD fix) + `detect_velocity_landmarks`; `AlignmentPreprocessingConfig` pydantic model
- `SobolSearch` (scipy qmc) as default HPO; `SOBOL_MIN_TRIALS=8` fallback to `RandomSearch`; `sobol_seed`/`sobol_randomize` flat config fields
- `DataPreprocessingConfig` + `DataFactory._standardize()` per-trajectory z-score; defaults to live instance in `EvalConfig`
- Gaps closed at milestone-close: IN-01 dtype fix, IN-04 stronger assertion, OSWD num_projs=3 for 3D, cross-embryo YAML target path

### What Worked

- **All 5 phases independent:** parallel execution reduced elapsed time to 2 days for 11 plans. Clear dependency modeling paid off.
- **StoredTransform reuse:** Extending the existing normalise→compute→denormalise→cache pattern to ICP and SWD kept the aligned-cloud construction path consistent across all 3 methods.
- **Sobol as opt-out default:** Making `search_strategy` default to `"sobol"` with `SOBOL_MIN_TRIALS=8` fallback was cleaner than opt-in — all existing configs benefit automatically without YAML changes.
- **`default_factory` for DataPreprocessingConfig:** Defaulting to a live instance (not `None`) activated standardization for all users without requiring migration; the WR-03 shared-mutable-default fix caught this pattern early.

### What Was Inefficient

- **Gaps found at milestone-close, not during execution:** OSWD `num_projs=50 > dim=3` failure, cross-embryo YAML copy-paste error, and IN-01 dtype inconsistency were all addressable during Phase 40/43 execution but surfaced only at close. Running `pytest` more broadly (not just per-phase) during execution would catch these earlier.
- **Phase 43 planning files only in worktrees:** The `43-per-trajectory-data-standardization/` directory with CONTEXT.md, REVIEW.md, VERIFICATION.md existed only in the worktree, not merged into the main worktree. Planning docs should be committed along with code at merge time.
- **No milestone audit before close:** Skipped `/gsd:audit-milestone` for v1.4. The gaps found manually at close were exactly what an audit would have surfaced.

### Patterns Established

- `SlicedWassersteinAligner` gradient descent + SO(3) SVD projection (`U @ Vt` with `det=-1` sign fix) — reusable for any rotation-registration over SWD loss
- OSWD requires `num_projs <= dim` at construction — tests must pass domain-appropriate `num_projs` (not the generic default of 50)
- `DataPreprocessingConfig` default via `Field(default_factory=DataPreprocessingConfig)` — avoids shared-mutable-default hazard for nested pydantic sub-models

### Key Lessons

1. **Run the full suite at phase completion, not just phase-scoped tests.** The OSWD and cross-embryo failures were in pre-existing test files that weren't in scope for any single phase — only a full run would have caught them.
2. **Commit planning docs at worktree merge time.** Phase 43 REVIEW.md and VERIFICATION.md were lost to the worktree; they should be included in the `chore: merge executor worktree` commit.
3. **Run `/gsd:audit-milestone` before `/gsd:complete-milestone`.** The manual gap-finding at close replicated what an audit would have done systematically — and the audit would have caught the YAML bug too.

### Cost Observations

- Model: claude-sonnet-4-6 (quality profile)
- Sessions: ~3 sessions across 10 days (2 days execution + 8 days gap before close)
- Notable: shortest milestone by phase count (5) and elapsed execution time (2 days) — parallel independence made planning straightforward

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 | 5 | 13 | First milestone — established TDD cycle and foundation-first ordering |
| v1.1 | 7 | 14 | Restructuring milestone — package splits, Protocol APIs, inserted phase for audit gap |
| v1.2 | 27 | 55 | Feature milestone — repo-root `eval/` framework wrapping existing `zreg.*`; dual-mode evaluation; live-suite audit caught a close-time regression |
| v1.4 | 5 | 11 | Algorithm expansion — ICP + SWD + preprocessing + Sobol + standardization; all 5 phases independent; gaps closed at close (no audit run) |

### Cumulative Quality

| Milestone | Tests Added | Requirements | Tech Debt |
|-----------|-------------|--------------|-----------|
| v1.0 | 275 | 23/23 | 7 items (all doc/observability, non-blocking) |
| v1.1 | +116 (391 total) | 29/29 | 5 items (typing inconsistency, Protocol doc-only, absolute import, config not top-level, no VALIDATION.md) |
| v1.2 | +585 (976 total) | 30/30 | 5 items (Propulate live-mpirun unverified, 14 phases no VALIDATION.md, empty SUMMARY frontmatter, EXT untracked until close, open CR/WR items) |
| v1.4 | +180 (1156 total) | 28/28 | 3 items (MaxSWD deferred v1.5, ALIGN-06-05 integration test gap, planning docs only in worktrees) |
