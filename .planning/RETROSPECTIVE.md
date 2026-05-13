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

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 | 5 | 13 | First milestone — established TDD cycle and foundation-first ordering |
| v1.1 | 7 | 14 | Restructuring milestone — package splits, Protocol APIs, inserted phase for audit gap |

### Cumulative Quality

| Milestone | Tests Added | Requirements | Tech Debt |
|-----------|-------------|--------------|-----------|
| v1.0 | 275 | 23/23 | 7 items (all doc/observability, non-blocking) |
| v1.1 | +116 (391 total) | 29/29 | 5 items (typing inconsistency, Protocol doc-only, absolute import, config not top-level, no VALIDATION.md) |
