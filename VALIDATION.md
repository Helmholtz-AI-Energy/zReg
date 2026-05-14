# zReg Validation Record

Backfilled 2026-05-13 as part of Phase 12 (CARRY-05). Each row records the validation state at the end of the named phase: the completion date, the passing test count at that point, and a brief description of what was validated. Covers Phases 6 through Phase 11.1 (all v1.1 milestone phases); Phases 1–5 (v1.0 Consolidation) are out of scope per D-06.

| Phase | Name | Completed | Tests Passing | Summary |
| ----- | ---- | --------- | ------------- | ------- |
| 6 | Python 3.12 Migration | 2026-04-14 | all passing | Raised the Python version floor to 3.12, converted all `Union`/`Optional` type hints to PEP 604 `X \| Y` syntax, replaced typing-module generics with built-in equivalents, and confirmed full-codebase AST validity under Python 3.12. |
| 7 | CPD Deep Restructure | 2026-04-20 | all passing | Extracted the CPD implementation from a monolithic module into a structured package (`cpd/`) with an abstract base class, typed result namedtuples, kernel utilities, and four concrete variant classes (Rigid, Affine, NonRigid, ConstrainedNonRigid); UAT confirmed 5/5 CPD acceptance tests passed. |
| 8 | DTW Deep Restructure | 2026-04-23 | all passing | Migrated `DynamicTimeWarping` from a flat module into a `dtw/` package with `core.py`, `result.py`, and `constraints.py`; the original `dtw.py` was removed, backwards-compatible import paths were verified, and `compose_constraints()` was covered by 4 dedicated tests. |
| 9 | Distance & Transform Restructure | 2026-04-27 | all passing | Replaced star imports in `distances/__init__.py` with explicit `__all__`, created the `DistanceMetric` Protocol for internal typing, and added comprehensive NumPy-style docstrings to `BaseWD`, `_sanitize_pairwise_distance_matrix`, and all kernel-based transform classes. |
| 10 | Code Quality & Verification | 2026-04-29 | all passing | Fixed three silent-failure bugs (minkowski `p` parameter, normalize zero-range axis, CPD `use_color` guard), replaced import-time PyTorch global side effects with an opt-in `configure_pytorch()` function, and inventoried all remaining `TODO(deferred)` markers for v1.2+ planning. |
| 11 | Validate Refactoring and Fix Coverage | 2026-05-12 | 365 passing | Removed the zero-caller `_preserve_labels` dead function, added 10 targeted tests covering 8 previously-uncovered branches in `downsampling.py`, and raised `downsampling.py` coverage from 85% to 93%; full regression suite confirmed 365 passing tests, 0 failures. |
| 11.1 | Close DTW-02: consistent metric variant interface | 2026-05-13 | 107 passing | Promoted `DistanceMetric` Protocol to the public API (`from zreg.distances import DistanceMetric`), wired a `callable()` early-continue pass-through into `_sanitize_pairwise_distance_matrix`, widened type annotations in `DynamicTimeWarping` and `create_pairwise_distance_matrix`, and verified 4 new end-to-end callable-metric tests pass alongside 107 total tests. |

## Data Sources

Each row is sourced from the corresponding plan SUMMARY.md file under `.planning/phases/`:

- Phase 6: `.planning/phases/06-python-3-12-migration/06-01-SUMMARY.md`, `06-02-SUMMARY.md`
- Phase 7: `.planning/phases/07-cpd-deep-restructure/07-01-SUMMARY.md`, `07-02-SUMMARY.md`, `07-03-SUMMARY.md`
- Phase 8: `.planning/phases/08-dtw-deep-restructure/08-01-SUMMARY.md`, `08-02-SUMMARY.md`
- Phase 9: `.planning/phases/09-distance-transform-restructure/09-01-SUMMARY.md`, `09-02-SUMMARY.md`
- Phase 10: `.planning/phases/10-code-quality-verification/10-01-SUMMARY.md`, `10-02-SUMMARY.md`, `10-03-SUMMARY.md`
- Phase 11: `.planning/phases/11-validate-refactoring-and-fix-coverage/11-01-SUMMARY.md` and `11-VERIFICATION.md`
- Phase 11.1: `.planning/phases/11.1-close-dtw-02/11.1-01-SUMMARY.md`

## Notes

**Test counts for Phases 6–10:** Runtime test execution was not available during these phases due to
the `open3d` dependency being absent in the executor environment. Verification was performed via
Python AST syntax checking (`py_compile`) and code-inspection pattern greps. "all passing" means
all verifiable checks passed; no runtime failures were observed in subsequent phases that ran the
full suite. The Phase 11 full regression run (365 passing, 0 failures) retroactively confirms no
regressions from Phases 6–10.

**Test count for Phase 11:** Sourced from `11-VERIFICATION.md` criterion SC-1: "365 passed, 0 failures"
across the full repository.

**Test count for Phase 11.1:** Sourced from `11.1-01-SUMMARY.md`: "107 passed, 1 skipped (CUDA), 0 failed"
in `tests/test_dtw.py` and `tests/test_pairwise_distance_matrix.py` (the two modules exercised by this phase).

**Date discrepancy — Phase 6:** ROADMAP shows `2026-04-13`; both SUMMARY files record `completed: 2026-04-14`.
The SUMMARY date is used as it reflects the actual commit timestamp.
