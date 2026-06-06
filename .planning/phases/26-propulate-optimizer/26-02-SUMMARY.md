---
phase: 26-propulate-optimizer
plan: "02"
subsystem: eval/hpo
tags: [propulate, mpi, evolutionary-search, hyperparameter-optimization, ext-03, testing]

dependency_graph:
  requires:
    - phase: 26-propulate-optimizer/plan-01
      provides: PropulateSearch class, HyperparamOptimizer._detect_backend(), propulate dispatch branch, test stub scaffold
  provides:
    - 9 active test methods across 5 test classes in tests/test_propulate.py
    - TestPropulateImportGuard: lazy ImportError guard verified via sys.modules patch
    - TestEvalConfigAcceptsPropulate: "propulate" and "auto" values accepted
    - TestDetectBackend: all 3 detection branches + mpi4py-missing fallthrough
    - TestRunDispatchPropulate: mocked PropulateSearch asserts best_params.json content
    - TestPropulateMPIIntegration: shutil.which guard + 60s timeout subprocess (skipped when propulate absent)
  affects:
    - EXT-03 gate criteria now locked in with executable evidence

tech_stack:
  added: []
  patterns:
    - monkeypatch.setitem(sys.modules, "propulate", None) for simulating missing optional dep without uninstalling
    - patch("eval.runners.optimizer.PropulateSearch") for unit-testing propulate dispatch without MPI
    - shutil.which("mpirun") guard + subprocess.run(..., timeout=60) for bounded MPI integration tests
    - importorskip placement inside test method body (not module level) to allow other 4 classes to collect

key-files:
  created: []
  modified:
    - tests/test_propulate.py (all 5 stub test classes populated with real assertions)

key-decisions:
  - "propulate.utils patched alongside propulate in sys.modules to ensure lazy import block fails cleanly"
  - "MPI integration test uses shutil.which (cross-platform) rather than hardcoded mpirun path from RESEARCH Example 3"
  - "TestRunDispatchPropulate uses run_alignment=True + run_label_transfer=True so HyperparamOptimizer.__init__ guard passes without calling _objective"
  - "fake_results = [(window_size:3, 0.7), (window_size:5, 0.5)] — best_params verified as {window_size:3} matching highest score"

requirements-completed:
  - EXT-03

duration: ~20 minutes
completed: "2026-06-06"
tasks_completed: 2
tasks_total: 2
files_modified: 1
---

# Phase 26 Plan 02: Propulate Test Population Summary

**8 active test methods across 5 classes covering all EXT-03 gate criteria: lazy ImportError guard (D-12), _detect_backend() detection order (D-04/D-05), EvalConfig acceptance (D-01), run() dispatch with mocked PropulateSearch asserting best_params.json content (D-11), and MPI integration test with shutil.which guard.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-06-06T00:00:00Z
- **Completed:** 2026-06-06T00:00:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Replaced all 9 stub `pass` bodies with real test assertions — 8 active tests now pass deterministically
- Verified D-12 lazy import guard fires with correct "pip install zreg[propulate]" message via `monkeypatch.setitem(sys.modules, "propulate", None)` without uninstalling propulate
- Verified all 3 `_detect_backend()` branches (MPI world_size>1, SLURM_JOB_ID, bayesian fallback) plus mpi4py-missing fallthrough using `patch("mpi4py.MPI.COMM_WORLD")` + monkeypatch env var idioms
- Verified propulate dispatch branch in `HyperparamOptimizer.run()` writes best_params.json with correct content using mocked PropulateSearch

## Task Commits

1. **Task 1: Populate TestPropulateImportGuard + TestEvalConfigAcceptsPropulate + TestDetectBackend** - `b8cdeee` (test)
2. **Task 2: Populate TestRunDispatchPropulate + TestPropulateMPIIntegration; verify regression** - `399141e` (test)

## Files Created/Modified

- `/Users/valeriekieslinger/Documents/Hiwi/BA/zReg/tests/test_propulate.py` — all 5 test classes populated; added `import json`, `import shutil`; replaced all `pass` stubs with real assertions

## Decisions Made

- `propulate.utils` also patched in `sys.modules` alongside `propulate` to ensure the lazy import block raises `ImportError` cleanly (the try block imports both `propulate.Propulator` and `propulate.utils.get_default_propagator` together)
- MPI integration test uses `shutil.which("mpirun")` (cross-platform PATH lookup) rather than the hardcoded `/opt/miniconda3/bin/mpirun` from RESEARCH Example 3 — more portable for CI environments
- `TestRunDispatchPropulate` sets `run_alignment=True, run_label_transfer=True` (matching `test_optimizer.py` pattern) so `HyperparamOptimizer.__init__` validation passes without needing a real stage to run — `PropulateSearch` is fully mocked so `_objective` is never called
- Call-arg verification for `PropulateSearch.search()` uses `call_kwargs.kwargs` membership check instead of positional index since optimizer passes `n_trials`, `output_dir`, `warm_start` as keyword args

## Deviations from Plan

None — plan executed exactly as written. All test classes populated per specification. The RESEARCH Example 3 reference pattern (hardcoded mpirun path) was adjusted to use `shutil.which("mpirun")` per the plan's Task 2 action text, which supersedes the example.

## Known Stubs

None — all 9 planned test method bodies are now populated with real assertions. The MPI test (`test_mpirun_n2_returns_results`) skips cleanly via `pytest.importorskip("propulate")` when propulate is not installed (expected in this macOS-ARM Python 3.13 environment).

## Test Suite Results

| Class | Tests | Status |
|-------|-------|--------|
| TestPropulateImportGuard | 1 | PASSED |
| TestEvalConfigAcceptsPropulate | 2 | PASSED |
| TestDetectBackend | 4 | PASSED |
| TestRunDispatchPropulate | 1 | PASSED |
| TestPropulateMPIIntegration | 1 | SKIPPED (propulate not installed on macOS-ARM Python 3.13) |
| **Full suite** | **800 passed, 18 skipped** | **GREEN** |

Phase 22 regression (tests/test_optimizer.py): 5 passed — intact.

## Threat Surface Scan

No new security surface introduced — test file only. The subprocess test uses `timeout=60` (T-26-06 mitigated) and `shutil.which` for binary lookup (no path injection vector).

## Self-Check: PASSED

Files exist:
- FOUND: tests/test_propulate.py (populated)

Commits exist:
- FOUND: b8cdeee (test(26-02): populate TestPropulateImportGuard, TestEvalConfigAcceptsPropulate, TestDetectBackend)
- FOUND: 399141e (test(26-02): populate TestRunDispatchPropulate and TestPropulateMPIIntegration)

Test assertions verified:
- `pytest tests/test_propulate.py --no-cov` → 8 passed, 1 skipped (EXIT 0)
- `pytest tests/test_optimizer.py --no-cov` → 5 passed (EXIT 0)
- `pytest --no-cov` → 800 passed, 18 skipped (EXIT 0)
