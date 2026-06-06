---
phase: 26-propulate-optimizer
verified: 2026-06-06T15:00:00Z
status: human_needed
score: 5/6 must-haves verified
overrides_applied: 0
gaps: []
human_verification:
  - test: "Run mpirun -n 2 integration test in an environment where propulate is fully installable (GPy dependency resolved)"
    expected: "TestPropulateMPIIntegration.test_mpirun_n2_returns_results passes and returns a non-empty list of [params, score] pairs from rank 0"
    why_human: "propulate fails to import in this environment due to missing GPy transitive dependency (ModuleNotFoundError: No module named 'GPy'), so pytest.importorskip('propulate') skips the test. Cannot verify SC 4 programmatically without a working propulate install."
---

# Phase 26: Propulate Optimizer Verification Report

**Phase Goal:** Add PropulateSearch as a selectable optimizer backend alongside Optuna. HyperparamOptimizer auto-selects Propulate when SLURM_JOB_ID is set or MPI world size > 1; defaults to Optuna otherwise. Users override via `optimiser: auto|optuna|propulate` in config.
**Verified:** 2026-06-06T15:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `search_strategy: propulate` selects PropulateSearch unconditionally | VERIFIED | `eval/runners/optimizer.py:229` — `elif strategy_name == "propulate": results = PropulateSearch().search(...)` |
| 2 | `search_strategy: auto` selects Propulate when SLURM_JOB_ID set or MPI world size > 1 | VERIFIED | `_detect_backend()` at lines 480-497; per-tier auto resolution at lines 195-196; confirmed by TestDetectBackend (4 passing tests) |
| 3 | Missing propulate install raises ImportError with install instructions | VERIFIED | Lazy import block at `search_strategies.py:315-323`; guard fires with exact "pip install zreg[propulate]" message confirmed by bash probe |
| 4 | PropulateSearch.search produces valid Trial objects (integration test via mpirun -n 2) | UNCERTAIN | TestRunDispatchPropulate (mocked) passes; TestPropulateMPIIntegration SKIPPED because propulate fails to import (missing GPy transitive dep). End-to-end mpirun path not exercised in this environment. |
| 5 | Existing Optuna path unaffected — all Phase 22 tests still pass | VERIFIED | `pytest tests/test_optimizer.py` → 5 passed |
| 6 | Docstrings explain when to prefer each backend | VERIFIED (marginal) | `EvalConfig.search_strategy` docstring describes propulate as "MPI-parallel evolutionary search… requires zreg[propulate]"; `_detect_backend` docstring describes exact auto-selection order; `PropulateSearch` class docstring explains single-island MPI semantics. No explicit "prefer X when Y" sentence but behavior is unambiguous from the docstrings. |

**Score:** 5/6 truths verified (SC 4 is UNCERTAIN — needs human verification)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_propulate.py` | 5 test classes, 9 active test methods | VERIFIED | 5 classes, 8 passing + 1 skipped; all 5 class names exactly match plan |
| `tests/_propulate_mwe.py` | MPI worker script with `main()` and `__main__` guard | VERIFIED | `main(out_path: str)` defined; `if __name__ == "__main__": main(sys.argv[1])`; not collected by pytest |
| `eval/search_strategies.py` | PropulateSearch class with lazy import guard | VERIFIED | `class PropulateSearch:` present (line 259); `PropulateSearch` in `__all__`; `import math` added; lazy import at line 315 |
| `eval/runners/optimizer.py` | `_detect_backend()` method + auto/propulate dispatch | VERIFIED | `def _detect_backend(self)` at line 453; auto dispatch at line 195-196; propulate branch at lines 229-254 |
| `eval/config.py` | `search_strategy` docstring lists all 5 values including propulate and auto | VERIFIED | Lines 67-74 describe all 5 values including EXT-03 reference |
| `setup.cfg` | `[options.extras_require]` propulate block with `propulate>=1.0,<2` and `mpi4py>=3.1` | VERIFIED | Lines 83-85; `all` extra extended at line 103 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `eval/runners/optimizer.py:run()` | `eval.search_strategies.PropulateSearch` | `elif strategy_name == 'propulate'` dispatch branch | WIRED | Line 229-230: `elif strategy_name == "propulate": results = PropulateSearch().search(...)` |
| `eval/runners/optimizer.py:run()` | `self._detect_backend()` | auto strategy resolution before dispatch | WIRED | Lines 195-196: `if strategy_name == "auto": strategy_name = self._detect_backend()` — inside per-tier loop |
| `eval/search_strategies.py:PropulateSearch.search()` | propulate library (lazy) | `try/except ImportError` with pip install message | WIRED | Lines 315-323: lazy import block confirmed functional |
| `tests/test_propulate.py::TestPropulateMPIIntegration` | `tests/_propulate_mwe.py` | `subprocess.run([mpirun, '-n', '2', sys.executable, str(helper), str(out_file)])` | WIRED | Line 185-188: wiring exists in code; skipped at runtime due to propulate import failure |
| `tests/test_propulate.py::TestDetectBackend` | `HyperparamOptimizer._detect_backend` | `patch("mpi4py.MPI.COMM_WORLD")` + `monkeypatch` env | WIRED | Confirmed working: mpi4py IS installed (v4.1.1), patching `mpi4py.MPI.COMM_WORLD` correctly intercepts the local `from mpi4py import MPI` inside `_detect_backend` |
| `tests/test_propulate.py::TestRunDispatchPropulate` | `HyperparamOptimizer.run()` | `patch("eval.runners.optimizer.PropulateSearch")` mock | WIRED | Confirmed: mock intercepts module-level import, asserts best_params.json written |

---

### Data-Flow Trace (Level 4)

Level 4 not applicable — this phase adds an optimizer backend and test coverage, not UI components rendering dynamic data.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| EvalConfig accepts propulate | `python -c "from eval.config import EvalConfig; EvalConfig(data_path='x', search_strategy='propulate'); print('OK')"` | OK | PASS |
| EvalConfig accepts auto | `python -c "from eval.config import EvalConfig; EvalConfig(data_path='x', search_strategy='auto'); print('OK')"` | OK | PASS |
| PropulateSearch importable | `python -c "from eval.search_strategies import PropulateSearch; print('importable')"` | importable | PASS |
| Lazy import guard fires | `python -c "import sys; sys.modules['propulate']=None; sys.modules['propulate.utils']=None; from eval.search_strategies import PropulateSearch; PropulateSearch().search({'x':[1,2]}, lambda p:0.5, n_trials=2, output_dir='/tmp')"` | ImportError with "pip install zreg[propulate]" | PASS |
| _detect_backend returns bayesian (world_size=1, no SLURM) | `patch('mpi4py.MPI.COMM_WORLD'); Get_size=1` | "bayesian" | PASS |
| _detect_backend returns propulate (world_size=2) | `patch('mpi4py.MPI.COMM_WORLD'); Get_size=2` | "propulate" | PASS |
| _detect_backend returns propulate (SLURM_JOB_ID set) | `os.environ['SLURM_JOB_ID']='12345'` | "propulate" | PASS |
| Phase 22 regression suite | `pytest tests/test_optimizer.py --no-cov` | 5 passed | PASS |
| Full test suite | `pytest --no-cov` | 800 passed, 18 skipped | PASS |
| setup.cfg propulate extra | configparser read | propulate>=1.0,<2 and mpi4py>=3.1 confirmed | PASS |
| mpirun -n 2 integration | `pytest tests/test_propulate.py::TestPropulateMPIIntegration` | SKIPPED (propulate import fails due to missing GPy dep) | SKIP |

---

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` probes declared for this phase.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EXT-03 | 26-01-PLAN.md, 26-02-PLAN.md | Add PropulateSearch as selectable optimizer backend with auto-detection | SATISFIED | PropulateSearch class in `eval/search_strategies.py`, `_detect_backend()` in `eval/runners/optimizer.py`, propulate extra in `setup.cfg`, 8 active tests covering all gate criteria |

**Note:** EXT-03 is referenced in ROADMAP.md as the phase requirement but is NOT listed in `.planning/REQUIREMENTS.md`. The REQUIREMENTS.md ends with FRAME-12. EXT-03 appears to be a "future extension" requirement that was scoped into this phase without being formally added to REQUIREMENTS.md. This is a documentation gap (WARNING level) — the requirement is satisfied in code but lacks a formal traceability entry.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `eval/runners/optimizer.py` | 239 | `# zero-filled placeholder (Open Question 2 — option a)` and `minimal_metrics = StageMetrics(all zeros)` | INFO | Intentional design decision (D-11): Propulate dispatch returns (params, score) tuples without real StageMetrics because the loss closure does not compute stage-level metrics. Zero-filled placeholder is correct for `best_params`/`best_score` selection; `search_history.json` will contain zero metrics for propulate trials. Documented. |
| `eval/runners/optimizer.py` | 208-254 | Double-counting: `obj = make_objective(tier_dataset, tier_name, all_history)` passed to `PropulateSearch.search()`, then explicit results loop ALSO appends to `all_history` | WARNING | Production correctness bug: every successful propulate trial is appended twice — once by `_objective` closure side-effect inside `PropulateSearch._loss`, and once by the explicit loop at lines 248-254. `best_params`/`best_score` are unaffected (max of duplicates = same max), but `search_history.json` will be 2x the expected length. Identified as CR-01 in code review. Unit test uses mocked PropulateSearch so `_objective` is never called during tests. |
| `tests/test_propulate.py` | 90-104 | `patch("mpi4py.MPI.COMM_WORLD")` patch target for `_detect_backend` MPI tests | INFO | Code review WR-01 flagged this as potentially fragile (local import `from mpi4py import MPI` in `_detect_backend` could bypass the patch if mpi4py is absent). In this environment, mpi4py 4.1.1 IS installed, so the patch correctly intercepts the attribute on the cached module object. All 4 TestDetectBackend tests pass. This becomes a genuine bug only in environments without mpi4py. |
| `setup.cfg` | 80 | Comment says "EXT-03 / D-13" but plan acceptance criteria specified "EXT-03, Phase 26" | INFO | Minor deviation from plan acceptance criteria — `grep -q "EXT-03, Phase 26" setup.cfg` returns non-zero. The EXT-03 reference is present. No functional impact. |

---

### Human Verification Required

#### 1. MPI Integration Test (SC 4)

**Test:** In an environment where propulate is fully installable (Python <3.13 or with GPy available):
1. `pip install propulate mpi4py`
2. `python -c "import propulate; print(propulate.__version__)"` — confirm import succeeds
3. `pytest tests/test_propulate.py::TestPropulateMPIIntegration -v --no-cov`

**Expected:**
- `propulate.Propulator` imports without error
- `test_mpirun_n2_returns_results` runs (not skipped)
- `mpirun -n 2 python tests/_propulate_mwe.py <out_file>` returns exit code 0
- `mpi_result.json` contains a non-empty list of `[[params_dict, score], ...]` pairs

**Why human:** propulate fails to import in this environment due to `ModuleNotFoundError: No module named 'GPy'` — a transitive dependency conflict with Python 3.13 / existing matplotlib version. `pytest.importorskip("propulate")` therefore skips the entire integration test. This is the only unverified success criterion (SC 4).

---

### Gaps Summary

No hard blockers found. One WARNING-level production correctness issue exists (double-counting of propulate trials in `all_history`) that does not block the phase goal but corrupts `search_history.json` in real MPI runs. This was identified in the code review (CR-01) and should be fixed in a follow-up.

The MPI integration test (SC 4) cannot be verified in this environment due to a propulate transitive dependency conflict. All other success criteria are met.

**Known production bug (WARNING, not BLOCKER):** In `HyperparamOptimizer.run()`, when `search_strategy="propulate"`, the `_objective` closure is called by `PropulateSearch._loss` for every trial (appending to `all_history`), and then the explicit `for params, score in results:` loop appends the same trials again. Fix: pass a throwaway sink list to `make_objective` for the propulate path instead of `all_history`, and rely solely on the explicit results loop for history population.

---

_Verified: 2026-06-06T15:00:00Z_
_Verifier: Claude (gsd-verifier)_
