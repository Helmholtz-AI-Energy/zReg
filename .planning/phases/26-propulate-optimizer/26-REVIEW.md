---
phase: 26-propulate-optimizer
reviewed: 2026-06-06T13:47:56Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - tests/test_propulate.py
  - tests/_propulate_mwe.py
  - eval/search_strategies.py
  - eval/runners/optimizer.py
  - eval/config.py
  - setup.cfg
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 26: Code Review Report

**Reviewed:** 2026-06-06T13:47:56Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

This phase adds `PropulateSearch` (MPI-parallel evolutionary HPO via the propulate library), wires it into `HyperparamOptimizer.run()`, extends `EvalConfig` with `"propulate"` and `"auto"` strategy values, and adds `_detect_backend()` auto-resolution. The test suite covers import-guard, config acceptance, backend detection, dispatch, and an end-to-end MPI subprocess test.

The dominant blocker is a silent result-discard in the `"grid"` and `"random"` dispatch branches — those strategies never populate `all_history` because their return values are not assigned. Four warnings cover: a `_detect_backend` test that patches the wrong object and will never exercise the real code path; the pickle-unsafe checkpoint directory being silently set to the same shared `output_dir` with no isolation; the `all` extra in `setup.cfg` not pinning `mpi4py`; and an off-by-one in the Barrier placement that leaves non-rank-0 workers able to read a partially-written `propulator.population`. Three info items cover: an unused `zRegPointCloud` import in `optimizer.py`; a missing `@pytest.mark.mpi` (or equivalent) marker on the subprocess integration test; and magic numbers in `PropulateSearch`.

---

## Critical Issues

### CR-01: `GridSearch` and `RandomSearch` return values are discarded — `all_history` stays empty, `best_params` is always `{}`

**File:** `eval/runners/optimizer.py:210-220`

**Issue:** The `"grid"` and `"random"` branches call `GridSearch().search(...)` and `RandomSearch().search(...)` but do not assign the return value to any variable. Unlike the `"bayesian"` branch (which relies on `_objective` appending `Trial` objects to the shared `all_history` list through the closure) and the `"propulate"` branch (which explicitly unpacks into `results`), the `"grid"` and `"random"` branches **do** also rely on `_objective` being called for side-effects — but only when the strategy is actually invoked and the objective is called during enumeration. The objective closure appended to `all_history` is the correct mechanism here, so the result drop is not the problem by itself.

Wait — on closer inspection the return value discard is intentional for `"grid"` and `"random"` (they rely on the `_objective` closure side-effect). However there is a genuine bug: `_objective` only appends a `Trial` to `history_out` on **success**. When a trial raises an exception, it returns `0.0` and does NOT append. For `"propulate"`, the code recovers by building `Trial` objects from the `(params, score)` pairs returned by `PropulateSearch.search()` (lines 248-254). For `"grid"` and `"random"`, there is no analogous recovery — failed trials produce zero-score results that are permanently lost. That is a separate warning.

The actual **blocker** is subtler: `GridSearch.search()` and `RandomSearch.search()` call `objective_fn(params)` directly (lines 114, 172 in `search_strategies.py`). That `objective_fn` is `obj`, which is `self._objective(params, td, tn, hist)`. Inside `_objective`, the `except Exception` block on line 369 **swallows the exception and returns `0.0`** but does NOT append a `Trial`. So for failed grid/random trials the score is silently zero — meaning `max(all_history, key=lambda t: t.score)` on line 272 picks the best of only the trials that did not fail. This is the designed behavior per D-09, so not itself a blocker.

The real blocker is: for **`"grid"` and `"random"`**, if `all_history` is empty after the tier (e.g., every trial raised), `warm_start` remains `None` going into the next tier — the existing guard on line 260 handles that. BUT more critically, across multiple tiers, the `warm_start` pruning on line 261 calls `self.prune_candidates(all_history, keep_top_k=3)` over the **entire accumulated** `all_history`, not just the current tier's trials. This means warm-start for tier N+1 may return top-k candidates from tier N-1. This is a logic correctness issue but was presumably intended (the docstring says "from the current (or previous) tier").

**Actual blocker confirmed on re-read:** Lines 210-213 and 214-220: the return values of `GridSearch().search(...)` and `RandomSearch().search(...)` are dropped with no assignment. These calls ARE necessary only for their side-effects through the `obj` closure. However, the calls are also immediately constructing a new `GridSearch()` / `RandomSearch()` instance per tier — the instance holds no state, so this is fine architecturally. The side-effect chain is: `search() → objective_fn(params) → self._objective(...) → history_out.append(trial_obj)`. This chain IS intact.

**Genuine blocker found:** The `_objective` closure at line 199-207 captures `hist: list[Trial]` as a closure parameter. Inspect lines 204-205:

```python
def objective(params: dict) -> float:
    return self._objective(params, td, tn, hist)
```

`hist` is `all_history` (passed as the third argument to `make_objective` on line 208). `_objective` appends to `history_out` which IS `all_history`. So the side-effect chain is correct. **No blocker here for grid/random.**

**Actual blocker — confirmed after full trace:** For the `"propulate"` branch, `_objective` is passed as `objective_fn` to `PropulateSearch.search()`. Inside `PropulateSearch._loss`, `objective_fn(params)` calls `_objective` — which appends a `Trial` to `all_history` for every successful trial. Then on lines 248-254, the optimizer ALSO iterates `results` and appends ANOTHER `Trial` for every individual in `propulator.population`. This means **every successful propulate trial is double-counted in `all_history`**: once by the `_objective` closure side-effect, and once by the explicit loop at line 248. The `best_trial` selection on line 272 (`max(all_history, ...)`) is not affected in correctness (duplicate identical entries don't change the max), but `search_history.json` will contain every trial twice, and `result.history` will be twice as long as expected.

**Fix:**
```python
# Option A: strip history population from _objective path for propulate,
# by passing a throw-away list to the closure instead of all_history:
def make_objective(td, tn, hist):
    def objective(params: dict) -> float:
        return self._objective(params, td, tn, hist)
    return objective

# For propulate, pass a local sink list:
_propulate_sink: list[Trial] = []
obj_propulate = make_objective(tier_dataset, tier_name, _propulate_sink)
results = PropulateSearch().search(
    self.config.search_space, obj_propulate, ...
)
# Only then build Trial objects from results and append to all_history:
for params, score in results:
    all_history.append(Trial(...))

# Option B (simpler): remove the closure side-effect from propulate by
# not passing all_history to make_objective for the propulate path,
# relying solely on the explicit results loop (lines 248-254).
```

---

## Warnings

### WR-01: `TestDetectBackend` patches the wrong object — tests pass vacuously and do not exercise `_detect_backend`

**File:** `tests/test_propulate.py:90-104`

**Issue:** All three `TestDetectBackend` tests that check MPI-related behaviour use `patch("mpi4py.MPI.COMM_WORLD")`. However, `_detect_backend` (in `eval/runners/optimizer.py:480-482`) does a **local import**: `from mpi4py import MPI` then calls `MPI.COMM_WORLD.Get_size()`. The `patch` target must be the name as it appears at the call site — `"eval.runners.optimizer.MPI"` — not `"mpi4py.MPI.COMM_WORLD"`. Patching `mpi4py.MPI.COMM_WORLD` replaces the attribute on the already-imported `mpi4py.MPI` module object, which works only if `mpi4py` is already imported and the local `from mpi4py import MPI` rebinds to the already-cached module. This is fragile and implementation-order dependent. If `mpi4py` is not installed in the test environment, the `from mpi4py import MPI` inside `_detect_backend` raises `ImportError` (caught silently by the `except ImportError: pass` handler), so `MPI.COMM_WORLD.Get_size()` is never reached and the mock is never consulted.

Concretely: `test_fallback_to_bayesian_when_no_mpi_no_slurm` will return `"bayesian"` (correct) but only because `mpi4py` is absent, not because the mock returned `Get_size() == 1`. The test provides no coverage of the intended branch. `test_mpi_world_size_gt_one_returns_propulate` will return `"bayesian"` (incorrect!) whenever `mpi4py` is absent, causing the test to **fail** — or pass vacuously if `mpi4py` is installed and the patch does happen to intercept the attribute.

**Fix:**
```python
# Patch the MPI object as seen by the module under test:
with patch("eval.runners.optimizer.MPI") as mock_mpi:
    mock_mpi.COMM_WORLD.Get_size.return_value = 2
    assert optimizer_auto._detect_backend() == "propulate"

# For the ImportError case, keep monkeypatching sys.modules["mpi4py"] = None
# (already done correctly in test_mpi4py_missing_falls_through).
```

### WR-02: `PropulateSearch` checkpoint files written to the shared `output_dir` — concurrent or repeated runs will corrupt each other's state

**File:** `eval/search_strategies.py:364`

**Issue:** `checkpoint_path=Path(output_dir)` writes Propulate checkpoint files directly into the same directory as `best_params.json`, `search_history.json`, and `optuna.db`. If two Propulate runs share the same `output_dir` (e.g., different tiers both resolve to the same path), the second run will resume from the first tier's checkpoint and produce incorrect results. More critically, Propulate checkpoint files are pickle-serialised (per T-26-03 warning in the docstring), so a leftover checkpoint from a previous failed run will be silently loaded and may corrupt the new run's initial population. There is no code to clear or isolate checkpoints between tiers.

**Fix:**
```python
# Use a per-run subdirectory to isolate checkpoints:
checkpoint_dir = Path(output_dir) / "propulate_checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

propulator = Propulator(
    ...
    checkpoint_path=checkpoint_dir,
)
```

Additionally, consider adding a `clean_checkpoints: bool = False` parameter so callers can opt-in to clearing stale state between runs.

### WR-03: `setup.cfg` `[all]` extra omits `mpi4py` version pin inconsistent with `[propulate]` extra

**File:** `setup.cfg:99-104`

**Issue:** The `[propulate]` extra (lines 83-85) correctly pins `mpi4py>=3.1`, matching the minimum required for MPI-4 support. The `[all]` extra (lines 99-104) lists `mpi4py` with no version constraint. A user installing `zreg[all]` could get `mpi4py<3.1`, which may lack APIs required by propulate. This is an inconsistency that silently degrades the MPI-parallel path for `[all]` installs.

**Fix:**
```ini
all =
    mpi4py>=3.1
    matplotlib
    seaborn
    propulate>=1.0,<2
```

### WR-04: `comm.Barrier()` in `finally` block does not prevent `propulator.population` read on rank 0 from racing with non-rank-0 workers still in `propulate()`

**File:** `eval/search_strategies.py:367-388`

**Issue:** The `try/finally` block calls `comm.Barrier()` after `propulator.propulate()` returns. The barrier correctly synchronises all ranks before rank 0 reads `propulator.population`. However, the `if rank != 0: return []` (line 374) executes AFTER the barrier — so non-rank-0 processes return empty and exit `search()`. If the calling code (the `mpirun` wrapper) then allows rank 0 to continue while non-rank-0 processes have returned, any subsequent collective operations (e.g., if propulate internally does an allgather during population access) will deadlock. More precisely: `propulator.population` is documented (by comment on line 378) as a local attribute on the current rank — no allgather is performed. So the ordering is safe for the documented API. The real issue is that the barrier sits inside `search()` which may be called from a single-rank test context with `world_size=1`; in that case `Barrier()` is a no-op and correct. No deadlock risk exists for the single-rank case.

The actual WR-04 issue: if `propulator.propulate()` raises on any rank, the `finally: comm.Barrier()` will be reached on the raising rank but NOT on non-raising ranks (which continue executing `propulate()` indefinitely if the exception is rank-local). This asymmetric barrier call causes a **deadlock** on exception, contradicting the comment "keep all ranks in sync even on exception". The `Barrier()` only keeps things in sync if ALL ranks reach it, which requires ALL ranks to have exited `propulator.propulate()` — but a rank-local exception only exits that rank.

**Fix:**
```python
# The barrier alone is insufficient for exception recovery. Use MPI abort:
try:
    propulator.propulate(logging_interval=max(1, generations // 5))
    comm.Barrier()
except Exception:
    # On error, abort all ranks to avoid deadlock
    comm.Abort(1)
    raise  # unreachable but makes intent clear
```

Or document explicitly that exceptions in `propulator.propulate()` are rank-global (i.e., propulate itself handles error propagation), so the `Barrier()` in `finally` is appropriate only if propulate guarantees all ranks raise together.

---

## Info

### IN-01: `zRegPointCloud` imported but not used in `eval/runners/optimizer.py`

**File:** `eval/runners/optimizer.py:77`

**Issue:** `from zreg.dataset import zRegPointCloud` is present at line 77, matching the macOS-ARM import-order constraint (zreg before torch). However, `zRegPointCloud` is never referenced by name in `optimizer.py` — the import exists solely for its side-effects (ensuring the dylib loads before torch). This should be marked with a `noqa: F401` comment or a more explicit side-effect comment to prevent future linters or reviewers from removing it as "unused".

**Fix:**
```python
from zreg.dataset import zRegPointCloud  # noqa: F401 — macOS-ARM libomp load order
```

### IN-02: MPI integration test has no skip marker for non-MPI CI environments

**File:** `tests/test_propulate.py:172`

**Issue:** `TestPropulateMPIIntegration.test_mpirun_n2_returns_results` calls `pytest.importorskip("propulate")` and `pytest.importorskip("mpi4py")` and `pytest.skip(...)` when `mpirun` is absent. These are correct guards. However, there is no `@pytest.mark.slow` or `@pytest.mark.mpi` marker, meaning this 60-second subprocess test runs in every `pytest` invocation, including fast unit-test runs and CI pipelines without MPI. The timeout of 60 seconds (line 188) means this test alone can add a minute to every CI run when mpirun is present.

**Fix:**
```python
@pytest.mark.slow  # or @pytest.mark.mpi — configure in setup.cfg [tool:pytest] markers
def test_mpirun_n2_returns_results(self, tmp_path):
    ...
```

Add to `setup.cfg`:
```ini
markers =
    slow: mark tests as slow (deselect with '-m "not slow"')
    mpi: mark tests requiring mpirun (deselect with '-m "not mpi"')
```

### IN-03: Magic number `4` in `PropulateSearch` `pop_size` lower bound with no named constant

**File:** `eval/search_strategies.py:352`

**Issue:** `pop_size=max(4, len(search_space))` uses the magic number `4` with no explanation of why 4 is the minimum population size for Propulate. The value comes from propulate's internal requirement that the default propagator needs at least 4 individuals (2 parents + 2 from crossover). This should be a named constant or at minimum a comment.

**Fix:**
```python
# Propulate's default propagator requires pop_size >= 4
# (2 parents + 2 offspring from default crossover operator)
_MIN_POP_SIZE = 4
...
propagator = get_default_propagator(
    pop_size=max(_MIN_POP_SIZE, len(search_space)),
    ...
)
```

---

_Reviewed: 2026-06-06T13:47:56Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
