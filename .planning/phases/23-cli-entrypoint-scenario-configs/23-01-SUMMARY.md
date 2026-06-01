---
phase: 23-cli-entrypoint-scenario-configs
plan: 01
subsystem: cli
tags: [argparse, cli, pyyaml, pydantic, unittest.mock, FRAME-11]

# Dependency graph
requires:
  - phase: 22-hyperparam-optimizer-search-strategies
    provides: HyperparamOptimizer(config).run() + best_params.json output
  - phase: 21-evaluationrunner-visualisation
    provides: EvaluationRunner(config, params).run() + EvalReport
  - phase: 17-framework-config-datafactory
    provides: EvalConfig.from_yaml + EvalConfigError
provides:
  - run_eval.py CLI entrypoint at repo root (FRAME-11)
  - argparse parser with 4 flags: --config, --mode, --output-dir, --verbose
  - Mode dispatch: optimize/eval/full with correct Pitfall 5+6 ordering
  - run_config.yaml exact copy to output_dir before pipeline executes (D-10)
  - _load_best_params with INFO-level logging of param source (D-01/D-02)
  - 8 passing FRAME-11 unit tests in tests/test_cli.py
  - TestScenarioConfigs skeleton (Plan 02 populates FRAME-12)
affects:
  - 23-02 (Plan 02 — FRAME-12 scenario configs, TestScenarioConfigs population)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "argparse with main(argv=None) for direct-import testability (Pattern 1)"
    - "shutil.copy for exact YAML copy (D-10) — no yaml.dump defaults expansion"
    - "run_eval.HyperparamOptimizer / run_eval.EvaluationRunner as patch targets (not eval.runners.*)"
    - "src/ path injection in CLI script for src-layout packages without pip install"

key-files:
  created:
    - run_eval.py
    - tests/test_cli.py
  modified: []

key-decisions:
  - "Patch targets: run_eval.HyperparamOptimizer and run_eval.EvaluationRunner (not eval.runners.optimizer.* / eval.runners.eval_runner.*) — because run_eval imports with `from eval.runners import ...`, creating local bindings"
  - "src/ injected into sys.path in run_eval.py so direct script invocation works without pip install (mirrors root conftest.py pattern)"
  - "T-23-01: Path(config.output_dir).resolve() before mkdir to prevent path traversal — matches optimizer.py T-22-02 pattern"
  - "em-dash U+2014 in log message 'No best_params.json found — using config defaults' (copied exactly from RESEARCH Pattern 3)"

patterns-established:
  - "main(argv=None) -> int pattern for testable CLIs without subprocess"
  - "model_copy(update={...}) for runtime config overrides (not direct attribute assignment)"
  - "EvalConfigError as the only caught exception in CLI — all other exceptions propagate as bugs"

requirements-completed:
  - FRAME-11

# Metrics
duration: 7min
completed: 2026-06-01
---

# Phase 23 Plan 01: CLI Entrypoint & FRAME-11 Tests Summary

**argparse CLI (run_eval.py) with 4 flags and 3-mode dispatch (optimize/eval/full), backed by 8 passing FRAME-11 unit tests using direct-import mock pattern**

## Performance

- **Duration:** 7 min
- **Started:** 2026-06-01T10:45:57Z
- **Completed:** 2026-06-01T10:53:10Z
- **Tasks:** 2 (Wave 0 RED + Wave 1 GREEN)
- **Files modified:** 2 created (run_eval.py, tests/test_cli.py)

## Accomplishments

- FRAME-11 closed: `run_eval.py` at repo root implements the full FRAME-11 spec (argparse 4 flags, EvalConfigError clean message, model_copy overrides, run_config.yaml exact copy BEFORE pipeline, 3-mode dispatch with correct ordering)
- 8 FRAME-11 unit tests all pass via direct `run_eval.main(argv=[...])` calls with `unittest.mock.patch` — no subprocess
- Full suite green: 731 passed, 18 skipped (+13 vs Phase 21 baseline 718)

## Task Commits

Each task was committed atomically:

1. **Task 1: RED scaffold — tests/test_cli.py with 6 classes, 9 stubs** - `1a95eea` (test)
2. **Task 2: GREEN — run_eval.py + populate 8 FRAME-11 test bodies** - `8666ff2` (feat)

**Plan metadata:** (docs commit follows)

_Note: TDD tasks had RED (NotImplementedError stubs) and GREEN (implementation + populated tests) commits_

## Files Created/Modified

- `/run_eval.py` — CLI entrypoint: `_build_parser()`, `_write_run_config()` (shutil.copy), `_load_best_params()` (json.load + INFO logging), `main(argv=None) -> int` with EvalConfigError catch, model_copy overrides, mkdir, run_config.yaml write, 3-mode dispatch
- `/tests/test_cli.py` — 6 test classes, 9 methods: 8 FRAME-11 tests (passed) + 1 TestScenarioConfigs.test_placeholder (skip, Plan 02 owns)

## Decisions Made

- **Patch target correction (Rule 1 — Bug):** RESEARCH/PATTERNS.md specified `eval.runners.optimizer.HyperparamOptimizer` as patch target, but `run_eval.py` imports `from eval.runners import HyperparamOptimizer` creating a local binding. The correct patch target is `run_eval.HyperparamOptimizer`. Fixed in Task 2 before commit.
- **src/ path injection:** `run_eval.py` must inject `src/` into sys.path so `zreg.*` is importable when running the script directly (mirrors root `conftest.py` pattern). RESEARCH mentioned only repo root injection; src/ was added as Rule 2 (missing critical functionality for correct operation).
- **T-23-01 path traversal mitigation:** Applied `Path(config.output_dir).resolve()` before `mkdir` — matches the optimizer's T-22-02 pattern at `eval/runners/optimizer.py:168`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected mock patch target path for HyperparamOptimizer and EvaluationRunner**
- **Found during:** Task 2 (test_optimize_mode_calls_optimizer assertion failed: MockOpt called 0 times)
- **Issue:** RESEARCH.md specified `eval.runners.optimizer.HyperparamOptimizer` as patch path, but `run_eval.py` uses `from eval.runners import HyperparamOptimizer` creating `run_eval.HyperparamOptimizer`. Patching at the definition site does not intercept the already-imported name.
- **Fix:** Changed all patch decorators in test_cli.py to use `run_eval.HyperparamOptimizer` and `run_eval.EvaluationRunner`.
- **Files modified:** `tests/test_cli.py`
- **Verification:** `pytest tests/test_cli.py -x -q` — 8 passed, 1 skipped
- **Committed in:** 8666ff2 (Task 2 commit)

**2. [Rule 2 - Missing Critical] Added src/ path injection to run_eval.py**
- **Found during:** Task 2 (direct `python run_eval.py --help` raised ModuleNotFoundError: No module named 'zreg')
- **Issue:** `eval.runners.eval_runner` imports `from zreg.dataset import zRegPointCloud`. When `run_eval.py` is invoked directly, `zreg` is not findable unless `src/` is in sys.path (src layout, not installed). The plan's repo root injection was insufficient.
- **Fix:** Added `_src_root = _repo_root / "src"` injection after the repo root injection, mirroring the root `conftest.py` pattern.
- **Files modified:** `run_eval.py`
- **Verification:** `python run_eval.py --help` exits 0 and prints all 4 flags.
- **Committed in:** 8666ff2 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug, 1 Rule 2 missing critical)
**Impact on plan:** Both fixes necessary for correct behavior. No scope creep.

## Known Stubs

| Stub | File | Line | Reason |
|------|------|------|--------|
| `TestScenarioConfigs.test_placeholder` | `tests/test_cli.py` | 257-258 | Intentional — Plan 02 (FRAME-12) populates this class with 5 scenario config assertions. The skip placeholder keeps collection count stable. |

This stub does NOT prevent the plan's goal (FRAME-11 closure) from being achieved.

## Issues Encountered

None beyond the two auto-fixed deviations documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Plan 02 (FRAME-12): 5 scenario YAML configs in `configs/` + TestScenarioConfigs populated. The `configs/` directory already exists (empty). `EvalConfig.from_yaml` and `EvalConfig.extra="forbid"` are the only validators needed.
- `run_eval.py` is fully functional for all 3 modes — Plan 02 just adds the YAML files.

## Self-Check

### Files exist:
- [x] `run_eval.py` — FOUND
- [x] `tests/test_cli.py` — FOUND
- [x] `.planning/phases/23-cli-entrypoint-scenario-configs/23-01-SUMMARY.md` — this file

### Commits exist:
- [x] `1a95eea` — test(23-01): add failing test scaffold for FRAME-11 CLI tests
- [x] `8666ff2` — feat(23-01): implement FRAME-11 CLI entrypoint and populate 8 FRAME-11 test bodies

---
*Phase: 23-cli-entrypoint-scenario-configs*
*Completed: 2026-06-01*
