---
phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu
plan: 01
subsystem: ml-models
tags: [torch_geometric, open3d, farthest-point-sampling, ball-query, radius-graph, pointnet2, egnn]

# Dependency graph
requires:
  - phase: 45-egnn-pointnet-label-transfer-framework-selection-evaluation-
    provides: locked architecture decisions (45-DESIGN.md) — per-model library choice, target src/zreg/models/ layout, joint-cloud conditioning pattern
  - phase: 46-training-data-pipeline-for-learned-label-transfer
    provides: DataFactory.generate_training_triple/generate_training_set, TrainingTriple type (consumed by later 47 plans, not this one)
provides:
  - "torch_geometric declared in setup.cfg install_requires (no compiled PyG siblings)"
  - "src/zreg/models/_ops.py — farthest_point_sample, ball_query (isolated-point guarded), build_radius_graph, all device-agnostic"
  - "src/zreg/models/__init__.py — package marker + _ops re-exports"
  - "tests/test_zreg_models_ops.py — 11 tests: correctness, isolated-point guard, D-03 benchmark, device placement"
  - "D-03 (Open3D wrapper benchmark-first) resolved: sub-50ms confirmed at 100/200/300 points, no vectorization needed"
affects: [47-02, 47-03, 47-04, 48, 49]

# Tech tracking
tech-stack:
  added: [torch_geometric (2.8.0, declared only — already installed/audited in Phase 45)]
  patterns:
    - "Open3D-wrapped geometry ops: extract positions -> build Open3D structure -> call native op -> convert back to torch.Tensor (mirrors src/zreg/registration/icp.py)"
    - "Isolated-point self-fallback guard (idx = [i]) for ball-query and radius-graph — prevents empty-tensor max-pool crashes on sparse sample_bowl regions"
    - "Device-agnostic ops: read pos.device, place outputs via torch.as_tensor(..., device=pos.device), zero bare CUDA calls"

key-files:
  created:
    - src/zreg/models/__init__.py
    - src/zreg/models/_ops.py
    - tests/test_zreg_models_ops.py
  modified:
    - setup.cfg

key-decisions:
  - "Combined Task 2 (implementation) and Task 3 (test file) into a single RED/GREEN TDD cycle: wrote the full tests/test_zreg_models_ops.py first (RED, confirmed ModuleNotFoundError), then implemented _ops.py + __init__.py to make it pass (GREEN) — since Task 3's test file was the natural RED artifact for Task 2's implementation and the plan's own tdd=\"true\" flags on both tasks pointed at the same underlying cycle."
  - "build_radius_graph excludes self-loops by design: EGNNConv's own forward() already applies a residual node-feature update (h + node_mlp(...)), so a self-loop edge would double-count the center node's own contribution. Isolated points fall back to a self-loop to guarantee every node has >=1 incoming edge."
  - "D-03 resolved: benchmark test (test_ops_benchmark_under_50ms) confirms combined FPS+ball-query wall-clock stays under 50ms at 100/200/300 points (measured 1.9-4.7ms this session) — vectorization NOT triggered, per RESEARCH.md's live measurements."

patterns-established:
  - "Pattern: Open3D-wrapped non-differentiable geometry ops live in src/zreg/models/_ops.py, consumed by both PointNet++ and eGNN model classes (added in later plans in this phase)."

requirements-completed: [MODEL-01, MODEL-04]

# Metrics
duration: 35min
completed: 2026-07-13
---

# Phase 47 Plan 01: torch_geometric Dependency + Shared Geometry Ops Summary

**Declared torch_geometric in setup.cfg and built the Open3D-wrapped `farthest_point_sample`/`ball_query`/`build_radius_graph` ops layer (with isolated-point self-fallback guard) that both PointNet++ and eGNN will consume — D-03's benchmark-first check confirms sub-50ms performance at Phase 46's actual 100-300 point regime, so no vectorization is needed.**

## Performance

- **Duration:** 35 min
- **Started:** 2026-07-13T08:31:00Z
- **Completed:** 2026-07-13T09:06:00Z
- **Tasks:** 3 (Task 1 standalone; Tasks 2+3 executed as one RED/GREEN TDD cycle)
- **Files modified:** 4 (1 modified, 3 created)

## Accomplishments
- `torch_geometric` is now a declared `install_requires` dependency in `setup.cfg` — a fresh `pip install -e .` will pull it in; no compiled PyG siblings (`torch_cluster`/`torch_scatter`/`torch_sparse`/`pyg-lib`) added, per 45-DESIGN.md's explicit prohibition
- `src/zreg/models/_ops.py` provides all three shared geometry ops from RESEARCH.md Pattern 1, verified correct and device-agnostic
- The isolated-point self-fallback guard (`idx = [i]`) is present in both `ball_query` and the new `build_radius_graph`, preventing empty-tensor crashes on `sample_bowl`'s sparser rim regions (Pitfall 2)
- D-03 (benchmark-first Open3D wrapper performance) is resolved with an explicit, repeatable pytest assertion — not just a one-off research measurement
- 11 new tests in `tests/test_zreg_models_ops.py`; full repo suite at 1259 passed (was 1248), 18 skipped, 1 xpassed — zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Declare torch_geometric in setup.cfg install_requires** - `c66b17d` (feat)
2. **Task 2+3 RED: Add failing tests for models ops** - `056ff84` (test)
3. **Task 2+3 GREEN: Implement _ops.py with FPS/ball-query/radius-graph ops** - `f3aad21` (feat)

**Plan metadata:** (this commit) (docs: complete plan)

_Note: Tasks 2 and 3 were both marked `tdd="true"` and were executed as a single RED -> GREEN cycle since Task 3's test file was the implementation target for Task 2's code — see Decisions Made._

## Files Created/Modified
- `setup.cfg` - Added `torch_geometric` to `install_requires`
- `src/zreg/models/__init__.py` - Package marker, re-exports `farthest_point_sample`, `ball_query`, `build_radius_graph`
- `src/zreg/models/_ops.py` - The three Open3D-wrapped geometry ops (FPS, ball-query with isolated-point guard, radius-graph)
- `tests/test_zreg_models_ops.py` - 11 tests: FPS correctness (4 point-count sizes), ball-query self-fallback + never-empty, radius-graph shape/validity, D-03 benchmark (3 point-count sizes), device placement

## Decisions Made
- Combined Task 2+3's TDD cycle into one RED (test file) -> GREEN (implementation) pair rather than two independent tdd cycles, since Task 3's file was literally the test target for Task 2's code (see key-decisions in frontmatter for full rationale)
- `build_radius_graph` excludes self-loops by design (documented in the module docstring, satisfying the plan's "exclude self-loops or keep them consistently; document the choice" instruction) — isolated points get a self-loop fallback instead
- Reworded a docstring phrase that literally contained the substring `.cuda()` (in a sentence describing what the code does NOT do) to `bare CUDA device calls`, so the acceptance criterion's literal `grep -n "\.cuda()"` check returns zero matches without weakening the actual guarantee (still zero real `.cuda()` calls in the file)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Docstring mention of `.cuda()` tripped the acceptance-criterion grep**
- **Found during:** Task 2/3 GREEN verification
- **Issue:** The module docstring for `_ops.py` originally read "...no bare ``.cuda()`` calls..." — this is documentation prose, not a code call, but the plan's own acceptance criterion (`grep -n "\.cuda()" src/zreg/models/_ops.py` returns zero matches) is a literal text match and doesn't distinguish prose from code.
- **Fix:** Reworded to "no bare CUDA device calls" — same meaning, no literal `.cuda()` substring anywhere in the file.
- **Files modified:** `src/zreg/models/_ops.py`
- **Verification:** `grep -n "\.cuda()" src/zreg/models/_ops.py` now returns zero matches; full test suite still green.
- **Committed in:** `f3aad21` (part of Task 2/3 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Cosmetic wording fix only; no behavior change. No scope creep.

## Issues Encountered
- **Environment quirk (not a code defect):** the repo's editable pip install's `.pth` file points to the main repo checkout (`/Users/valeriekieslinger/Documents/Hiwi/BA/zReg/src`), not this worktree (`.claude/worktrees/elegant-mclaren-6c5b99/src`). Bare `python -c "import zreg.models..."` therefore resolves to the main repo's `src/zreg` (which lacks the new `models/` subpackage) and raises `ModuleNotFoundError`, while `pytest` resolves correctly because it auto-prepends the worktree's local `src/` to `sys.path`. All of the plan's literal inline `<verify>` commands were re-run with `PYTHONPATH` explicitly pointed at the worktree's `src/` to confirm correctness independent of this tooling quirk — all passed. The full `pytest` suite (which is the authoritative verification path and unaffected by this quirk) passes at 1259/1259+18 skipped with zero regressions.

## User Setup Required

None - no external service configuration required. `torch_geometric` was already installed and audited in Phase 45; this plan only added the `setup.cfg` declaration.

## Next Phase Readiness
- `src/zreg/models/_ops.py`'s three geometry ops are ready for `pointnet2.py` and `egnn.py` (later plans in this phase) to consume directly
- D-03 is fully resolved — no Open3D wrapper vectorization work is needed before building the model layers
- `torch_geometric` is now a declared dependency, unblocking `egnn.py`'s `MessagePassing` subclass work
- No blockers for the next plan in this phase

## Self-Check: PASSED

- FOUND: setup.cfg
- FOUND: src/zreg/models/__init__.py
- FOUND: src/zreg/models/_ops.py
- FOUND: tests/test_zreg_models_ops.py
- FOUND: .planning/phases/47-egnn-and-pointnet-model-implementation-training-infrastructu/47-01-SUMMARY.md
- FOUND commit: c66b17d
- FOUND commit: 056ff84
- FOUND commit: f3aad21

---
*Phase: 47-egnn-and-pointnet-model-implementation-training-infrastructu*
*Completed: 2026-07-13*
