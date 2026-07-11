---
phase: 45-egnn-pointnet-label-transfer-framework-selection-evaluation-
plan: 01
subsystem: infra
tags: [torch_geometric, egnn, pointnet2, open3d, design-doc, label-transfer]

# Dependency graph
requires:
  - phase: 44-cpd-weighted-label-transfer-method
    provides: OPTIONAL_PARAMS + EvalConfig.label_transfer_method config-wiring precedent for VALID_METHODS dispatch
provides:
  - "45-DESIGN.md — locked per-model library decision (PointNet++ hand-rolled on Open3D; eGNN hand-rolled on torch_geometric.nn.MessagePassing)"
  - "Verified torch_geometric 2.8.0 core install (pure-Python wheel, no compiled extension)"
  - "Documented zreg-before-torch_geometric import-order requirement (extends existing macOS-ARM libomp workaround)"
  - "Downstream phase ownership map for Phases 46-49"
affects: [46-training-data-pipeline, 47-model-training-infra, 48-labeltransferstage-integration, 49-evaluation-benchmarking]

# Tech tracking
tech-stack:
  added: [torch_geometric==2.8.0 (environment-only, not yet a declared setup.cfg dependency)]
  patterns:
    - "Open3D-wrapped non-differentiable geometry ops (FPS/ball-query), mirroring src/zreg/registration/icp.py's wrapper pattern"
    - "torch_geometric.nn.MessagePassing scaffolding for hand-rolled equivariant math, never for compiled-extension-dependent convenience functions (fps/radius/knn_interpolate)"
    - "zreg (or any scipy-importing module) must be imported before torch_geometric to avoid libomp SIGABRT, extending the existing zreg-before-torch convention"

key-files:
  created:
    - .planning/phases/45-egnn-pointnet-label-transfer-framework-selection-evaluation-/45-DESIGN.md
  modified: []

key-decisions:
  - "PointNet++ hand-rolled on Open3D farthest_point_down_sample + KDTreeFlann.search_radius_vector_3d; eGNN hand-rolled MessagePassing subclass on torch_geometric — resolves D-02 per-model"
  - "torch_geometric core only in Phase 46/47's setup.cfg; torch_cluster/torch_scatter/torch_sparse/pyg-lib permanently rejected (live build-failure reproduction)"
  - "New src/zreg/models/ package (Phase 47) mirrors registration/ and cpd/ per-method-family pattern; color_transfer.py stays unchanged"
  - "Joint-cloud conditioning (concat source+target, one-hot label channel on source, zero/unknown on target) required for both models"
  - "Import order: zreg (or scipy) must precede torch_geometric import to avoid libomp SIGABRT — new finding, extends existing project convention"

patterns-established:
  - "Design-only phase deliverable pattern: locked decisions + phase-attributed ownership map + carried-forward open questions/security notes, zero production code"

requirements-completed: [D-01, D-02, D-03, D-04]

# Metrics
duration: 12min
completed: 2026-07-11
---

# Phase 45 Plan 01: eGNN & PointNet++ Framework Selection Design Summary

**Locked per-model library decision (PointNet++ hand-rolled on Open3D FPS/ball-query; eGNN hand-rolled MessagePassing subclass on torch_geometric 2.8.0) plus a verified, clean torch_geometric core install and a complete phase-46-49 ownership map — zero production model code.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-07-11T08:59:00Z
- **Completed:** 2026-07-11T09:02:00Z
- **Tasks:** 2 completed
- **Files modified:** 1 created (`45-DESIGN.md`); 0 repository source files touched

## Accomplishments
- Positively verified `torch_geometric` (core, pure-Python wheel) installs cleanly on this dev machine with no build/compile step, complementing RESEARCH.md's live reproduction of the `torch_cluster` build failure.
- Discovered and documented a new import-order requirement (`zreg` before `torch_geometric`) needed to avoid the same libomp SIGABRT this codebase already works around for `zreg`/`torch` — carried forward into `45-DESIGN.md` for Phase 46/47.
- Authored `45-DESIGN.md`, the single authoritative synthesis of `45-CONTEXT.md` (D-01..D-04) and `45-RESEARCH.md`, covering all 11 required sections and attributing every locked decision to its executing downstream phase (46-49).
- Confirmed zero production model/training/dispatch code was introduced — `git status` shows only the new design document.

## Task Commits

Each task was committed atomically:

1. **Task 1: Verify torch_geometric (core) installs cleanly and record the evidence** - no repository files modified (install-only task per plan scope); evidence embedded directly into Task 2's commit below.
2. **Task 2: Write 45-DESIGN.md — the locked architecture/library decisions for Phases 46-49** - `6c3506e` (docs)

**Plan metadata:** (this commit, produced after SUMMARY.md/STATE.md/ROADMAP.md updates)

_Note: Task 1 produced no file changes per its own `<files>` spec ("no repository files modified") — its evidence (torch_geometric version, import-OK result, import-order caveat) is embedded verbatim in `45-DESIGN.md`'s `## Dependency Decision` section, committed as part of Task 2._

## Files Created/Modified
- `.planning/phases/45-egnn-pointnet-label-transfer-framework-selection-evaluation-/45-DESIGN.md` - Locked architecture/library decisions for Phases 46-49 (251 lines, 11 required sections, all required tokens present)

## Decisions Made
- **PointNet++ = hand-rolled on Open3D** (`farthest_point_down_sample` + `KDTreeFlann.search_radius_vector_3d`), wrapped the way `src/zreg/registration/icp.py` wraps Open3D's ICP. Rationale: `torch_cluster` reproducibly fails to build on this machine (RESEARCH.md, re-confirmed not re-attempted this plan per explicit instruction).
- **eGNN = hand-rolled `MessagePassing` subclass on `torch_geometric`** (core only). Rationale: pure-Python wheel installs cleanly (verified this plan); avoids dense O(N²) third-party EGNN packages unfit for this repo's larger synthetic dataset sizes.
- **`torch_geometric` stays environment-only in this phase** — not added to `setup.cfg`; that's explicitly Phase 46/47's job.
- **New finding, not previously documented:** importing `torch_geometric` in a fresh process (no prior `zreg`/`scipy` import) reproduces the exact `OMP: Error #15` libomp SIGABRT already documented in `tests/conftest.py:20-24`. The fix — import `zreg` (or any `scipy`-importing module) before `torch_geometric` — mirrors the existing "zreg before torch" convention exactly, and is now recorded in `45-DESIGN.md`'s `## Dependency Decision` section for Phase 46/47 to follow.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan's literal Task 1 verify command required an import-order workaround to run without crashing**
- **Found during:** Task 1 (torch_geometric install verification)
- **Issue:** The plan's exact verify command (`python -c "import torch_geometric; from torch_geometric.nn import MessagePassing, MLP; print('TG_OK', ...)"`, run as the very first import in a fresh process) crashed with `OMP: Error #15` (libomp SIGABRT, exit code 134) — the identical failure mode this codebase already documents and works around elsewhere (`tests/conftest.py:20-24`, `eval/data_factory.py:18-35`) for `zreg` vs. `torch` import order.
- **Fix:** Ran the equivalent verification with `from zreg.dataset import zRegPointCloud` imported first (pulling in `scipy` before `torch_geometric`), which succeeded and printed `TG_OK 2.8.0`. No repository code was changed to "fix" this — it is a documented import-order convention, not a bug in the design or in `torch_geometric` itself. The finding was carried forward into `45-DESIGN.md`'s `## Dependency Decision` section as an explicit requirement for Phase 46/47's code.
- **Files modified:** None (verification-only; the finding is documented in `45-DESIGN.md`, committed as part of Task 2's commit `6c3506e`).
- **Verification:** `python -c "from zreg.dataset import zRegPointCloud; import torch_geometric; from torch_geometric.nn import MessagePassing, MLP; print('TG_OK', torch_geometric.__version__)"` → `TG_OK 2.8.0`.
- **Committed in:** `6c3506e` (finding recorded in Task 2's `45-DESIGN.md` commit; no separate commit needed since no code changed).

---

**Total deviations:** 1 auto-fixed (1 blocking, Rule 3)
**Impact on plan:** No scope creep — the workaround is a documented import-order convention already established elsewhere in this codebase, now extended to cover the newly-installed `torch_geometric` dependency. All of the plan's acceptance criteria (pure-Python wheel install, no `setup.cfg` change, no `torch_cluster`/etc. install, `TG_OK` + version string captured) are satisfied.

## Issues Encountered
None beyond the deviation documented above.

## User Setup Required

None - no external service configuration required. `torch_geometric` was installed locally into this project's existing Python environment via `pip install torch_geometric`; no API keys, dashboards, or manual steps were involved.

## Next Phase Readiness

`45-DESIGN.md` is the complete, locked blueprint Phases 46-49 execute against:
- Phase 46 (training-triple data pipeline) can proceed against the `label`-vs-`id` discipline and synthetic-only training data source sections.
- Phase 47 (model + training infra) can proceed against the per-model library decision, target module structure, joint-cloud conditioning pattern, and the newly-documented `zreg`-before-`torch_geometric` import-order requirement.
- Phase 48 (LabelTransferStage integration) can proceed against the exact `VALID_METHODS` values (`"egnn"`, `"pointnet++"`) and the `weights_only=True` checkpoint-security mandate.
- Phase 49 (evaluation/benchmarking) has Claude's-Discretion pointers (generalization dimensions, CPU-latency benchmarking, train/eval leakage guard) to build its own eval-strategy design from.

No blockers. `git status` confirms zero production model/training/dispatch code was introduced this plan — the "no production code in this phase" constraint (per ROADMAP.md) is honored.

---
*Phase: 45-egnn-pointnet-label-transfer-framework-selection-evaluation-*
*Completed: 2026-07-11*

## Self-Check: PASSED

- FOUND: `.planning/phases/45-egnn-pointnet-label-transfer-framework-selection-evaluation-/45-DESIGN.md`
- FOUND: `.planning/phases/45-egnn-pointnet-label-transfer-framework-selection-evaluation-/45-01-SUMMARY.md`
- FOUND: commit `6c3506e` in `git log --oneline --all`
