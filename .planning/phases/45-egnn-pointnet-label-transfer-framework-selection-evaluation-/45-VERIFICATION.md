---
phase: 45-egnn-pointnet-label-transfer-framework-selection-evaluation-
verified: 2026-07-11T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 45: eGNN & PointNet++ Label Transfer — Framework Selection & Evaluation Strategy Verification Report

**Phase Goal:** Decide the model architecture/library approach for eGNN and PointNet++ as new
LabelTransferStage methods, research each architecture's implementation requirements, and design
an evaluation strategy for learned label-transfer methods. Produces a design document
(45-DESIGN.md) synthesizing the locked decisions for Phases 46-49 — no production code in this
phase.

**Verified:** 2026-07-11
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `torch_geometric` (core, pure-Python) is confirmed installable on this project's dev machine, and `torch_geometric.nn.MessagePassing`/`MLP` import successfully | ✓ VERIFIED | Independently re-ran (not trusting SUMMARY): `python -c "from zreg.dataset import zRegPointCloud; import torch_geometric; from torch_geometric.nn import MessagePassing, MLP; print('TG_OK', torch_geometric.__version__)"` → `TG_OK 2.8.0` |
| 2 | A locked design document exists that fixes the per-model library decision for Phases 46-49 (PointNet++ hand-rolled on Open3D ops; eGNN hand-rolled on `torch_geometric.nn.MessagePassing`) | ✓ VERIFIED | `45-DESIGN.md` "Locked Library Decision (per model)" section (lines 11-42) states exactly this, with rationale citing the live `torch_cluster` build failure from RESEARCH.md |
| 3 | The design document specifies the target `src/zreg/models/` module layout, the joint-cloud conditioning adaptation both models need, the label-vs-id discipline, and the GPU-train/CPU-infer split | ✓ VERIFIED | Sections present: "Target Module Structure" (79-97), "Joint-Cloud Conditioning" (99-125), "label vs id Discipline" (127-143), "Train / Infer Split" (145-160) |
| 4 | The design document maps each locked decision to the downstream phase (46/47/48/49) that executes it, and carries the research's open questions + checkpoint-security note forward | ✓ VERIFIED | "Downstream Phase Ownership Map" table (200-207) attributes every decision to Phase 46/47/48/49; "Open Questions Carried Forward" (209-227) reproduces all 3 RESEARCH.md open questions plus the `weights_only=True` checkpoint-deserialization security note |
| 5 | No PointNet++/eGNN model code, training loop, or LabelTransferStage dispatch is written in this phase | ✓ VERIFIED | `git diff --stat c5a55d2~1 3d0c610 -- src/ eval/` → empty (zero changes to any `src/` or `eval/` file across all 3 phase-45 commits: `c5a55d2`, `7b1ef39`, `6c3506e`, `3d0c610`). `find src/zreg/models` → no such directory. `git diff --stat setup.cfg` since before phase 45 → no diff (torch_geometric not added as a declared dependency). `git status --porcelain` → clean. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `45-DESIGN.md` | Locked architecture/library decisions synthesizing 45-CONTEXT.md + 45-RESEARCH.md for Phases 46-49; contains `farthest_point_down_sample`; ≥120 lines | ✓ VERIFIED | 251 lines (exceeds 120-line minimum); contains `farthest_point_down_sample` (line 17); all 11 required section headings present (grep loop passed with `DESIGN_OK`); all 7 required tokens present (`farthest_point_down_sample`, `torch_cluster`, `MessagePassing`, `VALID_METHODS`, `pc["label"]`, `src/zreg/models`, `weights_only`) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `45-DESIGN.md` | `45-RESEARCH.md` | cites the `torch_cluster` build-failure finding and Standard Stack recommendation | ✓ WIRED | Lines 28-39 directly cite the live `torch_cluster` build failure (`OMP: Error #15`), staleness (2023-10-12), and no-macOS-ARM-wheel finding from RESEARCH.md's Summary/Standard Stack sections |
| `45-DESIGN.md` | `eval/stages/label_transfer.py` | names the Phase 44 `VALID_METHODS`/`OPTIONAL_PARAMS` dispatch that Phase 48 extends | ✓ WIRED | Lines 155-160 name the exact current tuple `VALID_METHODS: tuple = ("knn_voting", "cpd_weighted")`, independently confirmed against `eval/stages/label_transfer.py:104` — matches verbatim — and specifies the two new values Phase 48 adds (`"egnn"`, `"pointnet++"`) |

### Requirements Coverage

No formal `.planning/REQUIREMENTS.md` IDs exist for Phase 45 (ad hoc phase, per 45-CONTEXT.md and 45-RESEARCH.md's own Test Map note: "ROADMAP.md lists 'Requirements: TBD' for Phases 45-49"). Verified against the 4 CONTEXT.md decision IDs instead:

| Requirement | Source | Description | Status | Evidence |
|---|---|---|---|---|
| D-01 | 45-CONTEXT.md | Build eGNN and PointNet++ together, in parallel, across Phases 46-49 | ✓ SATISFIED | Implicit throughout DESIGN.md (every section addresses both models jointly); explicitly cited in "Source Decisions" closing line |
| D-02 | 45-CONTEXT.md | Decide per-model library approach during this phase's research | ✓ SATISFIED | "Locked Library Decision (per model)" section resolves this per-model, not uniformly, with rationale |
| D-03 | 45-CONTEXT.md | Training data solely from synthetic bowl/ball growth-model generator | ✓ SATISFIED | "Training Data Source" section locks this, attributes extension to Phase 46, notes Kobitski/Shah stay eval-only |
| D-04 | 45-CONTEXT.md | Training assumes GPU; inference stays CPU-friendly | ✓ SATISFIED | "Train / Infer Split" section locks GPU-training/CPU-inference split, attributes to Phase 47 (train) / Phase 48 (infer) |

No orphaned requirements — all 4 D-IDs from CONTEXT.md are addressed in DESIGN.md.

### Anti-Patterns Found

Scanned `45-DESIGN.md` (the only file this phase created) for debt markers and stub indicators:

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `45-DESIGN.md` | 150 | `"TBD at that phase"` (re: exact location of the Phase 47 training entry point — `scripts/` vs. new `eval/training/`) | ℹ️ Info | Not a blocker: this is an explicit, phase-attributed deferral of an implementation-location decision to Phase 47, consistent with this document's entire purpose (locking decisions in scope for Phase 45, deferring implementation-detail decisions explicitly out of scope to the phase that will execute them). No unresolved ambiguity about *this phase's* deliverable. |

No fenced Python code blocks exist in `45-DESIGN.md` (confirmed via search for ` ```python `) — consistent with the plan's constraint that the design doc contain no working model/training-loop implementations, only prose decisions (RESEARCH.md's illustrative code skeletons were deliberately NOT reproduced in DESIGN.md).

No other `FIXME`/`XXX`/`HACK`/`PLACEHOLDER`/"not yet implemented" markers found.

### Human Verification Required

None. This phase's entire deliverable (a design document plus an install-verification result) is
fully verifiable by direct inspection and command execution — no visual, real-time, or subjective-UX
judgment is required.

### Gaps Summary

No gaps. Both of the extra-rigor checks requested were independently re-verified (not trusting
SUMMARY.md):

1. **No production code outside `.planning/`:** `git diff --stat` across all Phase 45 commits
   (`c5a55d2` → `3d0c610`) touching `src/` or `eval/` is empty. No `src/zreg/models/` directory
   exists. `eval/stages/label_transfer.py` was last modified in Phase 44 (`1b352ac`), untouched by
   Phase 45. Working tree is clean (`git status --porcelain` → no output).

2. **Import-order deviation and dependency-declaration claim accurately reflected:** `45-DESIGN.md`'s
   "Dependency Decision" section (lines 44-77) documents the `zreg`-before-`torch_geometric`
   import-order fix, correctly attributes it as a new finding extending the existing
   "zreg-before-torch" convention (`tests/conftest.py:20-24`), and independently re-running the
   exact import sequence confirms `TG_OK 2.8.0`. `setup.cfg` shows zero diff since before Phase 45 —
   `torch_geometric` is correctly NOT a declared dependency yet; the document explicitly states this
   is Phase 46/47's job.

---

_Verified: 2026-07-11_
_Verifier: Claude (gsd-verifier)_
