# Phase 30: Two-Dataset Paired Alignment Architecture - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 30-two-dataset-paired-alignment-architecture
**Areas discussed:** LabelTransferStage update scope, paired_alignment.yaml content, Backward compat for existing EvaluationRunner tests

---

## LabelTransferStage update scope

| Option | Description | Selected |
|--------|-------------|----------|
| New (source, target, params) signature | Explicit; stage owns the distinction; EvaluationRunner just passes both through. Matches the AlignmentStage change pattern. | ✓ |
| Keep existing signature unchanged | EvaluationRunner._run_single merges what the stage needs before calling it. Smaller diff but hides the two-dataset distinction inside the runner. | |

**User's choice:** New (source, target, params) signature

**Follow-up — transfer logic:**

| Option | Description | Selected |
|--------|-------------|----------|
| Sequential: source[k] → target[k] | Same sequential pattern but across datasets. No warping path needed. | ✓ |
| Warping-path-aware | Use the DTW warping path to map source frames to target frames (not necessarily 1:1). Requires passing align_result into LabelTransferStage. | |

**User's choice:** Sequential: source[k] → target[k]
**Notes:** Warping-path-aware transfer deferred to a future phase.

---

## paired_alignment.yaml content

| Option | Description | Selected |
|--------|-------------|----------|
| Same dataset twice (smoke test) | Use Kobitski tracklets as both data_path and target_data_path. Proves pipeline mechanics without needing a real paired dataset. | ✓ |
| Kobitski (source) + Shah (target) | Two real datasets, different formats. Most realistic but requires load_target() to handle a different data_format or a target_data_format field. | |
| Kobitski + different Kobitski file | Same format, different sample. Clean paired scenario without format complexity. | |

**User's choice:** Same dataset twice (smoke test)
**Notes:** A true heterogeneous paired config (Kobitski + Shah) deferred until target_data_format is supported.

---

## Backward compat for existing EvaluationRunner tests

| Option | Description | Selected |
|--------|-------------|----------|
| Patch tests: mock load_target() | Update existing EvaluationRunner tests to mock DataFactory.load_target() returning a fixture. Minimal change per test; keeps production path clean. | ✓ |
| Guard in runner: skip load_target if path is None | EvaluationRunner.run() falls back to single-dataset behavior when target_data_path is None. No test changes needed but adds a silent fallback. | |

**User's choice:** Patch tests: mock load_target()
**Notes:** No production-path fallback added. paired mode contract is enforced cleanly.

---

## Claude's Discretion

- **EvalConfig validation timing** — EvalConfigError raised at `load_target()` call time (not at construction), consistent with existing codebase error-at-use-time convention.
- **AlignResult structure** — unchanged; no source/target fields added in this phase.
- **load_target() implementation** — near-copy of load_real() reading from `target_data_path`. Optional shared private helper (`_load_from_path`) acceptable if clean.

## Deferred Ideas

- **Heterogeneous paired scenario config** (Kobitski + Shah with different formats) — needs `target_data_format` field on EvalConfig or format inference from extension.
- **Warping-path-aware label transfer** — use `AlignResult.warping_path` to map non-1:1 source→target frame correspondence in LabelTransferStage. Future phase.
