# Phase 57: Ground-Truth Field Consistency - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-01
**Phase:** 57-ground-truth-field-consistency
**Areas discussed:** GT field convention, Dropout correspondence tracking, Shah/Kobitski config audit blocker

---

## GT Field Convention

| Option | Description | Selected |
|--------|-------------|----------|
| Hard rule: always label | GT extraction always reads `pc["label"]`; no config knob | |
| Configurable ground_truth_field | Add `EvalConfig` switch, `id`/`label` fully user-selectable | |
| Hybrid (user's answer) | Default to `label`, but overridable per-dataset | ✓ |

**User's choice:** "always label but overridable in case dataset has a differently named label field"
**Notes:** Interpreted as `ground_truth_field: Literal["id","label"] = "label"` on `EvalConfig` — the only two candidate fields on `zRegPointCloud`, default `"label"`.

---

## Dropout Correspondence Tracking

| Option | Description | Selected |
|--------|-------------|----------|
| Full correspondence tracking | Thread real original-index map through drop_points/sample_new_points | ✓ |
| Detect-and-guard | Warn/error on point-count divergence instead of fixing the pairing | |

**User's choice:** "Full correspondence tracking"
**Notes:** Explicitly framed as reusable groundwork for Phase 58's subsample-pair mechanism, which needs the same correspondence-tracking capability.

---

## Shah/Kobitski Config Audit Blocker

| Option | Description | Selected |
|--------|-------------|----------|
| You tell me now | User supplies the real answer directly, since no data files exist in this worktree | ✓ |
| Defensive check + defer verification | Add a degeneracy sanity check instead, defer real verification to user's own machine | |

**User's choice:** "shah has ID and label but only label is the correct one for label transfer. Kobitski only has ID which is not suitable for label transfer. In other words it is only suited as target data in label transfer."
**Notes:** This corrects a stale comment in `configs/experiments/stage1_alignment/rigid_cpd/synthetic.yaml` that claims Shah's `id` (not `label`) carries real classes — that comment predates this correction and needs fixing during the audit. The Kobitski-as-target idea was captured as a deferred idea (new capability, not this phase's scope).

---

## Claude's Discretion

- Exact data structure for correspondence tracking (index tensor vs. stored attribute)
- Whether to fix the fully-synthetic ball/bowl configs' degenerate placeholder `label` within this phase's audit scope, or leave flagged

## Deferred Ideas

- Kobitski as a label-transfer *target* dataset (heterogeneous paired scenario, building on Phase 32's HETERO-01) — new capability, future phase
