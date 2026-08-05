# Phase 58: Synthetic Labeled Subsample-Pair Generation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-04
**Phase:** 58-synthetic-labeled-subsample-pair-generation
**Areas discussed:** Reuse vs. generalize generate_training_triple, Config surface shape, HPO seed strategy (plus an ad-hoc transform-conditionality sub-question and a train/val split elaboration)

---

## Reuse vs. Generalize `generate_training_triple`

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic-only, reuse as-is | Wire `generate_training_triple()` directly into EvaluationRunner/HyperparamOptimizer, ball/bowl geometry only | |
| Generalize to accept real loaded data | New DataFactory method accepting any already-loaded dataset (real or synthetic), producing a labeled subsample pair | ✓ |

**User's choice:** "Generalize to accept real loaded data"
**Notes:** Matches the original vision from the session's first discussion ("subsampling a larger (labeled) point cloud"). `generate_training_triple()` itself stays unchanged; this is a new sibling method.

---

## Transform Application (ad-hoc follow-up question)

| Option | Description | Selected |
|--------|-------------|----------|
| Both subsample AND transform | Always apply rigid/noise perturbation on top of subsampling | |
| Subsample only, no transform | Never apply a geometric transform | |
| Conditional (user's answer) | Transform applied only when run_alignment=True; subsample-only when only label-transfer is being tested | ✓ |

**User's choice:** "depends what stages will be run. It should be possible to do both subsampling and transformation for alignment synthetic data and ground truth but for only label transfer synthetic data there should be no transformation."
**Notes:** Derived from `config.run_alignment`, not a new config flag — avoids confounding label-transfer accuracy with alignment error when only label transfer is under test.

---

## Config Surface Shape

| Option | Description | Selected |
|--------|-------------|----------|
| New transform_spec type: "subsample_pair" | Sibling to today's rigid/noise, reuses pipeline_mode: synthetic dispatch | ✓ |
| New pipeline_mode value | Dedicated third pipeline_mode with its own config fields | |

**User's choice:** "New transform_spec type: 'subsample_pair'"

---

## HPO Seed Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| One fixed seed per run | Matches existing transform_spec behavior, no optimizer changes | ✓ (as default) |
| Fixed set of seeds, averaged per trial | More robust signal, ~N× compute per trial | (opt-in middle path) |

**User's choice:** First dismissed the initial AskUserQuestion ("[User dismissed — do not proceed, wait for next instruction]"), then asked for elaboration on the tradeoff. After elaboration (trial comparability, HPO surrogate-model noise sensitivity, budget concerns, and a proposed middle path), user said: "go with one fixed seed, default to the middle path" — interpreted as: default behavior is one fixed seed (D-06), with the seed-list/averaging option available as an explicit opt-in gated by tier (D-07), not the default.

**Follow-up:** User asked "How would a train/val split be implemented?" — Claude verified via codebase check that no existing search-then-validate pattern exists (`val_split`/`prepare_split`/`split_seeds` all currently unconsumed by the optimizer/runner), gave a recommendation (reuse `split_seeds()` to derive disjoint train/val seeds, add a post-HPO held-out re-evaluation step), and flagged it as new plumbing rather than reuse of an existing pattern. User decided: "defer it to a follow-up phase" (recorded as D-08, explicitly deferred).

---

## Claude's Discretion

- New method name/signature
- Exact transform_spec key names for subsample_pair (source_fraction/target_fraction, real-vs-synthetic selector)
- Tier-gating mechanism for multi-seed averaging

## Deferred Ideas

- Train/val split for HPO against subsample_pair ground truth (D-08) — new EvalReport field + new post-HPO re-evaluation call site, deferred to a follow-up phase
