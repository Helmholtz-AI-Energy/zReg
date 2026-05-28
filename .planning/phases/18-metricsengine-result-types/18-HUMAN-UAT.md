---
status: partial
phase: 18-metricsengine-result-types
source: [18-VERIFICATION.md]
started: 2026-05-28T08:35:00Z
updated: 2026-05-28T08:35:00Z
---

## Current Test

[awaiting human decision]

## Tests

### 1. CR-01: normalize() out-of-range values for negative metric inputs
expected: Either pydantic rejects negative inputs via `ge=0` Field validators on StageMetrics lower-is-better fields, OR normalize() clamps output to [0,1], OR developer accepts this as a known limitation for a research codebase where upstream zreg.metrics.* primitives always return non-negative values.
result: [pending]

### 2. CR-02: compute_score() invariant broken for partial normalized dicts
expected: When metric_weights contains a key absent from metrics.normalized, either: (a) total is computed over only the present keys so score remains in [0,1], OR (b) developer documents that metrics.normalized is always expected to be fully populated before calling compute_score().
result: [pending]

### 3. CR-03: sanity_check() false-positive sentinel warning on empty tensors
expected: An empty label tensor (torch.tensor([], dtype=torch.long)) should not trigger the "all-sentinel" flag due to PyTorch vacuous truth semantics. Either an explicit numel()==0 guard is added, or the behavior is accepted.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
