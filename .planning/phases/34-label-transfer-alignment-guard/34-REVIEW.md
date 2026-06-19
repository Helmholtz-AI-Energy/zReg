---
phase: 34-label-transfer-alignment-guard
reviewed: 2026-06-19T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - eval/types.py
  - eval/stages/label_transfer.py
  - tests/test_label_transfer_stage.py
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: issues_found
---

# Phase 34: Code Review Report

**Reviewed:** 2026-06-19
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 34 adds a pre-transfer alignment guard to `LabelTransferStage`: a new `pre_transfer_alignment: float` field on `LabelResult`, a `_check_alignment()` static method that computes mean Chamfer distance across frame pairs, and a conditional warn/log in `run()` gated on `config.run_alignment`. The overall design is sound and the new `TestLabelTransferAlignmentGuard` test class adequately covers the four behavioural branches (warn vs log, aligned vs misaligned).

One critical issue exists: `_check_alignment()` can return `nan` when any paired frame has zero points, which causes pydantic to reject the `LabelResult` construction with a `ValidationError` whose message gives no indication of the root cause. Two warnings cover a missing docstring entry and unreachable dead code. Two informational notes cover a misleading comment and a semantic ambiguity in the model field.

---

## Critical Issues

### CR-01: `_check_alignment` produces `nan` for zero-point frames, causing misleading `ValidationError` crash

**File:** `eval/stages/label_transfer.py:171-177`

**Issue:** `_check_alignment` calls `chamfer(src_pos, tgt_pos)` with no guard against empty point clouds. When `src_pos` has shape `(0, 3)`, `torch.cdist` returns shape `(0, M)`, `min(dim=1).values` has shape `(0,)`, and `.mean()` returns `nan` (confirmed via runtime check). The computed `total` becomes `nan`, so `_check_alignment` returns `nan`. Downstream, `LabelResult(pre_transfer_alignment=nan)` fails pydantic's `ge=0.0` constraint with:

```
ValidationError: pre_transfer_alignment — Input should be greater than or equal to 0 [type=greater_than_equal, input_value=nan]
```

This message is wholly misleading — it looks like a configuration bug, not an empty-frame bug. The existing `k_neighbours > n_src` guard at line 276 only fires inside the transfer loop, well after `_check_alignment` has already crashed. `_validate_tensors` inside `chamfer()` does not reject empty tensors (`.all()` on an empty tensor is vacuously `True`).

**Fix:** Add an early check inside `_check_alignment` before accumulating the Chamfer sum:

```python
for k in range(n_pairs):
    src_pos = source[source_keys[k]]["pos"]
    tgt_pos = target[target_keys[k]]["pos"]
    if src_pos.shape[0] == 0 or tgt_pos.shape[0] == 0:
        # Skip empty frames rather than returning nan
        continue
    dist = chamfer(src_pos, tgt_pos)
    total += float(dist.item() if hasattr(dist, "item") else dist)
```

Alternatively, update the denominator to count only non-skipped pairs and return `0.0` when all pairs are skipped. Either way, the `nan` path into pydantic must be eliminated.

---

## Warnings

### WR-01: `run()` docstring omits `pre_transfer_alignment` from the Returns section

**File:** `eval/stages/label_transfer.py:201-207`

**Issue:** The `Returns` section of `run()` lists only `transferred_labels` and `params_used`. It does not mention `pre_transfer_alignment`, which is now always populated. A caller reading only the docstring will be unaware that the returned `LabelResult` carries an alignment quality metric.

```python
# Current Returns section (lines 201-207):
Returns
-------
LabelResult
    Pydantic-frozen result with:
    - ``transferred_labels``: per-frame label tensors, keyed by
      target frame index.
    - ``params_used``: shallow copy of ``params`` (Pitfall 7).
```

**Fix:** Extend the Returns block:

```python
Returns
-------
LabelResult
    Pydantic-frozen result with:
    - ``transferred_labels``: per-frame label tensors, keyed by
      target frame index.
    - ``params_used``: shallow copy of ``params`` (Pitfall 7).
    - ``pre_transfer_alignment``: mean Chamfer distance between
      source and target frames, computed before label transfer.
      Logged or warned depending on ``config.run_alignment``.
```

---

### WR-02: Dead code branch in `_check_alignment` — `n_pairs == 0` path is unreachable

**File:** `eval/stages/label_transfer.py:177`

**Issue:** The docstring comment says "Returns 0.0 when there are no paired frames" and line 177 guards `return total / n_pairs if n_pairs > 0 else 0.0`. However, `_check_alignment` is only ever called from `run()` at line 232, which is unconditionally after the non-empty guards at lines 227-230:

```python
if not source:
    raise ValueError("source must be non-empty; got 0 frames")
if not target:
    raise ValueError("target must be non-empty; got 0 frames")

alignment_dist = self._check_alignment(source, target)  # both non-empty here
```

Since both `source` and `target` are guaranteed non-empty by the time `_check_alignment` is called, `min(len(source_keys), len(target_keys)) >= 1`, so `n_pairs > 0` is always `True`. The `else 0.0` branch is dead code. This is not harmful but the docstring ("Returns 0.0 when there are no paired frames") describes behaviour that can never be triggered through the public `run()` API.

**Fix:** Either remove the dead-code branch and update the docstring to reflect this, or make `_check_alignment` a truly standalone utility (not just called from `run()`) that can be invoked with empty dicts independently. If the latter, add a unit test that calls `_check_alignment({}, {})` directly.

---

## Info

### IN-01: `noqa: F401` comment on `torch` import is inaccurate

**File:** `eval/stages/label_transfer.py:46`

**Issue:** The comment reads `import torch  # noqa: F401 — ensures consistent import order for downstream callers`. But `torch` is actually used at line 268 in the type annotation `transferred: dict[int, torch.Tensor] = {}`. It is not an unused import — `F401` would not fire. The comment is therefore misleading about why the import is present.

**Fix:** Remove the `# noqa: F401` suppression and update the comment to reflect the actual use:

```python
import torch  # used for dict[int, torch.Tensor] annotation and downstream callers
```

---

### IN-02: `pre_transfer_alignment = 0.0` is semantically ambiguous in `LabelResult`

**File:** `eval/types.py:132, 145`

**Issue:** The docstring says "Value of 0.0 indicates identical clouds or unchecked (default)." These are two different semantics:
1. The alignment check ran and the source and target are co-located (genuine 0.0).
2. The `LabelResult` was constructed manually (e.g. in tests or older pipeline code) without populating the field, so it defaults to 0.0.

Downstream consumers — e.g. an evaluation dashboard or a sanity checker — cannot distinguish a perfectly pre-aligned input from an unchecked one. Sixteen existing tests across `test_optimizer.py`, `test_metrics.py`, `test_viz.py`, etc. construct `LabelResult` directly and silently receive `pre_transfer_alignment=0.0`, which could mislead any future consumer of this field.

**Fix (optional):** Use `Optional[float]` with `default=None` to make the "unchecked" state explicit, or add a separate boolean flag `alignment_checked: bool = False`. If the ambiguity is an accepted design trade-off, add a note in the docstring explicitly discouraging treating `0.0` as evidence of alignment quality.

---

_Reviewed: 2026-06-19_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
