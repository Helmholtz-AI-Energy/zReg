---
phase: 14-synthetic-data-generators
plan: "01"
subsystem: eval/generators
tags:
  - synthetic-data
  - generators
  - point-cloud
  - eval-framework
dependency_graph:
  requires:
    - src/zreg/dataset.py (zRegPointCloud)
    - src/zreg/transforms/ (RigidTransformation, AffineTransformation)
    - src/zreg/metrics/alignment.py (4x4 matrix construction pattern)
  provides:
    - eval/generators/__init__.py — importable package with __all__
    - eval/generators/generators.py — generate_trajectory factory
    - eval/generators/transforms.py — apply_rigid, apply_affine wrappers
  affects:
    - Plan 02 (corruption + labels): extends eval/generators/__init__.py __all__
    - Plan 03 (tests + conftest): pytest coverage for all six public functions
tech_stack:
  added:
    - eval/generators/ Python package (torch-only, no numpy, no open3d)
  patterns:
    - Phase 13 4x4 matrix construction pattern (src/zreg/metrics/alignment.py)
    - transform_points_homogeneous w-divide pattern (src/zreg/transforms/homogeneous.py)
    - copy.deepcopy immutability (D-03 contract)
    - seed: int | None = 42 contract (torch.manual_seed only when seed is not None)
key_files:
  created:
    - eval/generators/__init__.py
    - eval/generators/generators.py
    - eval/generators/transforms.py
  modified: []
decisions:
  - "AffineTransformation() default t=[1,1,1] is not identity — plan acceptance criteria contain factual error; implementation is correct and verified with t=torch.zeros(3)"
  - "apply_affine and apply_rigid share _apply_matrix private helper for DRY matrix application"
  - "Defensive w-divide (clamp to finfo.eps) retained from transform_points_homogeneous pattern even for rigid transforms where w=1 is guaranteed"
metrics:
  duration_seconds: 181
  completed_date: "2026-05-18"
  tasks_completed: 3
  tasks_total: 3
  files_created: 3
  files_modified: 0
---

# Phase 14 Plan 01: Synthetic Data Generators — Factory and Transform Wrappers Summary

**One-liner:** Deterministic Gaussian-blob trajectory factory and immutable rigid/affine wrappers using torch-only homogeneous coordinate transforms and copy.deepcopy immutability.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create eval/generators/__init__.py package skeleton | 19edc7b | eval/generators/__init__.py |
| 2 | Implement generate_trajectory factory | 9567467 | eval/generators/generators.py |
| 3 | Implement apply_rigid / apply_affine wrappers | a4d8889 | eval/generators/transforms.py |

## Final Function Signatures

```python
# eval/generators/generators.py
def generate_trajectory(
    n_points: int,
    n_frames: int,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]: ...

# eval/generators/transforms.py
def apply_rigid(
    trajectory: dict[int, zRegPointCloud],
    tf: RigidTransformation,
) -> dict[int, zRegPointCloud]: ...

def apply_affine(
    trajectory: dict[int, zRegPointCloud],
    tf: AffineTransformation,
) -> dict[int, zRegPointCloud]: ...
```

## Defensive w-divide Pattern Decision

The defensive divide-by-w pattern (clamped to `torch.finfo(pos.dtype).eps`) from `transform_points_homogeneous` was **retained** in `apply_rigid` and `apply_affine`. Rationale:

- For rigid transforms the w-coordinate is always 1.0, so the divide is numerically a no-op
- Keeping it matches the canonical helper's contract and guards against floating-point edge cases
- It adds one clamp+divide op per call (negligible cost vs. clarity and correctness guarantee)
- Future callers composing transforms that have near-zero w values will be protected

## Deviations from Plan

### [Rule 1 - Bug] Plan acceptance criteria for apply_affine identity test are incorrect

**Found during:** Task 3 implementation verification

**Issue:** The plan's behavior test 2 and acceptance criteria state that `apply_affine(trajectory, AffineTransformation())` leaves `pos` unchanged (i.e., the default `AffineTransformation()` is an identity). This is factually wrong: `AffineTransformation()` defaults to `b=I, t=[1, 1, 1]` (a translation by ones — see `src/zreg/transforms/affine.py` line 62). An identity affine requires `AffineTransformation(t=torch.zeros(3))`.

**Fix:** Implementation is correct — `apply_affine` applies the matrix exactly as specified. The verification test was run with `AffineTransformation(t=torch.zeros(3))` for the identity check. The plan's inline verification code (`AffineTransformation()` expecting `pos` unchanged) would fail at runtime — this is a plan authoring error, not an implementation error.

**Impact:** Plan 03 test file will use `AffineTransformation(t=torch.zeros(3))` for the identity smoke test. `AffineTransformation()` (with default `t=[1,1,1]`) will be tested as a translation-by-ones transform, not as identity.

**Files modified:** None (implementation is correct; only plan verification script is wrong)

## Known Stubs

None — all three functions produce real output. `generate_trajectory` emits actual `torch.randn` tensors; `apply_rigid` and `apply_affine` apply real matrix transforms.

## Plan Note

`eval/generators/__init__.py` currently exports only three symbols (`generate_trajectory`, `apply_rigid`, `apply_affine`). Plan 02 is responsible for extending it with `add_gaussian_noise`, `add_outliers`, `generate_labels`, and `remove_labels` imports and appending those names to `__all__`.

## Threat Flags

No new network endpoints, auth paths, or trust boundary changes introduced. The `eval/generators/` package is pure computation (torch tensors, copy operations) with no I/O, network access, or external data loading.

## Self-Check: PASSED

Files exist:
- eval/generators/__init__.py: FOUND
- eval/generators/generators.py: FOUND
- eval/generators/transforms.py: FOUND
- eval/__init__.py: correctly ABSENT

Commits exist:
- 19edc7b (Task 1 — __init__.py skeleton): FOUND
- 9567467 (Task 2 — generate_trajectory): FOUND
- a4d8889 (Task 3 — apply_rigid/apply_affine): FOUND

Verification checks:
- `from eval.generators import generate_trajectory, apply_rigid, apply_affine` exits 0: PASS
- `eval.generators.__all__ == {'generate_trajectory', 'apply_rigid', 'apply_affine'}`: PASS
- generate_trajectory determinism, shape, validation: PASS
- apply_rigid identity + immutability: PASS
- apply_affine identity (t=zeros) + immutability: PASS
