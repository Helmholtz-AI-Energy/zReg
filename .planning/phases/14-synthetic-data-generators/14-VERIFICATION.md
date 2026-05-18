---
phase: 14-synthetic-data-generators
verified: 2026-05-18T00:00:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 14: Synthetic Data Generators — Verification Report

**Phase Goal:** Deliver the `eval/generators/` package at the repo root with a from-scratch trajectory factory and immutable corruption wrappers (rigid/affine transforms, Gaussian noise, outlier injection, Voronoi label generation, label removal) — all seed-deterministic, immutable, and `dict[int, zRegPointCloud]`-shaped per EVAL-03.
**Verified:** 2026-05-18T00:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `eval/generators` package is importable as a top-level package from the repo root | VERIFIED | `import eval.generators as g` succeeds; `__all__` check passed (`__all__ OK` printed) |
| 2 | `generate_trajectory(n_points, n_frames, seed=42)` returns `dict[int, zRegPointCloud]` with n_frames entries, each with `pos` shape `(n_points, 3)` | VERIFIED | `test_returns_dict_with_correct_keys` and `test_pos_shape` both pass; keys `{0,1,2}` correct, shape `(n_points, 3)` confirmed |
| 3 | `generate_trajectory` is seed-deterministic for non-None seeds and does not call `torch.manual_seed` for `seed=None` | VERIFIED | `test_seed_reproducibility` passes (two calls with seed=42 produce `torch.equal` pos); `test_seed_none_is_nondeterministic` passes (seed=None produces different tensors) |
| 4 | `apply_rigid` and `apply_affine` return new dicts and never mutate input (D-03 immutability) | VERIFIED | `test_apply_rigid_does_not_mutate_input` and `test_apply_affine_does_not_mutate_input` both pass; `copy.deepcopy` present in `_apply_matrix` helper in `transforms.py` |
| 5 | Identity rigid transform leaves `pos` unchanged up to float tolerance | VERIFIED | `test_apply_rigid_identity_preserves_pos` passes with `atol=1e-5`; `RigidTransformation()` default is identity (rot=I, scale=1, t=0) |
| 6 | `add_gaussian_noise` returns new dict with additive noise; input never mutated; seed-deterministic; sigma=0.0 is a no-op | VERIFIED | All 5 `TestCorruptionWrappers` noise tests pass including sigma=0.0 identity, reproducibility, and immutability |
| 7 | `add_outliers` increases every frame's `pos.shape[0]` by exactly n_outliers; extends `id`/`color`/`fps-idx` with sentinel -1 | VERIFIED | `test_add_outliers_increases_pos_length`, `test_add_outliers_extends_id_with_sentinel`, `test_add_outliers_extends_2d_color` all pass |
| 8 | `generate_labels` populates `pc['color']` with `torch.long (N,)` in `[0, n_classes)`; input never mutated; seed-deterministic | VERIFIED | `test_generate_labels_dtype_and_shape`, `test_generate_labels_value_range`, `test_generate_labels_reproducible`, `test_generate_labels_does_not_mutate_input` all pass |
| 9 | `remove_labels` returns dict where `pc['color'] is None` for every frame; input never mutated | VERIFIED | `test_remove_labels_sets_color_none` and `test_remove_labels_does_not_mutate_input` both pass |
| 10 | `eval.generators.__all__` contains exactly the 7 required names | VERIFIED | Python check `set(g.__all__) == {'generate_trajectory','apply_rigid','apply_affine','add_gaussian_noise','add_outliers','generate_labels','remove_labels'}` passed; 4 `from .X import` lines confirmed in `__init__.py`; `eval/__init__.py` correctly absent |
| 11 | `pytest tests/test_generators.py -x` exits 0 with all 34 tests passing across all 4 locked test classes | VERIFIED | `34 passed in 1.64s`; all 4 classes (TestGenerateTrajectory:8, TestTransformWrappers:7, TestCorruptionWrappers:11, TestLabelUtilities:8) verified by name |

**Score:** 11/11 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `eval/generators/__init__.py` | Package init with `__all__` containing all 7 symbols | VERIFIED | Non-empty; 4 `from .X import` lines; `__all__` = 7-element list; no `eval/__init__.py` present |
| `eval/generators/generators.py` | `generate_trajectory` factory returning `dict[int, zRegPointCloud]` | VERIFIED | Non-empty; `__all__ = ["generate_trajectory"]`; `from zreg.dataset import zRegPointCloud`; `import torch`; no numpy; validation raises; seed contract correct |
| `eval/generators/transforms.py` | `apply_rigid` and `apply_affine` immutable wrappers | VERIFIED | Non-empty; `__all__ = ["apply_affine", "apply_rigid"]`; `copy.deepcopy` in `_apply_matrix`; `from zreg.transforms import AffineTransformation, RigidTransformation`; private helpers `_rigid_to_matrix`, `_affine_to_matrix`, `_apply_matrix` present; defensive w-divide retained |
| `eval/generators/corruption.py` | `add_gaussian_noise` and `add_outliers` immutable wrappers | VERIFIED | Non-empty; `__all__ = ["add_gaussian_noise", "add_outliers"]`; `copy.deepcopy` in both functions; seed contract correct; sentinel -1 for `id`/1-D color/`fps-idx`; zeros for 2-D color; no numpy |
| `eval/generators/labels.py` | `generate_labels` (Voronoi) and `remove_labels` immutable utilities | VERIFIED | Non-empty; `__all__ = ["generate_labels", "remove_labels"]`; `torch.cdist` used for Voronoi assignment; `dtype=torch.long`; `copy.deepcopy` in both functions; no numpy |
| `tests/test_generators.py` | 4 test classes covering all 7 public symbols | VERIFIED | Non-empty; 4 classes (confirmed by `grep -c "^class Test"`); 34 tests; imports all 7 from `eval.generators`; `compute_f1` imported and used in `TestLabelUtilities` |
| `tests/conftest.py` | sys.path.insert for repo root so `eval.generators` resolves | VERIFIED | `sys.path.insert`, `_repo_root = Path(__file__).parent.parent`, idempotent guard at lines 14-16; all 5 fixtures and 2 hooks preserved |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `eval/generators/__init__.py` | `eval/generators/generators.py` | `from .generators import generate_trajectory` | WIRED | Line 20 confirmed |
| `eval/generators/__init__.py` | `eval/generators/transforms.py` | `from .transforms import apply_rigid, apply_affine` | WIRED | Line 21 confirmed |
| `eval/generators/__init__.py` | `eval/generators/corruption.py` | `from .corruption import add_gaussian_noise, add_outliers` | WIRED | Line 22 confirmed |
| `eval/generators/__init__.py` | `eval/generators/labels.py` | `from .labels import generate_labels, remove_labels` | WIRED | Line 23 confirmed |
| `eval/generators/transforms.py` | `zreg.transforms.RigidTransformation/AffineTransformation` | `from zreg.transforms import AffineTransformation, RigidTransformation` | WIRED | Line 23 of transforms.py; used as type annotations and in matrix construction |
| `eval/generators/generators.py` | `zreg.dataset.zRegPointCloud` | `from zreg.dataset import zRegPointCloud` | WIRED | Line 15 of generators.py; `zRegPointCloud(pos=torch.randn(...))` in return |
| `eval/generators/labels.py` | `torch.cdist` (Voronoi assignment) | `torch.cdist(pos, seeds, p=2)` | WIRED | Line 76 of labels.py; `dtype=torch.long` on argmin output (line 77) |
| `tests/conftest.py` | `eval/generators/` (at repo root) | `sys.path.insert(0, str(_repo_root))` | WIRED | Lines 14-16 of conftest.py; idempotent guard present |
| `tests/test_generators.py` | `eval.generators` (all 7 public symbols) | `from eval.generators import generate_trajectory, apply_rigid, apply_affine, add_gaussian_noise, add_outliers, generate_labels, remove_labels` | WIRED | Lines 19-27 of test_generators.py |
| `tests/test_generators.py` (TestLabelUtilities) | `zreg.metrics.label_transfer.compute_f1` | `from zreg.metrics.label_transfer import compute_f1` | WIRED | Line 30; used in `test_generate_labels_compatible_with_compute_f1` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `generate_trajectory` | `pos` (shape `(n_points,3)`) | `torch.randn(n_points, 3)` in generator factory | Yes — live `torch.randn` call, not static | FLOWING |
| `apply_rigid` / `apply_affine` | transformed `pos` | 4x4 matrix multiply via `_apply_matrix`; deepcopy of input pos then `homo @ M_cast.T` | Yes — real matrix math applied | FLOWING |
| `add_gaussian_noise` | noisy `pos` | `torch.randn_like(pc["pos"]) * sigma` added to deepcopy pos | Yes — real Gaussian noise | FLOWING |
| `add_outliers` | extended `pos` | `torch.randn(n_outliers, 3)` concatenated to deepcopy pos | Yes — real randn outlier points | FLOWING |
| `generate_labels` | `pc["color"]` (torch.long) | Voronoi via `torch.cdist(pos, seeds)` argmin | Yes — real distance computation | FLOWING |
| `remove_labels` | `pc["color"]` set to None | `pc["color"] = None` (deterministic, no stochastic source needed) | Yes — correct behavior | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `__all__` contains exactly 7 required names | `python3 -c "import eval.generators as g; assert set(g.__all__) == {...}; print('__all__ OK')"` | `__all__ OK` | PASS |
| All contract checks (determinism, immutability, Voronoi labels) | Full contract check script | `All contract checks passed` | PASS |
| `eval/__init__.py` absent (D-05) | `test -f eval/__init__.py` | File absent | PASS |
| `pytest tests/test_generators.py -x -q` | 34 tests collected and run | `34 passed in 1.64s` | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EVAL-03 | 14-01, 14-02, 14-03 | Synthetic data generators in `eval/generators/` at repo root: rigid/affine/noise transforms, label removal, synthetic label generation — all accept `seed: int | None = 42`, produce `dict[int, zRegPointCloud]`, support Gaussian noise and outlier injection corruption types | SATISFIED | All 5 source files exist and are substantive; 34 tests pass covering every contract; `__all__` has exactly 7 symbols per spec; seed-determinism, immutability, and `dict[int, zRegPointCloud]` output shape all confirmed by behavioral checks |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | No debt markers (TBD/FIXME/XXX/TODO), no placeholder returns, no stubs detected | — | — |

No `return null`, `return {}`, or `return []` patterns found in any source file. No numpy or open3d imports in any generator or test file. No hardcoded empty data flows.

### Human Verification Required

None. All contracts verified programmatically:

- Determinism (seed) verified by `torch.equal` across two identically-seeded calls
- Immutability (D-03) verified by `copy.deepcopy` snapshot comparison in tests
- `dict[int, zRegPointCloud]` shape verified by type and key checks
- `torch.long` dtype, `(N,)` shape, and `[0, n_classes)` value range verified by test assertions
- `compute_f1` compatibility (D-07) verified by `test_generate_labels_compatible_with_compute_f1` producing 1.0 on perfect prediction
- No UI, visual, or external service behavior involved

### Gaps Summary

No gaps. All phase must-haves are fully satisfied.

---

**Notable implementation detail confirmed:** The plan's acceptance criteria contained a factual error — `AffineTransformation()` default has `t=[1,1,1]` (not identity). The implementation correctly uses `AffineTransformation(t=torch.zeros(3))` for identity affine in tests (documented in both Plan 01 and Plan 03 SUMMARYs). This is a plan authoring deviation, not an implementation defect; the behavior is correct.

---

_Verified: 2026-05-18T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
