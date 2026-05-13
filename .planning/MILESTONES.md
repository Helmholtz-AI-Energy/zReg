# Milestones

## v1.0 Consolidation (Shipped: 2026-04-09)

**Phases completed:** 5 phases, 13 plans, 18 tasks

**Key accomplishments:**

- _validate_tensors() with NaN/inf and device-mismatch detection wired into 6 entry points (CPD, BaseWD, minkowski_distance, pairwise_distance_matrix) with 13 tests covering unit and integration scenarios
- ZREG_LOG_LEVEL env var and zreg.set_log_level() added; Python 3.8 conditional importlib block and dead importlib-metadata dependency removed
- Fixed MaxSWD gradient flow via in-place projection update and GSWD default degree propagation through pairwise_distance_matrix
- RigidCPD scale computation guarded against division by zero with torch.clamp, formula verified against Myronenko & Song 2010, and 3 regression tests added
- Removed redundant normalize_point_cloud from rbf_kernel; normalization now cached once per set_source in NonRigidCPD/ConstrainedNonRigidCPD
- One-liner:
- Extended MstepResult with 5-field namedtuple (n_iters, sigma2_history) and added torch.clamp sigma2 safety after each M-step with one-time WARNING log
- w-coordinate clamping with torch.finfo eps in transform_points_homogeneous, plus det/cond post-composition validation raising ValueError in RigidTransformation.__mul__
- Task 1 — ASWD tempfile auto-cleanup (`sw_varients.py`):
- One-liner:
- CPD numerical stability tests for extreme/degenerate inputs, 3-chain transform composition invariant tests, and CPU/GPU device handling tests
- DTW regression tests covering manhattan/cpd/minkowski metrics, windowed asymmetric trajectories, and single-timepoint boundary paths
- Empty-source guard in transfer_colors plus 8 edge case tests covering empty, single-point, and dimension-mismatched inputs

---

## v1.1 Code Quality & Refactoring (Shipped: 2026-05-13)

**Phases completed:** 7 phases (6–11.1, including 1 inserted), 14 plans
**Timeline:** 2026-04-13 → 2026-05-13 (30 days)
**Stats:** 70 commits, 85 files changed, +11,953 / −1,890 lines, ~6,195 LOC Python

**Key accomplishments:**

- Full Python 3.12 migration — PEP 604/673/585 applied across all modules; typing imports minimized; pyproject.toml updated to python_requires >= 3.12
- CPD deep restructure — abstract `BaseCPD` with M/E-step interface; Rigid, Affine, NonRigid, ConstrainedNonRigid variants; RBF kernel utility; 10-export public API
- DTW deep restructure — `DynamicTimeWarping`, `DTWResult`, `compose_constraints` as structured package; `_backtrace` private; variadic AND-semantics constraint composition
- Distance & Transform restructure — explicit `__all__` in both modules; consistent interfaces; comprehensive numpy-style docstrings; `DistanceMetric` Protocol; `TransformBase` exported
- Silent failure fixes + logging consistency — all `print` → `log.debug`; lazy import guards; swallowed exceptions surfaced; mutable defaults fixed in CPD package
- 391 passing tests at 94% coverage — dead code removed from downsampling; FPS/return_o3d paths fully exercised; zero regressions from any refactoring
- DTW-02 closed (Phase 11.1) — `DistanceMetric` Protocol promoted to public API from `zreg.distances`; callable pass-through added to `_sanitize_pairwise_distance_matrix`

**Requirements:** 29/29 satisfied (100%)

---
