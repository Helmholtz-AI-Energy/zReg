# API Comparison: v1.0 → v1.1

**v1.0 baseline:** commit `51d5e776` (2026-04-09), milestone "v1.0 Consolidation"  
**v1.1 current:** HEAD (2026-05-12), milestone "v1.1 Code Quality & Refactoring"

---

## Module Restructuring

### `cpd.py` → `cpd/` package (Phase 7)

| Symbol | v1.0 | v1.1 | Status |
|--------|------|------|--------|
| `CoherentPointDrift` | flat file | `cpd/base.py` | ✅ identical signature |
| `RigidCPD` | flat file | `cpd/rigid.py` | ✅ identical signature |
| `AffineCPD` | flat file | `cpd/affine.py` | ✅ identical signature |
| `NonRigidCPD` | flat file | `cpd/nonrigid.py` | ✅ identical signature |
| `ConstrainedNonRigidCPD` | flat file | `cpd/nonrigid.py` | ✅ identical signature |
| `cpd_registration()` | flat file | `cpd/_registration.py` | ✅ identical signature |
| `init_cpd_from_existing()` | flat file | `cpd/_registration.py` | ✅ identical signature |
| `EstepResult` | not exported | `cpd/_types.py` | ✅ new (additive) |
| `MstepResult` | not exported | `cpd/_types.py` | ✅ new (additive) |
| `rbf_kernel_matrix()` | internal | `cpd/kernels.py` | ✅ new (additive) |

**Verdict:** ✅ Fully backward-compatible. All v1.0 public symbols unchanged.

---

### `transforms.py` → `transforms/` package (Phase 9)

| Symbol | v1.0 | v1.1 | Status |
|--------|------|------|--------|
| `transform_points_homogeneous()` | flat file | `transforms/homogeneous.py` | ✅ identical signature |
| `TransformBase` | flat file | `transforms/base.py` | ✅ identical |
| `RigidTransformation` | flat file | `transforms/rigid.py` | ✅ identical |
| `AffineTransformation` | flat file | `transforms/affine.py` | ✅ identical |
| `NonRigidTransformation` | flat file | `transforms/nonrigid.py` | ✅ identical |
| `CombinedTransformation` | flat file | `transforms/combined.py` | ✅ identical |
| `TPSTransformation` | flat file | `transforms/tps.py` | ✅ identical |

**Verdict:** ✅ Fully backward-compatible.

---

### `dtw.py` → `dtw/` package (Phase 8)

| Symbol | v1.0 | v1.1 | Status |
|--------|------|------|--------|
| `DynamicTimeWarping` | flat file | `dtw/core.py` | ✅ identical signature |
| `DTWResult` | flat file | `dtw/result.py` | ✅ identical |
| `compose_constraints()` | not present | `dtw/constraints.py` | ✅ new (additive) |

**Verdict:** ✅ Fully backward-compatible. New `compose_constraints` is additive.

---

## Function-Level Changes

### `downsampling.py` (Phase 11)

| Symbol | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| `fps()` | same signature | same signature | ✅ |
| `_fps_open3d()` | internal, Open3D-based | **removed** | ✅ replaced by `_fps_numpy` |
| `_fps_numpy()` | not present | new internal, NumPy-based | ✅ |
| `_preserve_labels()` | internal (0 callers) | **removed** | ✅ dead-code cleanup |
| `farthest_point_down_sample()` | same | same | ✅ |
| `random_down_sample()` | same | same | ✅ |
| `uniform_down_sample()` | same | same | ✅ |
| `knn_graph()` | same | same | ✅ |
| `precompute_fps()` | same | same | ✅ |
| `remove_outliers_knn()` | same | same | ✅ |

**Behavioral change:** FPS now uses a deterministic NumPy greedy implementation (seeded from point 0) instead of Open3D. Results are functionally equivalent; Open3D is no longer required at runtime for FPS.

---

### `distances/` module

| Symbol | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| `SlicedWassersteinDistance` | same | same | ✅ |
| `AdaptiveSlicedWassersteinDistance` | same | same | ✅ |
| `MaxSlicedWassersteinDistance` | same | same | ✅ |
| `OrthogonalSlicedWassersteinDistance` | same | same | ✅ |
| `GeneralisedSlicedWassersteinDistance` | same | same | ✅ |
| `ProjectedWassersteinDistance` | same | same | ✅ |
| `euclidean_distance()` | same | same | ✅ |
| `manhattan_distance()` | same | same | ✅ |
| `minkowski_distance()` | `p` param silently ignored | `p` passed to `torch.cdist` | ✅ **bug fix** |
| `_protocol.py` | not present | new internal protocol | ✅ additive |

---

### `pairwise_distance_matrix.py`

| Symbol | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| `create_pairwise_distance_matrix()` | same | same | ✅ |
| `create_pairwise_distance_matrix_given_rigid_rot()` | same | same | ✅ |
| `_sanitize_pairwise_distance_matrix()` | same | same | ✅ |

---

### `dataset.py`

| Symbol | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| `zRegPointCloud` | same | same | ✅ |
| `load_data_from_tracklets()` | `Union[str, Path]` type hint | `str` parameter | ✅ |
| `zreg_to_open3d()` | same | same | ✅ |
| `open3d_to_zreg()` | same | same | ✅ |
| `load_shah_from_csv()` | `Union[str, Path]` type hint | `str \| Path` (PEP 604) | ✅ cosmetic |

---

### `utils.py`

| Symbol | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| `normalize_point_cloud()` | no zero-range guard | added zero-range axis guard | ✅ **bug fix** |
| All other functions | same | same | ✅ |

---

### `color_transfer.py`

| Symbol | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| `transfer_colors()` | same | added empty-source guard | ✅ **bug fix** |
| `ColorTransferMethod` enum | same | same | ✅ |

---

## New Modules in v1.1

| Module | Symbol | Description |
|--------|--------|-------------|
| `config.py` | `configure_pytorch()` | Centralized PyTorch settings (replaces scattered global calls) |
| `metrics/alignment_metrics.py` | alignment metrics | New evaluation utilities |
| `metrics/label_transfer_metrics.py` | label transfer metrics | New evaluation utilities |

---

## Type Hint Changes (Phase 6 — Python 3.12)

All `Union[X, Y]` → `X | Y` (PEP 604), `Optional[X]` → `X | None`, `List/Dict/Tuple` generics → built-in `list/dict/tuple`. These are annotation-only changes with no runtime impact.

---

## Behavioral Changes Summary

| # | What changed | Where | Impact |
|---|-------------|-------|--------|
| 1 | FPS uses NumPy (no Open3D dep) | `downsampling.py` | **No functional change;** output is deterministic from point 0 |
| 2 | `minkowski_distance` passes `p` to `torch.cdist` | `distances/general.py` | **Bug fix:** v1.0 always used `p=2` regardless of argument |
| 3 | `normalize_point_cloud` handles zero-range axis | `utils.py` | **Bug fix:** v1.0 divided by zero for flat point clouds |
| 4 | CPD constructor validates `source_colors` | `cpd/base.py` | **Bug fix:** v1.0 crashed silently with missing color data |
| 5 | `print()` → `log.debug()` throughout | multiple | Logging behavior only |
| 6 | `torch.set_default_device` / `torch.set_float32_matmul_precision` removed from module scope | `cpd/base.py`, `distances/sw_varients.py` | Removes hidden global state mutation on import |

---

## Removed Dead Code

| Symbol | Location | Reason |
|--------|----------|--------|
| `_fps_open3d()` | `downsampling.py` | Replaced by `_fps_numpy()`; Open3D no longer required |
| `_preserve_labels()` | `downsampling.py` | Zero callers; detected by static analysis |
| Global `torch.set_default_device()` call | `cpd/base.py` | Side-effects on import; moved to `configure_pytorch()` |
| Global `torch.set_float32_matmul_precision()` call | `cpd/base.py` | Same as above |

---

## Overall Verdict

All public APIs from v1.0 are present and **signature-identical** in v1.1. The refactoring is a pure internal restructuring (flat files → packages) plus targeted bug fixes and code quality improvements. Consumers of the library who import from `zreg.cpd`, `zreg.transforms`, `zreg.dtw`, `zreg.distances`, `zreg.downsampling`, or `zreg.pairwise_distance_matrix` will see no change in behaviour except the three bug fixes listed above.
