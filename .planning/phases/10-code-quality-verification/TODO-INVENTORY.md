# TODO Inventory

**Generated:** 2026-04-29
**Purpose:** Inventory of all TODO comments in zReg codebase (Phase 10 audit)
**Status:** Documentation only - these are NOT being implemented in v1.1

## Summary

| Category | Count |
|----------|-------|
| Feature additions | 2 |
| Performance optimizations | 0 |
| Refactoring | 2 |
| Documentation | 0 |
| Deferred code (TODO(deferred)) | 2 |
| **Total** | **6** |

## Detailed Inventory

### Feature Additions

| File | Line | TODO |
|------|------|------|
| src/zreg/distances/general.py | 76 | Add option to pass min/max to normalization function within minkowski_distance |
| src/zreg/pairwise_distance_matrix.py | 622 | Add partial for kwargs in farthest downsampling |

### Performance Optimizations

No performance-related TODOs found.

### Refactoring

| File | Line | TODO |
|------|------|------|
| src/zreg/utils.py | 307 | Fix normalize to use the points dicts not just the torch dicts |
| src/zreg/downsampling.py | 456 | Check out how to make voxel_down_sample work in the future |

### Documentation

No documentation-related TODOs found.

### Deferred Code (TODO(deferred))

| File | Line | TODO |
|------|------|------|
| src/zreg/transforms.py | 47 | dq3d import check preserved for DeformableKinematicModel |
| src/zreg/transforms.py | 470 | DeformableKinematicModel requires dq3d (dual quaternion) library |

## Categorization Details

### Feature Additions

1. **minkowski_distance min/max passthrough** (general.py:76)
   - Would allow reusing precomputed normalization bounds
   - Avoids redundant min/max computation when caller has bounds

2. **farthest downsampling kwargs** (pairwise_distance_matrix.py:622)
   - Would allow passing additional parameters to farthest_point_down_sample
   - Currently uses fixed partial with points=None

### Refactoring

1. **normalize_to_pc_w_most_points dict support** (utils.py:307)
   - Current implementation operates on raw tensors
   - Would need to support full zRegPointCloud dict structure

2. **voxel_down_sample implementation** (downsampling.py:456)
   - Entire function is commented out
   - Would need to investigate Open3D voxel_down_sample API

### Deferred Code

1. **DeformableKinematicModel** (transforms.py:47, 470)
   - Requires dq3d (dual quaternion) external library
   - Would need torch tensor adaptation from numpy
   - Preserved for potential future filterreg integration

## Notes

- These TODOs are tracked for future phases (v1.2+)
- None of these are being implemented in Phase 10
- See REQUIREMENTS.md "Future Requirements" section for related items
- The TODO at utils.py:265 is a docstring reference to another TODO, not an actionable item
