# Phase 9: Distance & Transform Restructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-27
**Phase:** 09-distance-transform-restructure
**Areas discussed:** Distance module structure, Transform module structure, Metric interface, Public API design

---

## Distance Module Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Keep flat (Recommended) | Keep current 3-file structure. Clean up BaseWD, add __all__ to __init__.py, document shared projection logic. Minimal disruption. | ✓ |
| Split SWD variants | Move each SWD variant to its own file, keep a base.py with BaseWD and shared projection functions. More files, clearer separation. | |
| You decide | Claude has flexibility to choose based on code clarity during implementation | |

**User's choice:** Keep flat (Recommended)
**Notes:** User preferred minimal disruption approach for a manageable-size module.

### BaseWD Cleanup

| Option | Description | Selected |
|--------|-------------|----------|
| Document, don't restructure (Recommended) | Add clear docstrings to BaseWD explaining its role. Shared projection utilities are already module-level functions. Keep nn.Module pattern as-is. | ✓ |
| Extract projection utilities | Move projection helper functions (minibatch_rand_projections, etc.) into a dedicated _projections section with clear grouping. Keep BaseWD minimal. | |
| You decide | Claude assesses during implementation which approach yields cleaner code | |

**User's choice:** Document, don't restructure (Recommended)
**Notes:** Follows the established "document, don't restructure" pattern from prior phases.

---

## Transform Module Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Keep single file (Recommended) | 336 lines is manageable. Clean up TransformBase interface, consolidate homogeneous handling, add proper __all__. Similar to keeping distances/ flat. | ✓ |
| Split into package | Create transforms/ package with base.py, rigid.py, nonrigid.py, tps.py, utils.py. More files, follows CPD/DTW pattern exactly. | |
| You decide | Claude assesses during implementation based on coupling and readability | |

**User's choice:** Keep single file (Recommended)
**Notes:** Consistent with distances/ decision — manageable modules stay as single files.

### Homogeneous Coordinate Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Keep in transforms.py (Recommended) | transform_points_homogeneous stays as standalone function. W-clamping pattern documented but kept inline where used. No utils.py extraction needed. | ✓ |
| Extract to utils.py | Move homogeneous coordinate utilities (w-clamping, point conversion) to utils.py for reuse across modules. | |
| You decide | Claude determines during implementation based on actual reuse patterns | |

**User's choice:** Keep in transforms.py (Recommended)
**Notes:** Consolidation within transforms.py rather than spreading to utils.py.

### TPS/RBF Kernel Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Document shared pattern (Recommended) | Both already use utils.py kernel functions. Add docstrings noting the parallel pattern. No code extraction needed. | ✓ |
| Create kernel base class | Extract a KernelTransformation base that both NonRigid and TPS inherit from, with shared kernel setup logic. | |
| You decide | Claude assesses whether code similarity warrants abstraction | |

**User's choice:** Document shared pattern (Recommended)
**Notes:** XFORM-03 satisfied through documentation, not code restructuring.

---

## Metric Interface (DTW-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Document dispatch pattern (Recommended) | Keep string dispatch in _sanitize_pairwise_distance_matrix(). Document the interface contract: (x: Tensor, y: Tensor, **kwargs) -> Tensor. Add type hints and docstrings. | |
| Protocol/ABC for metrics | Define a DistanceMetric Protocol that all metric functions/classes must satisfy. Adds type safety but more ceremony. | ✓ |
| You decide | Claude determines during implementation what level of formalization makes sense | |

**User's choice:** Protocol/ABC for metrics
**Notes:** User opted for more formal type safety approach.

### Protocol Design

| Option | Description | Selected |
|--------|-------------|----------|
| Callable Protocol (Recommended) | Define as Protocol with __call__(x, y, **kwargs) -> Tensor. Works for both functions and callable classes. Simple and Pythonic. | ✓ |
| ABC with compute method | Define ABC with abstract compute(x, y) method. Requires wrapping functions. More explicit but more boilerplate. | |
| You decide | Claude chooses based on what fits the existing code patterns best | |

**User's choice:** Callable Protocol (Recommended)
**Notes:** Callable Protocol works for both functional metrics and nn.Module classes.

---

## Public API Design

### Distance Module API

| Option | Description | Selected |
|--------|-------------|----------|
| All current exports (Recommended) | Export all 9 current symbols: 6 SWD classes + euclidean/manhattan/minkowski. Add explicit __all__ to __init__.py instead of star imports. | ✓ |
| Add Protocol export | Same as above, plus export the new DistanceMetric Protocol for users who want to implement custom metrics. | |
| You decide | Claude determines what public API makes most sense during implementation | |

**User's choice:** All current exports (Recommended)
**Notes:** Protocol kept internal; public API stays stable.

### Transform Module API

| Option | Description | Selected |
|--------|-------------|----------|
| Export base + concrete (Recommended) | Export TransformBase, Rigid, Affine, NonRigid, Combined, TPS, and transform_points_homogeneous. Users may want to subclass TransformBase. | ✓ |
| Concrete only | Export only the 5 concrete classes + transform_points_homogeneous. Keep TransformBase as internal implementation detail. | |
| You decide | Claude determines based on whether TransformBase has useful public methods | |

**User's choice:** Export base + concrete (Recommended)
**Notes:** TransformBase exported for user subclassing.

---

## Claude's Discretion

- Order of refactoring operations within each module
- Exact docstring wording for shared patterns
- Whether to add type aliases for common signatures

## Deferred Ideas

None — discussion stayed within phase scope
