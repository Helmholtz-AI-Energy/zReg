# Phase 9: Distance & Transform Restructure - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Restructure the distance metrics and transform modules for consistent interfaces, shared patterns, and explicit public APIs. Add a Callable Protocol for metric interface consistency (DTW-02). No new algorithms or features — purely structural refactoring and documentation improvements.

</domain>

<decisions>
## Implementation Decisions

### Distance Module Structure
- **D-01:** Keep flat 3-file structure in `distances/` — no further splitting
- **D-02:** Replace star imports in `distances/__init__.py` with explicit `__all__` list
- **D-03:** Document BaseWD role in docstrings; don't restructure the class hierarchy
- **D-04:** Keep projection helper functions as module-level in `sw_varients.py`

### Transform Module Structure
- **D-05:** Keep `transforms.py` as single file (~336 lines is manageable)
- **D-06:** Keep `transform_points_homogeneous()` as standalone function in transforms.py
- **D-07:** W-clamping pattern stays inline where used; document but don't extract
- **D-08:** Homogeneous coordinate handling consolidated in transforms.py (not utils.py)

### TPS/RBF Kernel Pattern
- **D-09:** Document shared kernel pattern in docstrings; no code extraction
- **D-10:** Both NonRigidTransformation (rbf_kernel) and TPSTransformation (tps_kernel) continue using utils.py kernel functions
- **D-11:** Note the parallel pattern but don't create a KernelTransformation base class

### Metric Interface (DTW-02)
- **D-12:** Define `DistanceMetric` Protocol with `__call__(x, y, **kwargs) -> Tensor` signature
- **D-13:** Protocol covers both functional metrics (euclidean_distance) and callable classes (SlicedWassersteinDistance)
- **D-14:** Add Protocol to distances module but do NOT export publicly — internal typing only
- **D-15:** Document the interface contract in pairwise_distance_matrix.py dispatch logic

### Public API — Distance Module
- **D-16:** Export all 9 current symbols: SlicedWassersteinDistance, MaxSlicedWassersteinDistance, ProjectedWassersteinDistance, AdaptiveSlicedWassersteinDistance, OrthogonalSlicedWassersteinDistance, GeneralisedSlicedWassersteinDistance, euclidean_distance, manhattan_distance, minkowski_distance
- **D-17:** Explicit `__all__` in `distances/__init__.py` replacing star imports

### Public API — Transform Module
- **D-18:** Export TransformBase (for user subclassing) plus all 5 concrete classes
- **D-19:** Public exports: TransformBase, RigidTransformation, AffineTransformation, NonRigidTransformation, CombinedTransformation, TPSTransformation, transform_points_homogeneous
- **D-20:** Add explicit `__all__` to transforms.py (already exists but verify completeness)

### Claude's Discretion
- Order of refactoring operations within each module
- Exact docstring wording for shared patterns
- Whether to add type aliases for common signatures

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Current Implementation
- `src/zreg/distances/__init__.py` — Current star-import aggregator
- `src/zreg/distances/general.py` — euclidean/manhattan/minkowski (90 lines)
- `src/zreg/distances/sw_varients.py` — 6 SWD variants as nn.Module (~500 lines)
- `src/zreg/transforms.py` — TransformBase + 5 subclasses + homogeneous function (~336 lines)
- `src/zreg/pairwise_distance_matrix.py` — String dispatch for metrics, uses distance functions

### Requirements
- `.planning/REQUIREMENTS.md` — DIST-01 through DIST-04, XFORM-01 through XFORM-04, DTW-02

### Patterns from Prior Phases
- `.planning/phases/06-python-3-12-migration/06-CONTEXT.md` — Type hint patterns (already modernized)
- `.planning/phases/07-cpd-deep-restructure/07-CONTEXT.md` — Package split pattern, `__all__` conventions
- `.planning/phases/08-dtw-deep-restructure/08-CONTEXT.md` — Minimal package structure, DTW-02 deferral

</canonical_refs>

<code_context>
## Existing Code Insights

### Distance Module Current Structure
- `distances/__init__.py` — 3-line star-import aggregator
- `distances/general.py` — 3 functions, all delegate to minkowski_distance (uses torch.cdist)
- `distances/sw_varients.py` — BaseWD(nn.Module) + 6 SWD variant classes
- Projection utilities are module-level functions (minibatch_rand_projections, etc.)
- All SWD classes inherit from BaseWD which handles batch dimension logic

### Transform Module Current Structure
- `TransformBase` — minimal base with `transform()` dispatch to `_transform()`
- 5 concrete classes: Rigid, Affine, NonRigid, Combined, TPS
- `transform_points_homogeneous()` — standalone function with w-clamping
- NonRigidTransformation uses `utils.rbf_kernel()`
- TPSTransformation uses `utils.tps_kernel()`

### Integration Points
- `pairwise_distance_matrix.py` — uses `_sanitize_pairwise_distance_matrix()` for string dispatch to distance functions
- DTW delegates metric computation to `create_pairwise_distance_matrix()`
- Tests in `tests/test_distances.py` and `tests/test_transforms.py`

### Reusable Assets
- Kernel functions already in `utils.py` — rbf_kernel, tps_kernel, squared_kernel
- Input validation via `_validate_tensors()` already used in distance functions
- Type hints already modernized in Phase 6

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard documentation and interface cleanup patterns apply. Follow the "document, don't restructure" principle established in prior phases for manageable-size modules.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-distance-transform-restructure*
*Context gathered: 2026-04-27*
