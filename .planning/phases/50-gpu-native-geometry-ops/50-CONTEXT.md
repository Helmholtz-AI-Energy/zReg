# Phase 50: GPU-Native Geometry Ops for Cluster Deployment - Context

**Gathered:** 2026-07-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 50 delivers GPU-native replacements for the three Open3D-CPU geometry ops in
`src/zreg/models/_ops.py`, using `torch_cluster`'s CUDA kernels — conditional on
confirming `torch_cluster` installs on HoreKa Linux/CUDA.

**Two-plan structure:**
1. **Plan 1 — Install verification:** Confirm `pip install torch_cluster` + import succeeds
   on HoreKa Linux/CUDA. If it fails, write a documented finding and close the phase (no
   code changes). If it succeeds, continue to Plan 2.
2. **Plan 2 — Code changes:** Add `torch_cluster` dual-path to all three ops in `_ops.py`,
   following the existing pattern in `src/zreg/downsampling.py`.

**All three ops are in scope:**
- `farthest_point_sample` — PointNet++ FPS → `torch_cluster.fps`
- `ball_query` — PointNet++ radius query → `torch_cluster.radius`
- `build_radius_graph` — eGNN graph construction → `torch_cluster.radius_graph`

The ROADMAP's eGNN exclusion applies to the equivariant conv architecture (EGNNConv math
stays hand-rolled), NOT to `build_radius_graph`'s graph construction. At the actual runtime
scale (1k points after `max_points_per_frame` subsampling), `build_radius_graph` queries
all N points as centers and is the most expensive of the three ops — GPU acceleration here
is most impactful.

No changes to callers (`SetAbstractionLayer`, `EGNNConv`). Public function signatures in
`_ops.py` are preserved.

</domain>

<decisions>
## Implementation Decisions

### Phase Structure
- **D-01:** Two-plan structure: Plan 1 = HoreKa install check; Plan 2 = code changes.
  If Plan 1 fails (install cannot succeed on Linux/CUDA), document the finding and close
  the phase as a no-op — the Open3D fallback already works and is sufficient.
- **D-02:** Plan 1 is a human-executed step on the HoreKa login node (not CI/automated):
  `pip install torch_cluster -f https://data.pyg.org/whl/torch-X.X.X+cuXXX.html` then
  `python -c "from torch_cluster import fps, radius, radius_graph; print('OK')"`. Plan 1
  produces a one-line result: PASS or FAIL with the error message.

### Scope
- **D-03:** All three ops in `_ops.py` get `torch_cluster` dual paths. `build_radius_graph`
  is included despite eGNN being architecturally excluded — the geometry indexing layer is a
  platform concern, not an architecture concern, and is the most expensive op at 1k points.
- **D-04:** Public signatures of all three functions are PRESERVED. No changes to
  `SetAbstractionLayer` or `EGNNConv`. Interface adaptation happens entirely inside `_ops.py`.

### Fallback Pattern
- **D-05:** Follow `src/zreg/downsampling.py` exactly:
  - `try/except ImportError` at module level to import `fps`, `radius`, `radius_graph`
    from `torch_cluster`
  - Local `TORCH_CLUSTER_AVAILABLE = True/False` flag (duplicated in `_ops.py` — NOT
    extracted to a shared module; matches `downsampling.py`'s self-contained pattern)
  - Each function auto-selects `torch_cluster` when `TORCH_CLUSTER_AVAILABLE and pos.is_cuda`,
    falls back to Open3D otherwise
  - Dev machine (macOS ARM) continues to work via the Open3D path with no changes

### Interface Adaptation (inside _ops.py)
- **D-06:** `farthest_point_sample(pos, n_samples)`: convert `ratio = n_samples / pos.shape[0]`,
  call `torch_cluster.fps(pos, ratio=ratio)`, clip result to `[:n_samples]` to preserve the
  exact-count contract (torch_cluster may return `ceil(N * ratio)` which is occasionally
  `n_samples + 1`).
- **D-07:** `ball_query(pos, center_idx, radius, max_neighbors)`: call
  `torch_cluster.radius(pos[center_idx], pos, radius, max_num_neighbors=max_neighbors)`,
  group the returned `[2, E]` edge_index by target-center index to reconstruct
  `list[Tensor]`. Apply isolated-point guard: any center with no neighbors → `[center]`
  self-index (same fallback as the Open3D path, prevents empty-tensor max-pool crashes in
  `SetAbstractionLayer`).
- **D-08:** `build_radius_graph(pos, radius, max_neighbors)`: call
  `torch_cluster.radius_graph(pos, r=radius, max_num_neighbors=max_neighbors, loop=False,
  flow='source_to_target')`. Apply isolated-point guard: any node with no incoming edges
  gets a self-loop added — mirrors the Open3D path's existing fallback.

### Claude's Discretion
- Exact `torch_cluster` wheel URL / version to pin in the install command (use the
  `https://data.pyg.org/whl/torch-X.X.X+cuXXX.html` format with the HoreKa torch version)
- Whether to add `torch_cluster` as a new optional extra in `setup.cfg` (e.g., `cluster`
  extra alongside `mpi` and `propulate`) — reasonable to do for clarity
- Test strategy for the `torch_cluster` path: `# pragma: no cover` marks matching
  `downsampling.py` since the GPU path cannot run locally; integration validated on HoreKa
- `pytest.mark.skipif(not TORCH_CLUSTER_AVAILABLE, ...)` markers for any torch_cluster
  smoke tests committed alongside the code change

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Established Pattern to Follow
- `src/zreg/downsampling.py` — **MUST READ**: the project's existing `torch_cluster`
  conditional-import pattern (`TORCH_CLUSTER_AVAILABLE` flag, try/except, dual-path functions).
  _ops.py's implementation must match this pattern exactly.

### Code Being Modified
- `src/zreg/models/_ops.py` — **MUST READ**: current Open3D implementations of all three ops,
  isolated-point guards, docstrings, benchmarks, and device-agnostic output placement.
  Phase 50 adds torch_cluster paths alongside — does NOT remove Open3D paths.
- `src/zreg/models/pointnet2.py` — read to understand how `farthest_point_sample` and
  `ball_query` are consumed by `SetAbstractionLayer` (confirm callers stay unchanged).
- `src/zreg/models/egnn.py` — read to understand how `build_radius_graph` is consumed by
  `EGNNConv.forward()` (confirm caller stays unchanged).

### Prior Architecture Decisions
- `.planning/phases/45-egnn-pointnet-label-transfer-framework-selection-evaluation-/45-DESIGN.md`
  — MUST READ: why `torch_cluster` was originally rejected on dev machine (macOS ARM, no wheel,
  build failure reproduced live). Phase 50 revisits only for HoreKa Linux/CUDA. The NEVER-add
  rule in 45-DESIGN.md was dev-machine-scoped, not cluster-scoped.
- `.planning/phases/47-egnn-and-pointnet-model-implementation-training-infrastructu/47-01-PLAN.md`
  — D-03 benchmark context: Open3D ops were ≤50ms at 100–300 points. At 1k points
  (`max_points_per_frame: 1000`) they exceed this threshold, motivating GPU acceleration.

### Runtime Scale Context
- `baseline_experiments/configs/selfcal/kobitski_ew06_alignment.yaml` — `max_points_per_frame: 1000`
  is the actual point count at which _ops.py runs during evaluation. Phase 47's 100–300 pt
  benchmarks are training-data scale, not inference scale.

### Tests to Read Before Adding New Tests
- `tests/test_zreg_models_ops.py` — existing correctness + isolated-point guard + D-03
  benchmark tests. New torch_cluster tests extend this file; don't duplicate existing coverage.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/zreg/downsampling.py:30–37` — copy the `try/except ImportError` block verbatim as the
  template for `_ops.py`'s module-level import. Adapt the imported names (`fps`, `radius`,
  `radius_graph` instead of `fps`, `knn_graph`).
- `src/zreg/downsampling.py:75–110` — the `fps()` dual-path function is the structural template
  for `farthest_point_sample()`'s new torch_cluster path. Note the `pos.is_cuda` auto-detect
  logic and the `ImportError` raise when `use_torch_cluster=True` but not installed.

### Established Patterns
- **Isolated-point guard** — `if len(idx) == 0: idx = [i]` in `ball_query` and
  `build_radius_graph`. This guard MUST be preserved in the torch_cluster paths. The Phase 47
  RESEARCH.md documented why: `sample_bowl`'s sparser rim regions produce isolated points that
  cause empty-tensor max-pool crashes in `SetAbstractionLayer`.
- **Device-agnostic output** — all ops use `torch.as_tensor(..., device=pos.device)`. The
  torch_cluster path must respect this: `torch_cluster` ops return tensors already on
  `pos.device` (CUDA), so no explicit `.to(pos.device)` should be needed — but verify.
- **`# pragma: no cover`** — `downsampling.py` marks the torch_cluster execution branch with
  this pragma since it can't run locally. `_ops.py` torch_cluster paths should do the same.
- **Import order** — `zreg.dataset` (or any scipy-importing module) must be imported before
  `torch_cluster` to avoid the `OMP: Error #15` libomp SIGABRT (per 45-DESIGN.md import-order
  caveat). Tests must import `zreg.dataset` first.

### Integration Points
- `torch_cluster.fps(src, ratio)` → clip to `[:n_samples]` → return to `SetAbstractionLayer`
- `torch_cluster.radius(x, y, r)` → group by center → `list[Tensor]` → return to `SetAbstractionLayer`
- `torch_cluster.radius_graph(x, r, loop=False, flow='source_to_target')` → isolated-point guard → return `[2, E]` to `EGNNConv`

</code_context>

<specifics>
## Specific Ideas

- Plan 1 install command: `pip install torch_cluster -f https://data.pyg.org/whl/torch-X.X.X+cuXXX.html`
  where `X.X.X+cuXXX` matches the HoreKa PyTorch version. Verify with
  `python -c "from torch_cluster import fps, radius, radius_graph; print('OK')"`.
- If Plan 1 FAILS: write `50-01-SUMMARY.md` documenting the exact error (version mismatch,
  missing CUDA toolkit, etc.) and mark phase closed. Open3D ops remain unchanged.
- If Plan 1 PASSES: Plan 2 adds ~30 lines to `_ops.py` — three `if TORCH_CLUSTER_AVAILABLE
  and pos.is_cuda:` branches, one per function. No other files change except possibly `setup.cfg`
  for the new optional extra.
- The `n_samples → ratio` conversion for FPS: `ratio = n_samples / pos.shape[0]`. Clip
  result with `idx[:n_samples]` to guard against `ceil()` rounding returning one extra index.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 50-GPU-Native Geometry Ops for Cluster Deployment*
*Context gathered: 2026-07-15*
