# Phase 14: Synthetic Data Generators - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Create synthetic data generators in `eval/generators/` at the repo root (not inside `src/zreg/`). The package delivers two things: a from-scratch trajectory factory that produces `dict[int, zRegPointCloud]` with Gaussian-blob point clouds, and a set of corruption wrappers (rigid/affine transforms, Gaussian noise, outlier injection, label generation/removal) that accept an existing `dict[int, zRegPointCloud]` and return a transformed copy. All generators accept `seed: int | None = 42`. Full unit tests in `tests/test_generators.py`.

Runners (`eval/run_synthetic.py` etc.) are out of scope — those are Phase 16 (EVAL-05).

</domain>

<decisions>
## Implementation Decisions

### Generator Input Model
- **D-01:** Both a from-scratch factory AND immutable corruption wrappers. The factory creates new trajectories; the wrappers transform existing ones and return a new dict (never mutate input).
- **D-02:** From-scratch factory (`generate_trajectory`) returns `dict[int, zRegPointCloud]` directly — a multi-frame trajectory already assembled. Each frame is a Gaussian blob point cloud (`N(0, I)` distribution, shape `(n_points, 3)`). The factory does NOT return a single point cloud for callers to assemble.
- **D-03:** All corruption wrappers (`apply_rigid`, `apply_affine`, `add_gaussian_noise`, `add_outliers`) are immutable: they accept `dict[int, zRegPointCloud]` and return a new `dict[int, zRegPointCloud]` with the transformation applied. The input dict is never modified.

### Module Structure
- **D-04:** `eval/generators/` is organized by category across four files:
  - `generators.py` — `generate_trajectory(n_points, n_frames, seed=42)` (from-scratch factory)
  - `transforms.py` — `apply_rigid()`, `apply_affine()` (transformation wrappers)
  - `corruption.py` — `add_gaussian_noise()`, `add_outliers()` (corruption wrappers)
  - `labels.py` — `generate_labels()`, `remove_labels()` (label utilities)
- **D-05:** `eval/generators/` is an importable Python package. `__init__.py` re-exports all public functions so runners can use `from eval.generators import generate_trajectory, add_gaussian_noise`.

### Label Representation
- **D-06:** Synthetic labels are stored in `zRegPointCloud['color']` — the same field used by the real data pipeline and `color_transfer.py`. This ensures generated data is directly compatible with evaluation metrics.
- **D-07:** Label format: `torch.tensor`, `dtype=torch.long`, shape `(N,)`. One integer class ID per point. Compatible with what `zreg.metrics.compute_f1` consumes (sklearn F1 expects hard integer labels).
- **D-08:** `remove_labels()` sets `zRegPointCloud['color'] = None` — matches the default `zRegPointCloud.__init__` behavior where unset fields are `None`. Callers check `if pc['color'] is None`.

### Test Strategy
- **D-09:** Tests in `tests/test_generators.py`, following the existing one-test-file-per-module convention. Test classes: `TestGenerateTrajectory`, `TestTransformWrappers`, `TestCorruptionWrappers`, `TestLabelUtilities`.
- **D-10:** `tests/conftest.py` gets a `sys.path.insert(0, repo_root)` addition so `from eval.generators import ...` works project-wide without per-file boilerplate.

### Claude's Discretion
- **Label assignment strategy** — Voronoi-based (k random seed points, assign each point to nearest seed) or random uniform (each point draws independently). Planner picks whichever is simpler to implement correctly with the seed contract.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §EVAL-03 — Full spec: generator types, seed contract (`seed: int | None = 42`), output type (`dict[int, zRegPointCloud]`), corruption types (Gaussian noise, outlier injection)

### Existing Infrastructure
- `src/zreg/dataset.py` — `zRegPointCloud` class definition (dict subclass with fields: `pos`, `color`, `id`, `fps-idx`); understand the constructor defaults before populating synthetic instances
- `src/zreg/color_transfer.py` — reference for how `color` field is used for celltype labels (integer IDs) in the real pipeline; synthetic generators must match this convention
- `src/zreg/transforms/__init__.py` — `RigidTransformation`, `AffineTransformation` classes; use as parameter types in `apply_rigid` / `apply_affine` wrappers (see how temporal_stability in Phase 13 built 4×4 matrices from their attributes)
- `src/zreg/metrics/label_transfer.py` — `compute_f1` signature; labels from `generate_labels()` must be directly passable here without conversion

### Testing
- `tests/conftest.py` — CUDA skip pattern (`@pytest.mark.skipif(not torch.cuda.is_available(), ...)`), fixture conventions; needs `sys.path.insert` addition for eval imports

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `zRegPointCloud({"pos": tensor, "color": tensor, ...})` — construct synthetic point clouds directly; `pos` field is `(N, 3)` float tensor, `color` field will be `(N,)` long tensor for labels
- `RigidTransformation`, `AffineTransformation` from `src/zreg/transforms/` — use as typed input parameters for `apply_rigid` / `apply_affine`; access `.rot`, `.t`, `.scale` / `.b`, `.t` attributes to build the 4×4 transform matrix (established in Phase 13 D-02)
- `torch.cdist` — already used in the codebase for pairwise distance computation; can be reused in Voronoi-based label assignment if planner chooses that approach

### Established Patterns
- All zreg public functions accept `torch.Tensor` inputs and validate shapes at entry — follow the same validation pattern in generator functions
- `seed: int | None = 42` — set `torch.manual_seed(seed)` (and `numpy.random.seed(seed)` if numpy is used) at the start of any stochastic function; `None` means caller controls the RNG state
- `HAS_OPEN3D` + lazy import pattern from `src/zreg/dataset.py` — NOT needed here; generators use pure torch/numpy, no Open3D dependency
- `__all__` list in `__init__.py` — existing pattern in all zreg subpackages; `eval/generators/__init__.py` should define `__all__`

### Integration Points
- `eval/generators/` will be consumed by Phase 16 runners (`eval/run_synthetic.py`, `eval/run_real.py`) via `from eval.generators import ...` — imports must work after adding repo root to sys.path
- Output `dict[int, zRegPointCloud]` from generators feeds directly into `zreg.metrics.*` functions (Phase 13 outputs) — ensure `pos` field dtype and device are consistent
- `tests/conftest.py` — adding `sys.path.insert(0, repo_root)` here is the single change needed to make eval imports available to all tests

</code_context>

<specifics>
## Specific Ideas

- From-scratch factory uses `N(0, I)` Gaussian distribution for point positions; `n_frames` frames each independently sampled (or optionally with per-frame perturbation — planner decides)
- Corruption wrappers produce a deep copy of the input dict's point clouds before applying transforms, so the source trajectory is never modified
- `generate_labels(trajectory, n_classes, seed=42)` and `remove_labels(trajectory)` operate on a full `dict[int, zRegPointCloud]` (frame-level, not point-level) — each frame gets its labels generated or removed

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 14-Synthetic Data Generators*
*Context gathered: 2026-05-18*
