# Phase 13: Core Metrics Library - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Create the installable metrics library at `src/zreg/eval/metrics/` with production-quality implementations of all alignment metrics (chamfer, hausdorff, path smoothness, kNN consistency, temporal stability) and the label transfer metric (F1), plus full unit tests. The `src/zreg/metrics/` proto stubs are left untouched. The `eval/` repo-root runner scripts are out of scope (phase 16).

</domain>

<decisions>
## Implementation Decisions

### Temporal Stability
- **D-01:** Temporal stability measures transformation smoothness across consecutive frames. Input: `list[RigidTransformation | AffineTransformation]`. Output: mean Frobenius norm of the matrix difference between consecutive transformations. Formula: `mean(||T_i.to_matrix() - T_{i-1}.to_matrix()||_F for i in 1..N)`.
- **D-02:** Use the existing `zreg.transforms` homogeneous matrix representation. Call `.to_matrix()` (or equivalent) on each transform to get a 4×4 tensor, then compute Frobenius norm on GPU if available.

### Migration Strategy
- **D-03:** Do NOT delete `src/zreg/metrics/`. Create fresh, correct implementations in `src/zreg/eval/metrics/alignment.py` and `src/zreg/eval/metrics/label_transfer.py`. The proto stubs remain as-is (orphaned, but not removed in this phase).

### eval Package API
- **D-04:** `src/zreg/eval/__init__.py` is minimal — does not re-export anything. Users import directly from submodules: `from zreg.eval.metrics.alignment import chamfer`. The `eval/` repo-root scripts (phase 16) are the primary consumers.
- **D-05:** `src/zreg/eval/` is NOT added to `src/zreg/__init__.py`. Keep the core `zreg` namespace clean (evaluation utilities are separate from the registration API).

### Test Expectations
- **D-06:** Phase 13 includes full unit tests in `tests/test_eval_metrics.py` covering each metric function: known-distance inputs with verified expected outputs, edge cases (single point, identical clouds, empty labels), sentinel masking for F1, and tensor shape/dtype validation.
- **D-07:** GPU tests use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`, following the existing test convention in `tests/conftest.py`. CPU-path correctness is always tested.

### Claude's Discretion
- Package init file for `src/zreg/eval/metrics/__init__.py`: can re-export the key functions at the subpackage level (e.g., `from .alignment import chamfer, hausdorff, path_smoothness, knn_consistency, temporal_stability`) for ergonomic imports — final call to planner.
- Whether `temporal_stability` should raise `ValueError` for a single-element list or return `0.0` — planner decides based on downstream HPO usage.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §EVAL-01 — Full spec for alignment metrics: function signatures, parameter defaults, implementation constraints (torch.cdist, torch.quantile, no NumPy on GPU, kNN detach/cpu entry point)
- `.planning/REQUIREMENTS.md` §EVAL-02 — Full spec for label transfer F1: sentinel masking, `average` parameter, `zero_division=0`

### Existing Proto Stubs (reference for interface compatibility, not correctness)
- `src/zreg/metrics/alignment_metrics.py` — Pseudocode proto stubs; note bugs: uses undefined `mean`/`min_distance`, numpy percentile without GPU guard, dtw_smoothness uses plain slopes not path indices
- `src/zreg/metrics/label_transfer_metrics.py` — Proto stubs; bugs: `average="macro"` hardcoded (should be param defaulting to `"weighted"`)

### Existing Infrastructure to Reuse
- `src/zreg/transforms/__init__.py` — `RigidTransformation`, `AffineTransformation` (used by temporal_stability input type)
- `src/zreg/transforms/homogeneous.py` — Matrix representation pattern for 4×4 tensors
- `tests/conftest.py` — CUDA skip pattern and fixture conventions to follow in test_eval_metrics.py

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `zreg.transforms.RigidTransformation` / `AffineTransformation`: Accept as `temporal_stability` input type — call `.T` or equivalent to get 4×4 homogeneous matrix tensor.
- `torch.cdist`: Already used in existing zreg code — use for chamfer/hausdorff pairwise distances.
- `pytest.mark.skipif(not torch.cuda.is_available(), ...)`: Existing CUDA skip pattern in tests/conftest.py — replicate exactly.

### Established Patterns
- All zreg functions accept `torch.Tensor` inputs and validate shapes at entry. Follow the same validation pattern.
- `from sklearn.neighbors import KDTree` with explicit `.detach().cpu().numpy()` before calling sklearn — existing pattern in the codebase (downsampling uses cKDTree from scipy).
- `from zreg.dataset import zRegPointCloud` — do NOT import zRegPointCloud in the eval metrics functions; accept raw tensors only so metrics are usable without the full dataset infrastructure.

### Integration Points
- `src/zreg/eval/` is a new top-level subpackage of `zreg`. It needs `__init__.py` at `src/zreg/eval/` and `src/zreg/eval/metrics/`.
- The `eval/` repo-root scripts (phase 16) will `from zreg.eval.metrics.alignment import chamfer` etc. — these import paths must work after `pip install -e .`.

</code_context>

<specifics>
## Specific Ideas

- Temporal stability: `mean(||T_i.to_matrix() - T_{i-1}.to_matrix()||_F)` over consecutive frame pairs — Frobenius norm on the 4×4 homogeneous matrix difference.
- Chamfer: `squared: bool = False` parameter — when True, square the per-point distances before averaging (useful for backprop). Use `torch.cdist`, take per-row min, then mean (and optionally square).
- F1: `average` parameter defaults to `"weighted"` (for HPO) but accepts `"macro"` (for reporting) — this is the key bug fix from the proto stub which hardcoded `"macro"`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 13-Core Metrics Library*
*Context gathered: 2026-05-15*
