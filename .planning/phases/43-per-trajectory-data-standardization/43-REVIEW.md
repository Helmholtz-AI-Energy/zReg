---
phase: 43-per-trajectory-data-standardization
reviewed: 2026-06-30T12:00:00Z
fixed: 2026-06-30T12:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - eval/config.py
  - eval/data_factory.py
  - tests/test_data_preprocessing_config.py
  - tests/test_data_factory.py
findings:
  critical: 1
  warning: 3
  info: 4
  total: 8
fixed:
  critical: 1
  warning: 3
open:
  info: 4
status: fixes_applied
fix_commit: 00bfd73
---

# Phase 43: Code Review Report

**Reviewed:** 2026-06-30T12:00:00Z
**Fixed:** 2026-06-30T12:00:00Z (commit `00bfd73`)
**Depth:** standard
**Files Reviewed:** 4
**Status:** fixes_applied — all Critical + Warning findings resolved; 4 Info findings remain open

## Summary

Phase 43 adds `DataPreprocessingConfig` (Pydantic v2 model) and wires per-trajectory scaling into `DataFactory.load_real()` / `load_target()`. The Pydantic model contracts are correct (`extra="forbid"`, `Literal` for method, eps placed in denominators). The three scaling formulas are mathematically sound for the common case. One crash path (CR-01), one silent NaN path (WR-01), a call-ordering hazard (WR-02), and a shared mutable default (WR-03) were identified and fixed in commit `00bfd73`. Test coverage gaps for these edge cases (IN-02, IN-03) remain open.

## Critical Issues

### CR-01: `_standardize` crashes with `RuntimeError` on empty dataset ✓ FIXED

**File:** `eval/data_factory.py:625`
**Issue:** When `dataset` is an empty dict (`{}`), `torch.cat([pc["pos"] for pc in dataset.values()], dim=0)` becomes `torch.cat([], dim=0)`, which raises:
```
RuntimeError: torch.cat: expected a non-empty list of Tensors
```
No guard exists before this call. An empty dataset can occur with a corrupt or mismatched `.mat` file, a `max_points_per_frame` that filters all points, or a caller supplying `{}` directly. The error message gives no hint that preprocessing is the culprit.

**Fix:**
```python
if stats is None:
    if not dataset:
        # Nothing to scale; leave stats uncached and return input unchanged.
        return dataset
    all_pos = torch.cat([pc["pos"] for pc in dataset.values()], dim=0)
    ...
```

## Warnings

### WR-01: NaN propagation when exactly one point exists across all frames ✓ FIXED

**File:** `eval/data_factory.py:627`
**Issue:** `all_pos.std(dim=0)` uses Bessel correction (`correction=1`, i.e., divides by N-1) by default. When the concatenated dataset contains exactly one point (e.g., one frame with a single-point cloud), `std(dim=0)` returns `tensor([nan, nan, nan])`. Because `NaN + 1e-8 = NaN`, the eps guard on line 652 does not prevent NaN propagation into the scaled `pos` tensor. All downstream stages — DTW, metrics, visualisation — silently receive NaN coordinates. The existing `test_zero_std_no_error` only covers 10 identical points (std = 0.0, not NaN) and misses this case.

**Fix:** Replace the uncorrected `std` with `correction=0` (population std) or add an explicit NaN guard:
```python
std = all_pos.std(dim=0, correction=1)   # Bessel-corrected — keep for semantic correctness
# Guard: std is NaN when N==1; treat as zero so eps kicks in
std = torch.where(torch.isnan(std), torch.zeros_like(std), std)
```
Alternatively, reject single-point datasets at the guard introduced for CR-01.

### WR-02: Paired-mode coordinate-space corruption when `load_target()` is called before `load_real()` ✓ FIXED

**File:** `eval/data_factory.py:122, 175`
**Issue:** `load_real()` always invokes `_standardize(dataset)` without passing `stats` (line 122). Inside `_standardize`, the `stats is None` branch always writes to `self._preprocessing_stats`. Consequently, if a caller invokes `load_target()` before `load_real()`:

1. `load_target()` → `_standardize(target, stats=None)` → computes TARGET statistics → stores them in `_preprocessing_stats`.
2. `load_real()` → `_standardize(source, stats=None)` → computes SOURCE statistics → **overwrites** `_preprocessing_stats`.

The target dataset, already cached and standardized with TARGET statistics, remains in the cache. The source is standardized with SOURCE statistics. Both datasets are now in different coordinate spaces, silently violating the paired-mode requirement. No exception or log message is emitted.

**Fix:** Document the required call order explicitly in both docstrings, and add an enforcement guard in `load_target()`:
```python
def load_target(self) -> dict[int, zRegPointCloud]:
    if self._target_dataset is not None:
        return self._target_dataset
    if (
        self.config.data_preprocessing is not None
        and self._preprocessing_stats is None
        and self._real_dataset is None
    ):
        import warnings
        warnings.warn(
            "DataFactory.load_target() called before load_real() in paired mode. "
            "Statistics will be computed from the TARGET dataset, not the source. "
            "Call load_real() first to share coordinate space.",
            stacklevel=2,
        )
    ...
```
A stronger fix would raise `RuntimeError` instead of warning, but that may be too strict if standalone target loading is a valid use case.

### WR-03: Shared mutable default instance for `data_preprocessing` field ✓ FIXED

**File:** `eval/config.py:280`
**Issue:** `data_preprocessing: DataPreprocessingConfig | None = DataPreprocessingConfig()` evaluates `DataPreprocessingConfig()` once at class-definition time, creating a single shared instance. In Pydantic v2, when a model-type field's default is already an instance of the declared type and `revalidate_instances` is not set to `'always'`, Pydantic v2 returns the existing instance rather than creating a copy. All `EvalConfig` instances constructed without an explicit `data_preprocessing` argument therefore reference the same `DataPreprocessingConfig` object. An in-place mutation — `cfg.data_preprocessing.method = "normalize"` — would silently affect every other default-using instance in the same process.

**Fix:**
```python
data_preprocessing: DataPreprocessingConfig | None = Field(
    default_factory=DataPreprocessingConfig
)
```

## Info

### IN-01: Dtype inconsistency in computed statistics dict

**File:** `eval/data_factory.py:626-633`
**Issue:** `mean`, `std`, `min_vals`, and `max_vals` are computed directly on `all_pos`, preserving its dtype. `median`, `q25`, and `q75` force float32 via `.float()` (required by `torch.quantile`). If the trajectory pos tensors are float64, the stats dict contains mixed dtypes: float64 for `mean`/`std`/`min`/`max`, float32 for `median`/`iqr`. PyTorch silently upcasts during arithmetic, so no computation error occurs in practice, but the inconsistency is error-prone for future maintenance.

**Fix:** Apply `.float()` consistently to all stats, or — better — cast `all_pos` to float32 once before the stats block:
```python
all_pos = torch.cat([pc["pos"] for pc in dataset.values()], dim=0).float()
```

### IN-02: No test for empty-dataset crash (CR-01 coverage gap)

**File:** `tests/test_data_factory.py`
**Issue:** `TestDataPreprocessing` has no test that passes `{}` to `_standardize`. The crash identified in CR-01 is undetected by the test suite. After CR-01 is fixed, a regression test should be added to confirm graceful handling.

**Fix:** Add:
```python
def test_standardize_empty_dataset_no_crash(self):
    factory = self._make_factory(method="standardize")
    result = factory._standardize({})
    assert result == {}
```

### IN-03: No test for single-point NaN propagation (WR-01 coverage gap)

**File:** `tests/test_data_factory.py`
**Issue:** The existing `test_zero_std_no_error` uses 10 identical points, producing `std = 0.0` (not NaN). The case of exactly 1 point total — where `std(dim=0, correction=1)` returns NaN — is untested. After WR-01 is fixed, a guard test should be added.

**Fix:** Add:
```python
def test_standardize_single_point_no_nan(self):
    factory = self._make_factory(method="standardize")
    pc = zRegPointCloud(
        pos=torch.tensor([[1.0, 2.0, 3.0]]),
        label=torch.zeros(1, dtype=torch.long),
        id=torch.zeros(1, dtype=torch.long),
    )
    pc["fps-idx"] = None
    result = factory._standardize({0: pc})
    assert torch.isfinite(result[0]["pos"]).all()
```

### IN-04: `test_load_target_without_prior_load_real_computes_own_stats` asserts too weakly

**File:** `tests/test_data_factory.py:1397`
**Issue:** The test only checks `factory._preprocessing_stats is not None`. It does not verify that the stored stats were derived from the mock target dataset (e.g., checking that `stats["mean"]` is close to the mock's known mean ≈ 0). The test would pass even if stats were populated from a stale cached call or initialised to a sentinel value.

**Fix:** Add a tighter assertion using the known mock dataset statistics:
```python
# mock_ds is drawn from N(0,1), mean per dim should be near 0
stats = factory._preprocessing_stats
assert stats["mean"].abs().max().item() < 2.0   # N(0,1) with 30 points
```

---

_Reviewed: 2026-06-30T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
