---
phase: 27-datafactory-geometric-augmentation-methods
reviewed: 2026-06-11T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - eval/data_factory.py
  - tests/test_data_factory.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-06-11
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Phase 27 adds five geometric augmentation methods to `DataFactory` (`scale`, `rotate`, `drop_points`, `sample_new_points`) and extends the `augment()` dispatcher with four new keys. The core augmentation logic is broadly correct, but two critical defects were found: `get_ground_truth` silently dispatches to the wrong loader for unknown `data_format` values (bypassing the `ValueError` guard present in `load_real`), and `sample_new_points` raises an unguarded `RuntimeError` when `n_extra < 0`. Four warnings address non-reproducible train/val splits, silent boundary violations in `drop_points`, a partial-immutability asymmetry in `scale`, and an inner-loop closure redefinition. Three informational items cover missing caching, silent swallowing of unknown augmentation keys, and a misleading test assertion.

---

## Critical Issues

### CR-01: `get_ground_truth` uses a bare `else` — silently dispatches to CSV loader for unknown `data_format`

**File:** `eval/data_factory.py:304-309`
**Issue:** `load_real()` (lines 103–113) raises `ValueError` for any `data_format` that is not `"tracklets"` or `"csv"`. `get_ground_truth()` (lines 305–308) uses a bare `else` that silently calls `load_shah_from_csv` for any `data_format` value that is not `"tracklets"`, including completely invalid values. A user who configures `data_format: "mat"` will get a clean error from `load_real` but a cryptic CSV-parse failure from `get_ground_truth`. The inconsistency also means the two code paths can diverge in future when new formats are added.

**Fix:**
```python
# Replace lines 304-309 with:
if self.config.ground_truth_path is not None:
    if self.config.data_format == "tracklets":
        gt_ds, _ = load_data_from_tracklets(self.config.ground_truth_path, device="cpu")
    elif self.config.data_format == "csv":
        gt_ds = load_shah_from_csv(self.config.ground_truth_path, device="cpu")
    else:
        raise ValueError(
            f"DataFactory: unknown data_format {self.config.data_format!r}; "
            f"expected 'tracklets' or 'csv'"
        )
    return {i: pc["id"] for i, pc in gt_ds.items()}
return {i: pc["id"] for i, pc in dataset.items()}
```

---

### CR-02: `sample_new_points` raises unguarded `RuntimeError` for negative `n_extra`

**File:** `eval/data_factory.py:464`
**Issue:** `torch.rand(n_extra, 3, ...)` raises `RuntimeError: Trying to create tensor with negative dimension` when `n_extra < 0`. There is no input validation guard. By contrast, the analogous `add_outliers` in `src/zreg/generators/corruption.py` (line 103) explicitly raises `ValueError` for `n_outliers < 0`. When `n_extra` comes from `augmentation_params["n_new_points"]` (line 226 in `augment()`), a mis-configured YAML produces an opaque PyTorch error rather than an actionable `ValueError`.

**Fix:**
```python
def sample_new_points(
    self,
    dataset: dict[int, zRegPointCloud],
    n_extra: int,
    seed: int = 42,
) -> dict[int, zRegPointCloud]:
    if n_extra < 0:
        raise ValueError(f"n_extra must be >= 0, got {n_extra}")
    ...
```

---

## Warnings

### WR-01: `prepare_split` uses unseeded `random.sample` — train/val splits are not reproducible

**File:** `eval/data_factory.py:267`
**Issue:** `random.sample(keys, k=val_count)` draws from the Python `random` module without setting a seed. Every call to `prepare_split` on the same dataset will produce a different split unless the caller explicitly seeds `random` externally. All other stochastic operations in `DataFactory` (`drop_points`, `sample_new_points`, `add_gaussian_noise`, `add_outliers`, `generate_trajectory`) accept an explicit `seed` parameter and call `torch.manual_seed` or pass `seed=42`. The omission makes cross-run experiment comparisons unreliable and is not documented in the docstring.

**Fix:**
```python
def prepare_split(
    self,
    dataset: dict[int, zRegPointCloud],
    seed: int = 42,
) -> tuple[dict[int, zRegPointCloud], dict[int, zRegPointCloud]]:
    ...
    rng = random.Random(seed)
    val_keys = set(rng.sample(keys, k=val_count))
    ...
```
Using `random.Random(seed)` instead of the global `random` state avoids affecting other callers.

---

### WR-02: `drop_points` silently clamps `fraction=1.0` to keep 1 point; no validation for `fraction` outside `[0, 1]`

**File:** `eval/data_factory.py:413`
**Issue:** `keep = max(1, round(n * (1.0 - fraction)))` silently keeps 1 point when `fraction=1.0` (instead of 0) and keeps `n + round(n * abs(fraction))` points when `fraction < 0`. Neither case raises an error or warning. A user who writes `dropout_fraction: 1.0` expecting all points to be removed will instead get a 1-point frame with no diagnostic. By contrast, `add_outliers` validates `n_outliers >= 0` explicitly.

**Fix:**
```python
if not (0.0 <= fraction <= 1.0):
    raise ValueError(f"fraction must be in [0, 1], got {fraction}")
```
Add this guard at the start of `drop_points`, before `torch.manual_seed(seed)`.

---

### WR-03: `scale()` shares `color`, `id`, and `fps-idx` by reference — partially non-immutable

**File:** `eval/data_factory.py:339-344`
**Issue:** `scale()` constructs new `zRegPointCloud` objects using `pos=pc["pos"] * factor` (a new tensor) but passes `color=pc["color"]` and `id=pc["id"]` by reference. Line 344 then does `result[i]["fps-idx"] = pc["fps-idx"]`, also a shared reference. If any downstream consumer modifies `result[i]["color"]` or `result[i]["id"]`, they silently mutate the original dataset. This is asymmetric with `rotate()`, which delegates to `apply_rigid` → `copy.deepcopy` and is fully immutable. The docstring mentions this behaviour for `pos` but does not call out that `id` and `color` are shared. The composed augmentation chain `scale → drop_points` is safe (since `drop_points` creates new indexed tensors), but other compositions may not be.

**Fix:**
```python
result[i] = zRegPointCloud(
    pos=pc["pos"] * factor,
    color=pc["color"].clone() if pc["color"] is not None else None,
    id=pc["id"].clone() if pc["id"] is not None else None,
)
fps = pc["fps-idx"]
result[i]["fps-idx"] = fps.clone() if fps is not None else None
```

---

### WR-04: `_extend` closure is redefined on every loop iteration in `sample_new_points`

**File:** `eval/data_factory.py:468-474`
**Issue:** The nested function `_extend` is defined inside the `for i, pc in dataset.items()` loop body (lines 468–474). Python creates a new function object on every iteration. `_extend` captures only `n_extra` from the enclosing scope, which is a fixed parameter and never changes. There is no functional bug, but creating a closure per-frame wastes allocations proportional to the dataset size and obscures that the helper is loop-invariant.

**Fix:** Move the `_extend` definition to before the loop:
```python
def _extend(t, fill=-1):
    if t is None:
        return None
    if t.dim() == 1:
        return torch.cat([t, torch.full((n_extra,), fill, dtype=t.dtype, device=t.device)])
    else:
        return torch.cat([t, torch.zeros((n_extra, t.shape[1]), dtype=t.dtype, device=t.device)])

result: dict[int, zRegPointCloud] = {}
for i, pc in dataset.items():
    ...
```

---

## Info

### IN-01: `get_ground_truth` does not cache the external GT dataset — re-reads from disk on every call

**File:** `eval/data_factory.py:304-309`
**Issue:** When `config.ground_truth_path` is set, every call to `get_ground_truth` loads the file from disk again. The `_real_dataset` and `_synthetic_dataset` caches implement D-09 for the main data paths, but there is no equivalent `_gt_dataset` cache. In evaluation loops that call `get_ground_truth` per trial, this can produce many redundant disk reads. This is not a correctness issue, but it is inconsistent with the D-09 lazy-caching pattern the class is built around.

**Fix:** Add a `_gt_dataset` cache attribute in `__init__` and populate it on first call.

---

### IN-02: Unknown keys in `augmentation_params` are silently ignored

**File:** `eval/data_factory.py:190-227`
**Issue:** `augment()` dispatches on a fixed set of known keys (`"sigma"`, `"n_outliers"`, `"scale_factor"`, `"rotation_deg"`, `"dropout_fraction"`, `"n_new_points"`). Any other key (e.g. a misspelled `"dropout_fracion"`) produces a silent no-op. No warning is logged. A user debugging why their dropout augmentation has no effect would get no diagnostic. The behaviour is consistent with how Python dict-based dispatch typically works, but a log warning would improve usability.

**Fix:**
```python
KNOWN_KEYS = {"sigma", "n_outliers", "scale_factor", "rotation_deg",
              "dropout_fraction", "n_new_points", "scale", "rotation_axis"}
unknown = set(params) - KNOWN_KEYS
if unknown:
    import warnings
    warnings.warn(f"DataFactory.augment: unknown augmentation_params keys: {unknown}")
```

---

### IN-03: `test_scale_then_dropout_chain` (line 648) uses a misleading assertion

**File:** `tests/test_data_factory.py:648`
**Issue:** After `scale(2.0)` followed by `drop_points(0.5)`, the test asserts `not torch.allclose(out[i]["pos"], ds[i]["pos"][:50])`. The `[:50]` slice of the original dataset is the first 50 rows by index position, whereas `out[i]["pos"]` is a random 50-row subset selected by `torch.randperm`. These are different point subsets, so the comparison does not validate that the scale was applied — it just checks that two arbitrary sets of points differ, which will almost always be true regardless of correctness. The test passes even if `scale()` is a no-op, as long as `drop_points` selects different rows than `[:50]`.

**Fix:** To verify scale was applied before dropout, clone the positions before augmentation and compare against the scaled versions after recovering which indices were kept:
```python
# Simpler alternative: check that magnitudes of out positions are ~2x larger than original
for i in ds:
    assert out[i]["pos"].shape[0] == 50
    # All kept points should have been scaled by 2.0
    assert out[i]["pos"].norm(dim=1).mean() > ds[i]["pos"].norm(dim=1).mean() * 1.5
```

---

_Reviewed: 2026-06-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
