---
phase: 17-framework-config-datafactory
reviewed: 2026-05-27T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - eval/config.py
  - eval/data_factory.py
  - tests/test_data_factory.py
  - setup.cfg
findings:
  critical: 3
  warning: 4
  info: 3
  total: 10
status: issues_found
---

# Phase 17: Code Review Report

**Reviewed:** 2026-05-27
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

This phase delivers `EvalConfig` (a pydantic v2 `BaseModel` loaded from YAML) and `DataFactory` (a lazy, cached orchestrator for real and synthetic point cloud trajectories). The test file covers construction, dispatch, caching, augmentation, splitting, and ground-truth extraction. The `setup.cfg` configures the package build.

The overall structure is sound and the lazy-caching pattern is correctly implemented. However, three issues rise to BLOCKER severity: the `eval` package is not installed (making every import in the test suite fail in any fresh environment), `prepare_split` is not reproducible (uses an unseeded `random.sample`), and `EvalConfig` accepts arbitrary negative / out-of-range values for several constrained fields without validation. Four additional warnings cover silent error swallowing, an untested code path, a missing test, and an import side-effect risk.

---

## Critical Issues

### CR-01: `eval` package not discoverable — all tests fail in a fresh install

**File:** `setup.cfg:40-41`

**Issue:** `setup.cfg` declares `package_dir = =src` and `packages = find_namespace:` scoped to `src/`. The `eval/` directory lives at the repository root, not inside `src/`. Therefore `eval.config` and `eval.data_factory` are **never installed** when the project is installed via `pip install -e .` or `pip install .`. Every `from eval.config import …` and `from eval.data_factory import …` in the test suite and in any downstream script will raise `ModuleNotFoundError` in any environment that does not have the repo root on `sys.path` by accident.

`conftest.py` happens to add the repo root to `sys.path` (line 16), which masks the problem for the test runner when tests are invoked from the project root. This is a fragile workaround, not a fix. Any import of `eval.*` outside the test runner (e.g. a CLI, a Jupyter notebook, or another package) will break silently.

**Fix:** Either move `eval/` into `src/` so it is discovered alongside `zreg`, or add an explicit entry in `setup.cfg`:

```ini
[options]
package_dir =
    = src
    eval = eval
```

Or, more cleanly, move the package:

```
src/
  zreg/
  eval/
    __init__.py   ← must be added too (see CR-02)
    config.py
    data_factory.py
```

---

### CR-02: `eval/` directory has no `__init__.py` — package is not a proper Python package

**File:** `eval/config.py:1` (directory level)

**Issue:** The `eval/` directory contains `config.py` and `data_factory.py` but no `__init__.py`. Without it, `eval` is not a regular package; it only works as a namespace package under `find_namespace:` discovery. Combined with CR-01 (the directory is outside `src/`), the package is doubly broken. Even if the path issue from CR-01 were fixed by adding the repo root to `PYTHONPATH`, namespace-package semantics differ from regular-package semantics in ways that can cause subtle import failures (e.g., `from eval import config` vs. `import eval.config` may behave differently across Python versions and import-system configurations).

**Fix:** Add `eval/__init__.py` (may be empty):

```python
# eval/__init__.py
```

---

### CR-03: `prepare_split` is non-reproducible — no seed for `random.sample`

**File:** `eval/data_factory.py:229`

**Issue:** `prepare_split` uses `random.sample` from Python's `random` module with no seed. The `generate_synthetic` and `augment` methods both use `seed=42` explicitly for reproducibility. `prepare_split` does not, which means:

1. Two calls to `prepare_split` on the same dataset in the same process return different train/val partitions.
2. Experiment results are not reproducible across runs unless the caller pins the global random state beforehand.
3. The test `test_split_returns_two_dicts_with_disjoint_keys` passes by coincidence (it only checks sizes and disjointness, never the specific keys selected).

The docstring and notes section make no mention of this non-determinism, which means downstream consumers will be surprised when re-running experiments produces different train/val splits.

**Fix:** Either accept a `seed` parameter (consistent with the rest of the codebase) or use a seeded `random.Random` instance:

```python
def prepare_split(
    self,
    dataset: dict[int, zRegPointCloud],
    seed: int | None = 42,
) -> tuple[dict[int, zRegPointCloud], dict[int, zRegPointCloud]]:
    ...
    rng = random.Random(seed)
    val_keys = set(rng.sample(keys, k=val_count))
```

---

## Warnings

### WR-01: No field validators for constrained numeric fields — invalid configs pass silently

**File:** `eval/config.py:99-103`

**Issue:** Several fields have documented semantic constraints that are not enforced by pydantic validators:

- `val_split`: must be in `[0.0, 1.0)`. A value of `1.5` or `-0.3` is accepted without error, causing `prepare_split` to call `random.sample(keys, k=n*1.5)` which raises a `ValueError` with a confusing message far from the config declaration, or to return an empty training set.
- `n_synthetic`: must be `>= 1` (enforced downstream by `generate_trajectory` which raises `ValueError` for `n_frames < 1` — but the error message will not mention the config field name).
- `n_trials`: must be `>= 1`.
- `transform_degree`: documented semantics imply `>= 0`.
- `search_strategy`: documented as one of `"grid"`, `"random"`, `"bayesian"` — any string is accepted.
- `tier`: documented as one of `"sanity"`, `"dev"`, `"full"` — any string is accepted.
- `data_format`: documented as one of `"tracklets"`, `"csv"` — any string is accepted (the error only surfaces at `load_real()` call time).

**Fix:** Add pydantic `Field` constraints or `@field_validator` for at least the numeric range fields:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator
import re

val_split: float = Field(default=0.2, ge=0.0, lt=1.0)
n_synthetic: int = Field(default=100, ge=1)
n_trials: int = Field(default=10, ge=1)
transform_degree: float = Field(default=0.1, ge=0.0)
data_format: str = Field(default="tracklets", pattern=r"^(tracklets|csv)$")
search_strategy: str = Field(default="grid", pattern=r"^(grid|random|bayesian)$")
tier: str = Field(default="sanity", pattern=r"^(sanity|dev|full)$")
```

---

### WR-02: `get_ground_truth` swallows the external-GT load error silently when `data_format` is `"csv"` but the path is missing

**File:** `eval/data_factory.py:266-272`

**Issue:** When `config.ground_truth_path` is set and `config.data_format` is not `"tracklets"`, the code falls through to `load_shah_from_csv`. There is no `else: raise ValueError(...)` guard equivalent to the one in `load_real` (line 108). If `data_format` is some future/unknown value (which WR-01 allows through), `load_shah_from_csv` is called unconditionally — the `elif`/`else` pattern in `load_real` is not mirrored here. This is a logic inconsistency that could mask bugs.

More concretely: in `load_real` an unknown `data_format` raises `ValueError` with a clear message. In `get_ground_truth` the same invalid format silently calls the CSV loader. The two methods are inconsistent in their handling of the same field.

**Fix:** Mirror the guard from `load_real`:

```python
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
```

---

### WR-03: `prepare_split` returns the original `dataset` reference (not a copy) in the degenerate path

**File:** `eval/data_factory.py:228`

**Issue:** When `n <= 1 or val_count == 0`, the method returns `(dataset, {})` — the **same dict object** that was passed in, not a copy. The class docstring and method docstring both state "do not mutate in place", but the degenerate return hands the caller the exact same mutable reference with no safeguard. If a caller receives the train dict and mutates it (which the docstring warns against but cannot prevent), it will also mutate the original dataset passed into `prepare_split`.

The non-degenerate path builds a new dict via dict comprehension (`{k: dataset[k] for k in train_keys}`), so the train dict there is a fresh object (though its values are still references to the same `zRegPointCloud` objects). The degenerate path is therefore inconsistent: it returns the original reference, not even a shallow copy.

**Fix:**

```python
if n <= 1 or val_count == 0:
    return dict(dataset), {}   # shallow copy for consistency
```

---

### WR-04: `from pathlib import Path  # noqa: F401` — intentionally unused import with a misleading comment

**File:** `eval/data_factory.py:16`

**Issue:** `Path` is imported and immediately suppressed with `noqa: F401` and a comment `(available for future use)`. Keeping dead imports in production code — even with a suppression comment — is a maintenance liability: it appears in autocomplete, misleads readers into thinking it is used, and the `noqa` suppression hides future accidental introduction of real unused imports. The comment "available for future use" is not a valid justification for committing an unused import.

**Fix:** Remove the import. Add it back when a concrete use is introduced:

```python
# Remove this line:
from pathlib import Path  # noqa: F401  (available for future use)
```

---

## Info

### IN-01: `test_empty_yaml_handled` tests `EvalConfigError` but not the actual `TypeError` prevention

**File:** `tests/test_data_factory.py:87-93`

**Issue:** The test comment explains that `yaml.safe_load("")` returns `None` and that without `or {}` the code would raise `TypeError`. The test only asserts that `EvalConfigError` is raised — it does not verify that a `TypeError` does **not** escape. This is fine as written (since `EvalConfigError` is not a `TypeError`), but the test comment is slightly misleading: it implies the guard `or {}` converts `None → {}`, but the actual failure mode without the guard would be `TypeError: argument of type 'NoneType' is not iterable` leaking out unwrapped. A comment clarifying this would improve maintainability.

**Fix:** Add an assertion or a comment clarification:

```python
# The `or {}` guard converts yaml.safe_load("") == None to {}
# Without it, cls(**None) raises TypeError, not ValidationError — it would escape the except block
with pytest.raises(EvalConfigError, match="data_path"):
    EvalConfig.from_yaml(p)
# Confirm it is not a raw TypeError
```

---

### IN-02: No test for `prepare_split` when `val_count` truncates to zero (4-frame dataset)

**File:** `tests/test_data_factory.py:251-285`

**Issue:** The docstring at `data_factory.py:215-219` explicitly documents the `int(4 * 0.2) = 0` silent-empty-val edge case as a known pitfall. The test suite tests a 1-frame dataset (D-07 guard) and a 10-frame dataset, but does not test the 4-frame / default-val_split scenario that the implementation explicitly calls out. This edge case — where `val_count == 0` due to floor truncation — silently returns an empty val set even though the caller passed a non-trivial val_split. It is a footgun worth covering.

**Fix:** Add a test:

```python
def test_zero_val_count_returns_empty_val(self):
    """4 frames * 0.2 val_split = int(0.8) = 0 → val is silently empty (Pitfall 8)."""
    ds = self._make_ds(4)
    cfg = EvalConfig(data_path="x", val_split=0.2)
    factory = DataFactory(cfg)
    train, val = factory.prepare_split(ds)
    assert val == {}
    assert len(train) == 4
```

---

### IN-03: `setup.cfg` coverage target is `src/zreg` only — `eval/` package is excluded from coverage

**File:** `setup.cfg:113` and `pyproject.toml:48`

**Issue:** `addopts = --cov zreg --cov-report term-missing` and `[tool.coverage.run] source = ["src/zreg"]` mean that `eval/config.py` and `eval/data_factory.py` are never included in coverage reports. Any coverage gate (if introduced in the future) will ignore the `eval` package entirely. This is likely unintentional given that the test file is specifically named `test_data_factory.py`.

**Fix:** Extend coverage to include the `eval` package:

```ini
# setup.cfg
addopts =
    --cov zreg --cov eval --cov-report term-missing
    --verbose
```

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/zreg", "eval"]
```

---

_Reviewed: 2026-05-27_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
