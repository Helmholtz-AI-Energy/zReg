---
phase: 16-runner-scripts
reviewed: 2026-05-18T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - eval/run_real.py
  - eval/run_synthetic.py
  - tests/test_runners.py
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 16: Code Review Report

**Reviewed:** 2026-05-18T00:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the two sweep orchestrators (`eval/run_real.py`, `eval/run_synthetic.py`) and their
functional test suite (`tests/test_runners.py`). Both scripts execute without crashing in the
happy path, but each contains a distinct logic error that causes incorrect data to be persisted
to disk — one a silent wrong-value bug, the other a silent dropped-metric bug. The test suite
has a hard-coded relative path that makes one test CWD-dependent, and three additional quality
issues reduce the overall trustworthiness of the sweep.

---

## Critical Issues

### CR-01: `scale` loop variable is never applied — all 9 logged cells record the same (unscaled) point counts

**File:** `eval/run_real.py:45-57`

**Issue:** `run_sweep` iterates `for scale in SCALES` but `scale` is used only inside the
`run_id` string label. The `n_points_base` count and the dataset trajectory are never
subsampled by `scale`. As a result, every row logged to disk for a given `density` fraction
carries identical `n_points_before` / `n_points_after` values regardless of the scale cell,
making the outer sweep loop produce duplicated data under mismatched labels. The module
docstring says "The sweep iterates over scale and density fractions" — but scale is a label,
not an operation.

```python
# Current — scale is computed but discarded
for scale in SCALES:
    for density in DENSITY_FRACTIONS:
        run_id = f"real_scale{scale}_density{density}_seed{SEED}"
        n_after = int(n_points_base * density)   # scale not applied anywhere
        log_run(
            ...
            n_points_before=n_points_base,
            n_points_after=n_after,
            ...
        )
```

**Fix:** Either (a) apply scale to derive the base count before the density fraction, or (b)
remove `SCALES` from the sweep if only density is meaningful for this dataset:

```python
# Option A: apply scale to derive n_points_scaled, then density fraction of that
for scale in SCALES:
    n_points_scaled = int(n_points_base * scale)          # actual scale application
    for density in DENSITY_FRACTIONS:
        run_id = f"real_scale{scale}_density{density}_seed{SEED}"
        n_after = int(n_points_scaled * density)
        log_run(
            ...
            n_points_before=n_points_scaled,
            n_points_after=n_after,
            ...
        )
```

---

### CR-02: Computed `chamfer` and `hausdorff` metrics are printed but never persisted — logged records are permanently incomplete

**File:** `eval/run_synthetic.py:56-73`

**Issue:** Each sweep cell computes `cd` (Chamfer distance) and `hd` (Hausdorff distance) at
lines 56-61, then prints them at line 73. However, `log_run()` is called at lines 64-72
without passing `cd` or `hd`. The module docstring explicitly states the script "comput[es]
chamfer/hausdorff against the clean reference, and persisting every run", but the metrics are
only printed to stdout and silently dropped from the persisted JSON/CSV records. Any downstream
analysis that reads the logged files will find no metric columns.

```python
# Current — cd and hd are computed and printed but not logged
cd = chamfer(...).item()
hd = hausdorff(...).item()
log_run(
    run_id=run_id,
    dataset_path="synthetic",
    ...
    # cd and hd are absent here
)
print(f"{run_id}: chamfer={cd:.4f}, hausdorff={hd:.4f}")
```

**Fix:** `log_run()` does not currently accept metric keyword arguments, so either extend
`log_run` to pass through extra fields, or log them via a separate mechanism. The minimum
viable fix that preserves the current `log_run` contract is to write a companion sidecar:

```python
# Option A: extend log_run signature (preferred — keeps records self-contained)
# In eval/tracking/tracking.py, add **extra_fields parameter and merge into record.

# Option B: emit a sidecar metrics JSON per cell until log_run is extended
import json
metrics_path = Path(output_dir) / f"{run_id}_metrics.json"
metrics_path.write_text(json.dumps({"chamfer": cd, "hausdorff": hd}, indent=2))
```

---

## Warnings

### WR-01: Hard-coded relative path in `test_script_has_main_guard` causes CWD-dependent test failure

**File:** `tests/test_runners.py:203`

**Issue:** The test opens `Path("eval/run_real.py")` using a bare relative path. When pytest
is invoked from any directory other than the repository root (e.g. `pytest tests/` from within
`tests/`), `Path("eval/run_real.py")` resolves relative to the process CWD, not relative to
the test file, and raises `FileNotFoundError`. Every other file-inspection test in the suite
uses `inspect.getfile()` which is CWD-independent.

```python
# Line 203 — fragile
source_text = Path("eval/run_real.py").read_text()
```

**Fix:** Use `inspect.getfile()` to mirror `TestRunSynthetic.test_metrics_import_path`:

```python
import eval.run_real as rr
source_text = Path(inspect.getfile(rr)).read_text()
assert 'if __name__ == "__main__"' in source_text
```

---

### WR-02: `generate_trajectory` is called inside the inner loop, regenerating identical clean data on every cell

**File:** `eval/run_synthetic.py:51`

**Issue:** `generate_trajectory(N_POINTS, N_FRAMES, seed=SEED)` is called at the top of the
inner loop on every `(sigma, n_out)` combination. Because `seed=SEED` is constant, the call
always produces the same result. Moving it outside the loops would be both semantically
correct (one fixed clean reference for the entire sweep) and avoids repeated allocation of
identical tensors (12 redundant calls for the default 4×3 grid). Additionally, passing the
same `seed=SEED` to `add_gaussian_noise` and `add_outliers` inside the same iteration means
every cell that shares the same `n_out > 0` produces the same outlier pattern regardless of
`sigma`, which undermines the independence of the sweep dimensions.

```python
# Current — clean trajectory regenerated 12 times with identical result
for sigma in SIGMAS:
    for n_out in N_OUTLIERS_LIST:
        traj_clean = generate_trajectory(N_POINTS, N_FRAMES, seed=SEED)
```

**Fix:** Hoist `traj_clean` outside both loops:

```python
traj_clean = generate_trajectory(N_POINTS, N_FRAMES, seed=SEED)
for sigma in SIGMAS:
    for n_out in N_OUTLIERS_LIST:
        traj_noisy = add_gaussian_noise(traj_clean, sigma=sigma, seed=SEED)
        traj_noisy = add_outliers(traj_noisy, n_outliers=n_out, seed=SEED)
        ...
```

---

### WR-03: `DATASET_PATH` is a bare relative string resolved at call time, not pinned to the repo root

**File:** `eval/run_real.py:23,38`

**Issue:** `DATASET_PATH = "data/raw/example.mat"` is resolved by `Path(DATASET_PATH).exists()`
at runtime using the process CWD. The script injects `_repo_root` into `sys.path` at line
15-17 for import resolution, but never uses `_repo_root` to anchor the dataset path. When the
script is invoked from any directory other than the repo root, `Path(DATASET_PATH).exists()`
returns `False` even if the file exists, causing a silent skip instead of an informative
"wrong directory" error.

```python
# Line 23
DATASET_PATH = "data/raw/example.mat"   # resolved relative to CWD, not repo root
```

**Fix:** Pin `DATASET_PATH` to the already-computed `_repo_root`:

```python
DATASET_PATH = str(_repo_root / "data" / "raw" / "example.mat")
```

---

## Info

### IN-01: No `__main__`-guard test for `eval/run_synthetic.py`

**File:** `tests/test_runners.py`

**Issue:** `TestRunReal` includes `test_script_has_main_guard` to confirm the real-data script
is directly executable, but `TestRunSynthetic` has no analogous test. Both scripts are
described as standalone in their module docstrings and both carry `if __name__ == "__main__"`
blocks, so the asymmetric coverage is a gap.

**Fix:** Add a mirror test to `TestRunSynthetic`:

```python
def test_script_has_main_guard(self):
    """eval/run_synthetic.py is executable as `python eval/run_synthetic.py`."""
    import eval.run_synthetic as rs
    source_text = Path(inspect.getfile(rs)).read_text()
    assert 'if __name__ == "__main__"' in source_text
```

---

### IN-02: `frame_idx = 0` magic number — only frame 0 is used for metric computation

**File:** `eval/run_synthetic.py:55`

**Issue:** `frame_idx = 0` is a magic literal that selects the first frame for Chamfer/Hausdorff
computation. The intent (representative single-frame evaluation) is not documented; if
`N_FRAMES` were changed to 1 a reader would not know whether 0 is deliberate or a default,
and if outliers were applied per-frame the choice of frame 0 vs. an average matters.

**Fix:** Replace with a named constant and a brief comment:

```python
EVAL_FRAME = 0  # module-level constant: representative frame for per-cell metric snapshots
...
cd = chamfer(traj_clean[EVAL_FRAME]["pos"], traj_noisy[EVAL_FRAME]["pos"]).item()
hd = hausdorff(traj_clean[EVAL_FRAME]["pos"], traj_noisy[EVAL_FRAME]["pos"]).item()
```

---

_Reviewed: 2026-05-18T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
