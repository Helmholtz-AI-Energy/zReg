---
phase: 28-script-integration-generate-datasets-uses-datafactory
reviewed: 2026-06-11T00:00:00Z
depth: standard
files_reviewed: 1
files_reviewed_list:
  - scripts/generate_datasets.py
findings:
  critical: 1
  warning: 3
  info: 1
  total: 5
status: issues_found
---

# Phase 28: Code Review Report

**Reviewed:** 2026-06-11
**Depth:** standard
**Files Reviewed:** 1
**Status:** issues_found

## Summary

`scripts/generate_datasets.py` integrates the Phase 28 requirement of using `DataFactory`
for semi-synthetic augmentation. The augmentation key mapping (`aug_type` → `EvalConfig`
param name) is correct and verified against `DataFactory.augment`'s dispatch table.
The bowl geometry scaling invariant is preserved across frames. Import order correctly
satisfies the libomp SIGABRT constraint.

One critical defect was found: the chunked-write + existence-check pattern silently
accepts a partially written file as complete on re-run. Three warnings cover an
inconsistent API surface in the geometry helpers, unconditional source-data loading,
and a hardcoded seed that produces identical noise/dropout patterns across all sources.

---

## Critical Issues

### CR-01: Interrupted write leaves partial file that is silently skipped on re-run

**File:** `scripts/generate_datasets.py:285-303` (write) and `scripts/generate_datasets.py:316` / `342` (skip guard)

**Issue:** `save_as_csv` writes large trajectories in sequential 50-frame chunks using
`mode="w"` for the first chunk then `mode="a"` for subsequent ones. If the process is
interrupted after the first chunk is flushed to disk (e.g. OOM, Ctrl-C, SLURM wall-time
kill) the output CSV already exists but is incomplete. On the next invocation
`out_path.exists()` returns `True` and the file is silently skipped. For the
`realistic_kobitski` variant this means 7 of 8 chunks (roughly 320 out of 370 frames)
are silently dropped from every downstream evaluation that consumes the file.

The same risk applies to all 5 sources × 9 augmentation variants in `generate_semi_synthetic`.

**Fix:** Write to a temp path and rename atomically on success:

```python
def save_as_csv(
    traj: dict,
    path: Path,
    chunk_frames: int = 50,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    frames     = sorted(traj.keys())
    total_rows = 0
    first      = True

    try:
        for start in range(0, len(frames), chunk_frames):
            chunk = frames[start : start + chunk_frames]
            df    = pd.concat([_frame_to_df(fi, traj[fi]) for fi in chunk])
            df.to_csv(tmp_path, index=False, mode="w" if first else "a", header=first)
            total_rows += len(df)
            first = False
        tmp_path.rename(path)  # atomic on POSIX; same filesystem guaranteed
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"    → {path.relative_to(ROOT)}  ({total_rows:,} rows)")
```

---

## Warnings

### WR-01: `sample_bowl_frame` lacks the `n <= 0` guard present in all sibling functions

**File:** `scripts/generate_datasets.py:151-159`

**Issue:** `sample_ball_shell` (line 131) and `sample_bowl_shell` (line 171) both return
`np.empty((0, 3), ...)` immediately when `n <= 0`. `sample_bowl_frame` does not. When
called with `n=0` the `while total < n` loop is never entered, `collected` stays an empty
list, and `np.concatenate([], axis=0)` raises `ValueError: need at least one array to
concatenate`.

Currently the only call site passes `n_base >= 100`, so this is not reachable in practice.
However the inconsistency creates a silent trap for any future caller or refactor that
introduces a smaller `n_base`.

**Fix:**

```python
def sample_bowl_frame(
    n: int,
    R: float,
    d: float,
    rng: np.random.Generator,
    batch_mult: int = 10,
) -> np.ndarray:
    """Uniform sample inside bowl(R, d) via rejection sampling."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    collected: list[np.ndarray] = []
    ...
```

### WR-02: Source dataset loaded unconditionally even when all augmented variants already exist

**File:** `scripts/generate_datasets.py:330-335`

**Issue:** `generate_semi_synthetic` loads each source dataset (lines 332–334) before
entering the inner augmentation loop. The per-variant skip check (line 342) is inside the
inner loop. If all 9 augmented variants for a source already exist on disk, the source is
still loaded in full — potentially 15 000 points × 370 frames for `realistic_kobitski`.
This wastes significant I/O and memory on every subsequent run (e.g. a re-run to add only
one missing variant).

**Fix:** Check whether any variant needs generating before loading:

```python
for src_name, src_cfg in REAL_SOURCES.items():
    # Determine which (aug_type, param) pairs still need work.
    pending = []
    for aug_type, param_list in AUGMENTATION_GRID.items():
        for param_name, value in param_list:
            label    = f"{param_name}{value:.1f}"
            name     = f"{src_name}_{aug_type}_{label}"
            out_path = SEMI_DIR / name / f"{name}.csv"
            if not out_path.exists():
                pending.append((aug_type, param_name, value, out_path))

    if not pending:
        print(f"\n  skip (all exist): {src_name}")
        continue

    print(f"\n  loading {src_name} …")
    if src_cfg["fmt"] == "tracklets":
        dataset, _ = load_data_from_tracklets(str(src_cfg["path"]), device="cpu")
    else:
        dataset = load_shah_from_csv(src_cfg["path"], device="cpu")
    print(f"  loaded {len(dataset)} frames")

    for aug_type, param_name, value, out_path in pending:
        ...
```

### WR-03: Hardcoded `seed=42` in `DataFactory.augment` makes noise and dropout patterns identical across all sources

**File:** `scripts/generate_datasets.py:347-349`

**Issue:** A new `DataFactory` instance is created per `(aug_type, value)` pair. Each
instance calls `add_gaussian_noise(..., seed=42)` or `drop_points(..., seed=42)`. Because
the seed is reset identically every time, `kobitski_ew06` and `shah_sample1` with
`sigma=1.0` receive numerically identical noise fields (up to per-frame point count),
and all three dropout fractions for a given source share the same initial permutation
from `torch.manual_seed(42)`. This reduces the independence of the augmented variants and
may invalidate cross-source comparisons in the evaluation.

The root cause is that `DataFactory.augment` has no per-call seed injection — it always
uses 42. The script has no mechanism to pass a source-specific or variant-specific seed.

**Fix (minimal):** Pass a source-dependent seed by deriving one from the source name and
augmentation parameters, then wrap the augment call:

```python
import hashlib

def _derive_seed(src_name: str, aug_type: str, value: float) -> int:
    digest = hashlib.md5(f"{src_name}_{aug_type}_{value}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFF_FFFF  # positive int32

# In the inner loop, before DataFactory.augment:
_seed = _derive_seed(src_name, aug_type, value)
torch.manual_seed(_seed)   # sets global torch seed before DataFactory resets it
```

A cleaner fix would be to add a `seed` parameter to `DataFactory.augment` and propagate
it to the individual step methods.

---

## Info

### IN-01: `generate_semi_synthetic` does not accept a seeded `rng` parameter — inconsistent with `generate_fully_synthetic`

**File:** `scripts/generate_datasets.py:327`

**Issue:** `generate_fully_synthetic(rng)` takes an `np.random.Generator` and uses it
throughout. `generate_semi_synthetic()` takes no RNG parameter and uses only the
hardcoded `seed=42` via `DataFactory` internals. The entry point at line 358 passes
`rng` to the first call but nothing to the second:

```python
rng = np.random.default_rng(SEED)
generate_fully_synthetic(rng)
generate_semi_synthetic()        # no rng
```

This is an API inconsistency that makes the seeding strategy of the two halves of the
script harder to reason about and audit together. It is also the structural reason
WR-03 exists: there is nowhere to thread an external seed into the semi-synthetic path.

**Fix:** Add an `rng` (or `seed`) parameter to `generate_semi_synthetic` and propagate
it to `_derive_seed` (see WR-03). Callers at the entry point then control reproducibility
from a single `SEED` constant.

---

_Reviewed: 2026-06-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
