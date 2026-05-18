---
phase: 14-synthetic-data-generators
plan: "02"
subsystem: eval/generators
tags:
  - synthetic-data
  - corruption
  - labels
  - point-cloud
  - eval-framework

dependency_graph:
  requires:
    - phase: 14-01
      provides: "eval/generators/__init__.py (3-symbol package), generators.py, transforms.py"
    - src/zreg/dataset.py (zRegPointCloud field semantics)
    - src/zreg/metrics/label_transfer.py (compute_f1 dtype contract D-07)
  provides:
    - eval/generators/corruption.py — add_gaussian_noise and add_outliers immutable wrappers
    - eval/generators/labels.py — generate_labels (Voronoi) and remove_labels utilities
    - eval/generators/__init__.py — updated to 7-symbol __all__ (full EVAL-03 surface)
  affects:
    - Plan 14-03 (tests): pytest coverage for all seven public functions including corruption + labels
    - Phase 16 (runners): eval/run_synthetic.py can now import all seven generator symbols

tech-stack:
  added: []
  patterns:
    - "Voronoi label assignment via torch.cdist argmin — spatially coherent clusters without numpy"
    - "Sentinel -1 extension for all per-point auxiliary fields (id, 1-D color, fps-idx) in add_outliers"
    - "2-D color tensor extension with zeros (not -1) to preserve row-count invariant"

key-files:
  created:
    - eval/generators/corruption.py
    - eval/generators/labels.py
  modified:
    - eval/generators/__init__.py

key-decisions:
  - "fps-idx extended with sentinel -1 in add_outliers: keeps all per-point fields length-consistent with pos"
  - "2-D color tensors extended with zeros in add_outliers: sentinel -1 is valid only for 1-D label tensors; RGB-style (N,3) color uses zero rows"
  - "sigma==0.0 handled via math (t + 0*r == t) without branching: torch.equal holds for float tensors without NaN/subnormal values"
  - "Voronoi chosen over random-uniform label assignment: leverages existing torch.cdist pattern and produces spatially coherent clusters"
  - "generate_labels seeds torch.manual_seed once at entry, consuming per-frame randn calls in dict insertion order for determinism"

patterns-established:
  - "deepcopy before mutate: all corruption/label wrappers call copy.deepcopy(trajectory) and operate on the copy"
  - "seed contract: if seed is not None: torch.manual_seed(seed) once at function entry; None = caller owns RNG"
  - "no numpy in eval/generators: all stochastic ops use torch.randn / torch.randn_like / torch.cdist"

requirements-completed:
  - EVAL-03

duration: 12min
completed: "2026-05-18"
---

# Phase 14 Plan 02: Corruption Wrappers and Voronoi Label Utilities Summary

**Gaussian-noise and outlier-injection corruption wrappers plus Voronoi-based label assignment, completing the seven-symbol EVAL-03 eval/generators package surface.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-18T00:00:00Z
- **Completed:** 2026-05-18
- **Tasks:** 3
- **Files modified:** 3 (2 created, 1 updated)

## Accomplishments

- `add_gaussian_noise` — deepcopy + `torch.randn_like * sigma`, immutable, seed-deterministic, raises `ValueError` on `sigma < 0`
- `add_outliers` — deepcopy + randn outlier append; extends `id`, 1-D `color`, and `fps-idx` with sentinel `-1`; 2-D `color` extended with zeros; raises `ValueError` on `n_outliers < 0`
- `generate_labels` — Voronoi assignment via `torch.cdist` argmin; `torch.long` output, shape `(N,)`, directly passable to `compute_f1` per D-07
- `remove_labels` — deepcopy + `pc["color"] = None` per D-08; no seed parameter needed (deterministic)
- `eval/generators/__init__.py` updated to 7-element `__all__`, four `from .X import` lines (no aliases)

## Final Function Signatures

```python
# eval/generators/corruption.py
def add_gaussian_noise(
    trajectory: dict[int, zRegPointCloud],
    sigma: float = 0.01,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]: ...

def add_outliers(
    trajectory: dict[int, zRegPointCloud],
    n_outliers: int,
    scale: float = 3.0,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]: ...

# eval/generators/labels.py
def generate_labels(
    trajectory: dict[int, zRegPointCloud],
    n_classes: int,
    seed: int | None = 42,
) -> dict[int, zRegPointCloud]: ...

def remove_labels(
    trajectory: dict[int, zRegPointCloud],
) -> dict[int, zRegPointCloud]: ...
```

## `fps-idx` Sentinel Extension Decision

The `add_outliers` implementation extends `pc["fps-idx"]` with sentinel `-1` values when that field is set (non-None). This was kept — all per-point fields must remain length-consistent with `pos` after outlier injection. Without this, `pos.shape[0] != fps_idx.shape[0]` would cause downstream consistency violations.

## Edge Case Decisions

- **`sigma == 0.0`:** No branch; `pc["pos"] + 0 * randn_like(pos)` is element-wise equal to the original (float addition of exact zero, no subnormals in Gaussian-blob inputs). `torch.equal` assertion passes.
- **Empty frames (`N == 0`):** `torch.cdist` returns `(0, n_classes)`; `argmin(dim=1)` returns `(0,)` `torch.long` — assigned without special-casing.
- **`n_outliers == 0`:** Allowed (no-op outlier pass useful in parameter sweeps); only `n_outliers < 0` is rejected.

## `eval.generators.__all__` after Plan 02

```python
__all__ = [
    "generate_trajectory",
    "apply_rigid",
    "apply_affine",
    "add_gaussian_noise",
    "add_outliers",
    "generate_labels",
    "remove_labels",
]
```

Exactly 7 entries, no duplicates.

## Task Commits

Each task was committed atomically:

1. **Task 1: add_gaussian_noise + add_outliers (corruption.py)** - `cff45bb` (feat)
2. **Task 2: generate_labels + remove_labels (labels.py)** - `2347395` (feat)
3. **Task 3: extend __init__.py to 7-symbol __all__** - `feed6f8` (feat)

**Plan metadata:** (see final commit below)

## Files Created/Modified

- `eval/generators/corruption.py` — `add_gaussian_noise` (randn noise, deepcopy) and `add_outliers` (randn outliers, sentinel extension for id/color/fps-idx)
- `eval/generators/labels.py` — `generate_labels` (Voronoi via torch.cdist) and `remove_labels` (color=None)
- `eval/generators/__init__.py` — extended from 3-symbol to 7-symbol `__all__`, 4 `from .X import` lines

## Decisions Made

- **Voronoi over random-uniform:** `torch.cdist` pattern already established in the codebase (`alignment.py` line 68); produces spatially coherent clusters that better reflect real dataset structure.
- **`fps-idx` sentinel extension retained:** All per-point fields must remain length-consistent with `pos` after outlier injection. Omitting fps-idx extension would break downstream length invariants.
- **2-D color uses zeros not sentinel:** Sentinel `-1` is semantically meaningful only for 1-D integer label tensors. A 2-D RGB tensor row with `-1` values would corrupt downstream colour operations; zeros are the safe neutral extension.
- **`sigma == 0.0` no branch:** Mathematical approach (`t + 0 * r == t`) avoids dead-code branching and satisfies the `torch.equal` acceptance criterion for normal float tensors.

## Deviations from Plan

None — plan executed exactly as written. All acceptance criteria satisfied on first implementation attempt.

## Issues Encountered

The import path for running verification scripts required `KMP_DUPLICATE_LIB_OK=TRUE` to suppress an OpenMP runtime conflict (`libomp.dylib already initialized`) when running from the repo root. This is a pre-existing environment issue unrelated to this plan's changes; the library code itself contains no OpenMP dependency.

## Known Stubs

None — all four functions produce real output. `add_gaussian_noise` applies real Gaussian noise; `add_outliers` appends real randn points; `generate_labels` runs actual Voronoi assignment; `remove_labels` sets color fields to None.

## Threat Flags

No new network endpoints, auth paths, or trust boundary changes introduced. The new modules are pure computation (torch tensors, copy operations) with no I/O, network access, or external data loading.

## Next Phase Readiness

- `eval/generators` package is complete (7 symbols): ready for Plan 14-03 test suite
- Plan 14-03 will create `tests/test_generators.py` and update `tests/conftest.py` with repo-root sys.path insertion
- Downstream Phase 16 runners can `from eval.generators import *` and receive exactly the seven EVAL-03 symbols

## Self-Check: PASSED

Files exist:
- eval/generators/corruption.py: FOUND
- eval/generators/labels.py: FOUND
- eval/generators/__init__.py: FOUND (updated)

Commits exist:
- cff45bb (Task 1 — corruption.py): FOUND
- 2347395 (Task 2 — labels.py): FOUND
- feed6f8 (Task 3 — __init__.py update): FOUND

Verification checks:
- All 7 imports from eval.generators succeed: PASS
- __all__ has exactly 7 names, set matches expected: PASS
- 4 `from .X import` lines, no aliases: PASS
- add_gaussian_noise defaults (0.01, 42): PASS
- add_outliers defaults (3.0, 42): PASS
- sigma==0.0 returns torch.equal pos: PASS
- sigma=0.1 reproducible across two calls: PASS
- add_outliers increases pos.shape[0] by n_outliers: PASS
- input trajectory not mutated after any call: PASS
- generate_labels returns torch.long (N,) with values in [0, n_classes): PASS
- generate_labels reproducible: PASS
- compute_f1(labelled, labelled) == 1.0: PASS
- remove_labels sets color to None without mutating input: PASS
- ValueError messages contain correct field name (sigma / n_classes / n_outliers): PASS

---
*Phase: 14-synthetic-data-generators*
*Completed: 2026-05-18*
