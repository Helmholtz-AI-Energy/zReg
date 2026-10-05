---
plan: 55-01
phase: 55
status: complete
started: 2026-07-29
completed: 2026-07-29
---

# Plan 55-01 Summary: Spherical-Cap and Gaussian Label Generators

## What Was Built

Two new public label generators added to `src/zreg/data_generation/labels.py` and exported from `zreg.data_generation`:

- **`assign_cap_labels(trajectory, pole, theta_deg, label_inside=2, label_outside=1, seed=None)`** — hard spherical-cap binary labelling. Points with angular distance strictly less than `theta_deg` from `pole` receive `label_inside`; all others receive `label_outside`.

- **`assign_gaussian_labels(trajectory, pole, sigma_deg, label_inside=2, label_outside=1, seed=42)`** — soft Bernoulli labelling. Per-point probability `p = exp(-θ²/(2·σ²))` where θ is angular distance from `pole` in degrees; `torch.bernoulli(p)` determines label assignment.

A shared private helper `_angular_distance_deg(pos, pole)` implements the angular-distance contract (pole normalisation, eps-guarded per-point norm, clamped cosine similarity, `torch.rad2deg(torch.acos(...))`).

## Key Files

- `src/zreg/data_generation/labels.py` — `_angular_distance_deg`, `assign_cap_labels`, `assign_gaussian_labels` added; `__all__` extended
- `src/zreg/data_generation/__init__.py` — import line and `__all__` extended with both new functions
- `tests/test_generators.py` — `TestSphericalLabelGenerators` class (13 new tests)

## Commits

- `c67c429` feat(55-01): add assign_cap_labels and assign_gaussian_labels to data_generation.labels
- `be97d78` test(55-01): add TestSphericalLabelGenerators covering cap and gaussian label functions

## Test Results

- 13 new tests in `TestSphericalLabelGenerators` — all pass
- `tests/test_generators.py` full suite: 65 passed, 0 failed (no regressions)

## Must-Have Verification

- [x] `assign_cap_labels` sets `pc["label"]` to `label_inside` for points within `theta_deg` of the pole and `label_outside` otherwise
- [x] `assign_gaussian_labels` uses `exp(-theta²/(2·sigma²))` Bernoulli probability
- [x] Both return a new dict and never mutate input (deep-copy contract, D-03)
- [x] Both produce `torch.long` tensors of shape `(N,)`; empty frames handled by tensor ops
- [x] Pole normalised internally; non-unit-length pole yields identical results

## Deviations

**Executor agent (worktree) deviated** from the plan's specified interface: used `axis: int` + radians + sigmoid instead of `pole` direction vector + degrees + Gaussian probability. The worktree was discarded without merging. Orchestrator implemented the correct interface inline per the plan spec.

## Self-Check: PASSED
