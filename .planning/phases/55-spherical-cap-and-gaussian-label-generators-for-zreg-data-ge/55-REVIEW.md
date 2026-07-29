---
phase: 55
status: issues
reviewed: 2026-07-30
findings: 4
severity_breakdown: {high: 2, medium: 1, low: 1}
---

# Code Review — Phase 55: Spherical-Cap and Gaussian Label Generators

Files reviewed: `src/zreg/data_generation/labels.py`, `src/zreg/data_generation/__init__.py`, `tests/test_generators.py`

## Findings

### [HIGH] Zero-length pole vector causes silent NaN corruption
**File:** `src/zreg/data_generation/labels.py:133`

`_angular_distance_deg` computes `pole_unit = pole_vec / pole_vec.norm()` with no guard. When `pole=(0,0,0)`, `pole_vec.norm()` is 0.0, producing a NaN tensor. This propagates through the dot product and `acos`, so every point in every frame receives a NaN angular distance. `torch.where(NaN < theta_deg, ...)` returns NaN, and `.to(torch.long)` casts NaN to 0 on CPU silently — all labels become 0 with no error raised.

**Fix:** add `if pole_vec.norm() < eps: raise ValueError("pole must be non-zero")` before the division.

---

### [HIGH] sigma_deg=0 causes division by zero, silently corrupting all labels
**File:** `src/zreg/data_generation/labels.py:243`

`assign_gaussian_labels` computes `prob = torch.exp(-(theta**2) / (2.0 * sigma_deg**2))`. When `sigma_deg=0`, the denominator is `0.0`. PyTorch evaluates `-(theta**2) / 0.0` as `-inf` for theta>0 and `nan` for theta=0 (0/0). The resulting `prob` tensor contains `0.0` and `nan`; `torch.bernoulli(nan)` raises a `RuntimeError` on GPU or returns garbage on CPU. No validation exists.

**Fix:** add `if sigma_deg <= 0: raise ValueError(f"sigma_deg must be > 0, got {sigma_deg}")` at function entry.

---

### [MEDIUM] assign_cap_labels perturbs global PyTorch RNG state via seed with no RNG draws
**File:** `src/zreg/data_generation/labels.py:182`

When `seed is not None`, `assign_cap_labels` calls `torch.manual_seed(seed)` before the frame loop, but the cap assignment is purely deterministic and draws no random numbers. The result: calling `assign_cap_labels(traj, pole, 30.0, seed=0)` resets the global RNG, silently disrupting the reproducibility of any subsequent `assign_gaussian_labels`, `generate_labels`, or other RNG-using call in the same process. The docstring acknowledges the parameter "consumes no RNG" but does not warn about the global side effect.

**Fix:** Either remove the `seed` parameter entirely from `assign_cap_labels`, or guard with a no-op: do not call `torch.manual_seed` when `seed is not None` for a deterministic function.

---

### [LOW] pole_unit recomputed on every frame inside the per-frame loop
**File:** `src/zreg/data_generation/labels.py:129`

`_angular_distance_deg` is called once per frame from inside the trajectory loop. Each call re-runs `torch.tensor(pole, ...)`, `.norm()`, and the division — all per frame. For a 500-frame trajectory this is 500 redundant Python→Tensor conversions and GPU kernel launches. Normalizing the pole once before the frame loop and passing the pre-computed unit tensor would eliminate the overhead.

**Fix (optional):** pre-normalize pole before the per-frame loop and pass the unit tensor into a simplified helper.
