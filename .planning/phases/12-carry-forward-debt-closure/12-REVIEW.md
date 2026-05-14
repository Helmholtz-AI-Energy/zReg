---
phase: 12-carry-forward-debt-closure
reviewed: 2026-05-13T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - src/zreg/cpd/base.py
  - src/zreg/cpd/_registration.py
  - src/zreg/color_transfer.py
  - src/zreg/__init__.py
  - src/zreg/dataset.py
  - src/zreg/downsampling.py
  - src/zreg/transforms/homogeneous.py
  - tests/conftest.py
  - VALIDATION.md
findings:
  critical: 5
  warning: 6
  info: 2
  total: 13
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-05-13T00:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 12 made four categories of changes: (1) replacing `typing.Callable` with
`collections.abc.Callable`; (2) converting an absolute import to relative in
`color_transfer.py` and adding `config` to `__init__.py`; (3) creating
`VALIDATION.md`; and (4) making all Open3D imports lazy across `dataset.py`,
`downsampling.py`, `homogeneous.py`, and `_registration.py`.

The lazy-import work and the `Callable` migration are mechanically correct.
However, five pre-existing and newly-introduced bugs survive in the reviewed
files that will cause crashes or incorrect results at runtime. The most
severe are in `color_transfer.py` (where both `cpd_weighted` and the
`source_color=None` path crash unconditionally) and in `_registration.py`
(where `cpd_registration` with `use_color=True` crashes before any
computation runs).

---

## Critical Issues

### CR-01: `cpd_registration` with `use_color=True` always raises `ValueError`

**File:** `src/zreg/cpd/_registration.py:108-128`

**Issue:** When `use_color=True`, lines 109-110 concatenate `pos` and `color`
into a single `(N, 6)` tensor called `sourcei` and then pass it as the `source`
argument to the CPD constructor — but `source_colors` is never extracted or
passed separately. The base class `__init__` immediately raises:

```
ValueError: use_color=True requires source_colors to be provided. Got source_colors=None.
```

Additionally, `targeti` (also a concatenated 6-D tensor) is passed to
`cpd.registration()` without a `target_colors` keyword argument, so even if
the `ValueError` were bypassed, the E-step would fail with an
`AttributeError` on `None.shape[1]`. The same defect exists in
`init_cpd_from_existing` (line 195-197).

**Fix:**
```python
# In cpd_registration, split pos and color instead of concatenating:
if use_color:
    source_pos   = source["pos"]
    source_colors = source["color"]
    target_pos   = target["pos"]
    target_colors = target["color"]
else:
    source_pos    = source["pos"]
    source_colors = None
    target_pos    = target["pos"]
    target_colors = None

# Pass them separately:
cpd = RigidCPD(source_pos, use_color=use_color, source_colors=source_colors,
                log_freq=log_freq, **kwargs)
cpd.set_callbacks(callbacks)
return cpd.registration(target_pos, w, maxiter, tol, target_colors=target_colors)
```

Apply the same fix in `init_cpd_from_existing`.

---

### CR-02: `_transfer_colors_cpd_weighted` shape guard is inverted — always raises `ValueError`

**File:** `src/zreg/color_transfer.py:196-207`

**Issue:** The CPD E-step produces `pmat` of shape `(N_source, N_target)` —
confirmed by `cdist(t_source, target)` in `base.py:140` and the `_types.py`
docstring ("shape: N x M"). The guard at line 196 accepts `(n_target,
n_source)` (the transposed, non-standard orientation) and raises `ValueError`
on the actual shape at line 199-202. Every call from real CPD output will
always raise. The weighted-average at line 215 is also computed in the wrong
direction when the wrong orientation is accepted.

**Fix:**
```python
# Accept the real (n_source, n_target) shape and transpose for per-target weighting:
if pmat.shape == (n_source, n_target):
    prob_matrix = pmat.T   # (n_target, n_source)
elif pmat.shape == (n_target, n_source):
    prob_matrix = pmat     # already transposed
else:
    raise ValueError(
        f"pmat has unexpected shape {pmat.shape}; "
        f"expected ({n_source}, {n_target}) or ({n_target}, {n_source})"
    )
# Then normalize and matmul as before.
```

---

### CR-03: `transfer_colors` crashes with `TypeError` when `source["color"]` is `None`

**File:** `src/zreg/color_transfer.py:73-79`

**Issue:** When `source` is a `zRegPointCloud`, line 75 assigns
`source_colors = source["color"]` without checking for `None`. The
`zRegPointCloud` constructor initializes `color` to `None` by default. The
`None` check at line 78 only triggers for the `torch.Tensor` path. Downstream
private functions (`_transfer_colors_nearest_neighbor` line 155,
`_transfer_colors_knn_voting` line 254, etc.) all assume `source_colors` is
a `Tensor` and crash with `TypeError: 'NoneType' object is not subscriptable`.

**Fix:**
```python
if isinstance(source, zRegPointCloud):
    source_pos = source["pos"]
    source_colors = source["color"]
    if source_colors is None:
        raise ValueError(
            "source['color'] is None — cannot transfer colors from a "
            "zRegPointCloud without color data."
        )
```

---

### CR-04: `_farthest_point_ds_internal` with `use_precomputed_indexes=True` self-indexes `fps-idx`, causing `IndexError`

**File:** `src/zreg/downsampling.py:308-312`

**Issue:** When `use_precomputed_indexes=True`:
```python
indexes = target["fps-idx"][:points]          # e.g. [42, 7, 99]
# ...
target["fps-idx"] = target["fps-idx"][indexes] # indexes fps-idx BY its own values
```
`fps-idx` holds indices into the original point array (e.g., values up to
`N-1` where `N` can be in the thousands). Indexing a short array with those
large values triggers `IndexError`. Even when no exception occurs, the result
is semantically wrong: `fps-idx[42]` is not the 43rd selected point; it is
an arbitrary precomputed FPS rank.

**Fix:**
```python
# When use_precomputed_indexes=True, indexes already IS the fps-idx slice:
if not use_precomputed_indexes:
    # ... compute indexes via fps() ...
    target["fps-idx"] = target["fps-idx"][indexes] if target["fps-idx"] is not None else None
else:
    indexes = target["fps-idx"][:points]
    # fps-idx for the subset is just `indexes` itself (the selected original indices):
    target["pos"]     = target["pos"][indexes]
    target["color"]   = target["color"][indexes]
    target["id"]      = target["id"][indexes]
    target["fps-idx"] = indexes
    return target
```

---

### CR-05: `load_shah_from_csv` crashes with `KeyError` when time indices are non-contiguous

**File:** `src/zreg/dataset.py:338`

**Issue:** Line 338 iterates `range(min(pcs), max(pcs) + 1)`, which assumes
all integer time values between the minimum and maximum are present in the
CSV. Any gap (e.g., time-points 0, 1, 5 in the data) causes a `KeyError`
when the loop attempts to access `pcs[2]`, `pcs[3]`, `pcs[4]`. The function
also produces a 1D color tensor of shape `(N,)` at line 341 (from a flat
list of scalar `layer` values), which conflicts with the `(N, C)` shape
expected by `transfer_colors` and the CPD E-step.

**Fix:**
```python
# Iterate only over keys that actually exist:
for i in sorted(pcs.keys()):
    pcs[i] = zRegPointCloud(
        pos=torch.tensor(pcs[i]["pos"], device=device),
        # Wrap scalar labels in a list-of-lists to produce shape (N, 1):
        color=torch.tensor([[v] for v in pcs[i]["labels"]], device=device),
        id=torch.tensor(pcs[i]["id"], device=device),
    )
```

---

## Warnings

### WR-01: `log_freq: int = True` in `CoherentPointDrift.__init__` — boolean default for int parameter

**File:** `src/zreg/cpd/base.py:55`

**Issue:** The default value is `True` (a `bool`), which Python treats as `1`.
With `log_freq=1` every single iteration is logged, contradicting the
docstring's "Log frequency" description and the default of `100` used in all
subclass constructors and `cpd_registration`. Any caller that relies on the
base class default gets unexpectedly verbose output.

**Fix:**
```python
log_freq: int = 100,   # consistent with subclass defaults
```

---

### WR-02: `_transfer_colors_knn_voting` has no guard when `k > n_source`

**File:** `src/zreg/color_transfer.py:261`

**Issue:** `torch.topk(distances, k=k, dim=1, largest=False)` raises
`RuntimeError` if `k` exceeds the number of source points. There is no
validation before this call.

**Fix:**
```python
n_source = source_pos.shape[0]
if k > n_source:
    raise ValueError(
        f"k={k} exceeds the number of source points ({n_source}). "
        "Reduce k or provide more source points."
    )
```

---

### WR-03: `_transfer_colors_gaussian_kernel` produces `NaN` when `sigma=0`

**File:** `src/zreg/color_transfer.py:305`

**Issue:** When `sigma=0` (either passed explicitly or defaulting to a
near-zero value), division by `2 * sigma**2` is a division by zero. PyTorch
produces `inf` weights, and after normalization the result becomes `NaN`,
silently corrupting all output colors.

**Fix:**
```python
if sigma <= 0:
    raise ValueError(f"sigma must be positive, got sigma={sigma}")
```

---

### WR-04: `running_avg` initialized with garbage values `[0, 1, 2, 3]` pollutes convergence check

**File:** `src/zreg/cpd/base.py:344`

**Issue:** `running_avg = torch.arange(4, ...)` seeds the circular buffer
with `[0.0, 1.0, 2.0, 3.0]`. These fictitious "previous-iteration" values
are mixed with real `q` values for the first four iterations. Depending on
the magnitude of `q`, this can cause the convergence condition
(`running_avg.abs().diff().mean().abs() < tol`) to fire prematurely (all
four slots filled with the same large `q` before the buffer sees a real trend)
or be delayed by spurious large diffs from the initial garbage.

**Fix:**
```python
# Initialize all slots to the first real q value (set after first M-step):
running_avg = None
# Then inside the loop, after the first M-step:
if running_avg is None:
    running_avg = torch.full((4,), res.q, dtype=target.dtype, device=target.device)
else:
    running_avg[i % 4] = res.q
```

---

### WR-05: `farthest_point_down_sample` docstring states wrong default for `use_precomputed_indexes`

**File:** `src/zreg/downsampling.py:250-251`

**Issue:** The docstring at line 251 reads "by default True" but the function
signature at line 233 declares `use_precomputed_indexes: bool = False`. Any
caller reading only the docstring will believe precomputed indices are used
by default and be surprised when they are not.

**Fix:**
```
use_precomputed_indexes : bool, optional
    Whether to use precomputed farthest point indices, by default False
```

---

### WR-06: `cpd_registration` and `init_cpd_from_existing` shadow the module-level `HAS_OPEN3D` name

**File:** `src/zreg/cpd/_registration.py:100, 187`

**Issue:** Both functions contain `o3d, HAS_OPEN3D = _get_open3d()`, creating
a local variable that shadows the module-level `HAS_OPEN3D` imported from
`dataset`. The local variable is semantically equivalent in value but the
name reuse is confusing: a reader scanning the file sees `HAS_OPEN3D` used
both as a module constant and as a return value, and a linter may warn about
the shadowing.

**Fix:**
```python
o3d, _has_open3d = _get_open3d()
if _has_open3d and isinstance(source, o3d.t.geometry.PointCloud):
    ...
```

---

## Info

### IN-01: `set_source` validates `source` redundantly when `source_colors` is also given

**File:** `src/zreg/cpd/base.py:85-87`

**Issue:** Line 85 validates `source` alone; line 87 then validates `(source,
source_colors)` together. The second call re-validates `source`, which is
harmless but wasteful and slightly misleading.

**Fix:**
```python
_validate_tensors(source, names=["source"])
if source_colors is not None:
    _validate_tensors(source_colors, names=["source_colors"])
```

---

### IN-02: Commented-out code blocks left in `dataset.py` and `downsampling.py`

**File:** `src/zreg/dataset.py:150-152`; `src/zreg/downsampling.py:447-456`

**Issue:** `dataset.py` lines 150-152 contain a commented-out alternative
tensor-assignment block (`# pc[i]["pos"] = ...`). `downsampling.py` lines
447-456 contain a fully commented-out `voxel_down_sample` function with a
TODO. Dead code increases maintenance burden and can mislead future readers.

**Fix:** Remove both commented-out blocks. If `voxel_down_sample` is planned
for a future milestone, track it in the project backlog rather than in source
code.

---

_Reviewed: 2026-05-13T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
