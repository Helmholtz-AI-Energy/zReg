# Phase 57: Synthetic Labeled Subsample-Pair Generation - Pattern Map

**Mapped:** 2026-08-04
**Files analyzed:** 4 (all modified, none newly created — new method added inside an existing file)
**Analogs found:** 4 / 4 (all in-file analogs; no cross-file "closest match elsewhere" search was needed because every touched file already contains the closest sibling pattern to copy)

## IMPORTANT: The `transform_spec["type"]` key is NOT currently dispatched on anywhere

Before mapping patterns, a direct codebase check (required because CONTEXT.md flags a prior
citation-accuracy issue) found:

```
grep -rn 'transform_spec.get("type")\|transform_spec\["type"\]\|== "rigid"\|== "noise"' eval/
# => no matches
```

`"type"` is written into `transform_spec` dicts (e.g. `{"type": "rigid", ...}` at
`eval/data_factory.py:430`) purely as **documentation/metadata**. The actual code path taken by
`generate_target()` → `augment()` is determined entirely by **which augmentation keys are
present** (`"sigma"`, `"rotation_deg"`, `"dropout_fraction"`, ...), not by the `"type"` value.
`DataFactory.generate_target` strips `"type"` unconditionally at
`eval/data_factory.py:327` and never inspects it again:

```python
# eval/data_factory.py:327
augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
```

**Consequence for planning:** there is no existing `type`-discriminator dispatch pattern to
"reuse" for `type: "subsample_pair"` — CONTEXT.md's phrase "reusing the existing `pipeline_mode:
synthetic` dispatch" (D-04) refers to the `pipeline_mode == "synthetic"` branches (which DO exist
and ARE dispatched on, at the 4 call sites below), not to a `transform_spec["type"]` branch (which
does not exist). The planner must introduce **genuinely new** `transform_spec.get("type") ==
"subsample_pair"` conditionals at each of the 4 integration points below. The closest copyable
precedent for *how* to write such a conditional is the existing `tier_name == "sanity"` /
`pipeline_mode == "synthetic"` if/else structure already present at each site (shown per-site
below) — not a preexisting type-dispatch table.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog (in same file) | Match Quality |
|---|---|---|---|---|
| `eval/data_factory.py` (new method, name TBD e.g. `generate_subsample_pair`) | service (data-generation orchestrator) | transform / batch | `DataFactory.generate_target()` (lines 266-346) + `DataFactory.generate_training_triple()` (lines 348-439) + `DataFactory.drop_points()` (961-1025) | exact (composition of 3 existing methods in the same class) |
| `eval/config.py` (no schema change — confirm only) | config (pydantic model) | n/a | `EvalConfig.transform_spec` field (line 290) already typed `dict \| None` | exact — no analog needed, field already generic |
| `eval/runners/eval_runner.py` (`run()`, one new branch) | controller / orchestrator | request-response | Existing `pipeline_mode == "paired"` / `else` branch (lines 189-193) | exact (same `if/else` shape, needs a nested branch) |
| `eval/runners/optimizer.py` (`run()` + `_objective()`, three new branches) | controller / orchestrator (HPO) | request-response / batch (per-trial) | Existing `pipeline_mode == "synthetic"` branches at lines 231-237, 403-416, 444-456 | exact (same `if pipeline_mode == "synthetic": ... else: ...` shape, needs a nested branch in each) |

## Pattern Assignments

### `eval/data_factory.py` — new method (service, transform/batch)

**Analog 1 — transform-application machinery to call into (D-03 conditional):** `DataFactory.generate_target()`

**Imports** (`eval/data_factory.py:14-46`, already available, no new imports needed for the transform-reuse path):
```python
import logging
import math
import random
from pathlib import Path  # noqa: F401  (available for future use)

from zreg.core.dataset import (
    load_data_from_tracklets,
    load_shah_from_csv,
    zRegPointCloud,
)
from zreg.data_generation import (
    add_gaussian_noise,
    add_outliers,
    apply_affine,  # noqa: F401
    apply_rigid,
    generate_labels,
    generate_trajectory,
    sample_ball,
    sample_bowl,
)
from zreg.core.transforms import RigidTransformation

import torch

from eval.config import EvalConfig, EvalConfigError
from eval.types import TrainingTriple
```

**Guard-clause pattern to copy** (`eval/data_factory.py:319-333`, `generate_target`):
```python
# Guard: transform_spec must be non-empty
if not transform_spec:
    raise ValueError(
        "DataFactory.generate_target: transform_spec must be non-empty. "
        "Passing None or {} would produce a no-op (augment returns input "
        "by reference when augmentation_params is empty — Pitfall 1)."
    )
# Strip the 'type' discriminator key before passing to augment() dispatch (D-02)
augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
if not augment_params:
    raise ValueError(
        "DataFactory.generate_target: transform_spec contains only the "
        "'type' key — no augmentation keys remain after stripping 'type'. "
        "Provide at least one augmentation key (e.g. 'sigma', 'rotation_deg')."
    )
```
The new method's own `type: "subsample_pair"` guard should follow the same shape: validate the
subsample-specific keys are present (D-05: `source_fraction`/`target_fraction`, `seed`) before
doing any work.

**Save/restore-state try/finally pattern to copy** (`eval/data_factory.py:334-346`):
```python
original_params = self.config.augmentation_params
self._correspondence_idx = None  # D-03/D-04 (Phase 56): reset so this call composes a fresh correspondence map
try:
    self.config.augmentation_params = augment_params
    result = self.augment(dataset)
finally:
    self.config.augmentation_params = original_params
# D-03: store instance state after successful augmentation
self._synthetic_target = result
self._source_dataset = dataset
self._transform_spec = transform_spec
return result
```
D-03 (this phase) says: `run_alignment=True` → subsample **and** call `generate_target`-style
transform application; `run_alignment=False` → subsample only. This save/restore block is exactly
what to call into (or mirror) for the `run_alignment=True` half — reading
`self.config.run_alignment` to decide whether to invoke this path at all.

**Analog 2 — synthetic-from-scratch generation mode (D-02: "no dataset available" path):**
`DataFactory.generate_training_triple()` (`eval/data_factory.py:348-439`)

**Seed-driven RNG + geometry dispatch** (`eval/data_factory.py:410-427`):
```python
rng = random.Random(seed)
if n_points is None:
    n_points = rng.randint(100, 300)
if shape is None:
    shape = "ball" if seed % 2 == 0 else "bowl"

if shape == "ball":
    pos = sample_ball(n_points, seed=seed)
elif shape == "bowl":
    pos = sample_bowl(n_points, seed=seed)
else:
    raise ValueError(
        f"DataFactory.generate_training_triple: unknown shape {shape!r}; "
        f"expected 'ball' or 'bowl'"
    )

base = {0: zRegPointCloud(pos=pos)}
source = generate_labels(base, n_classes=n_classes, seed=seed)
```
This is the "synthesize fresh ball/bowl geometry first" mode from D-02/D-05 — the new method
should offer this as one of its two input modes (the other being "subsample the already-loaded
dataset" from an explicit `dict[int, zRegPointCloud]` argument, e.g. from `load_real()`/
`load_target()`).

**Critical anti-pattern documented at `eval/data_factory.py:374-384`** (do not violate in the new
method either):
```python
CRITICAL (Pitfall 2): this method deliberately does NOT read or write
``self._synthetic_dataset`` — it has no early-return cache, unlike
:meth:`generate_synthetic`'s D-08/D-09 "construct once, cache by
reference" contract. Every call recomputes from scratch so that N
seeds produce N distinct results. ... Do NOT "fix" this method to cache its result;
that would silently return the first seed's triple for every
subsequent seed
```
D-06/D-07 of this phase (fixed seed by default, optional seed-list averaging) has the exact same
requirement: the new method must NOT cache-by-reference across calls with different seeds, or
HPO's seed-list averaging (D-07) would silently collapse to one repeated pair.

**Analog 3 — correspondence-tracked subsampling to call twice (once per view):**
`DataFactory.drop_points()` (`eval/data_factory.py:961-1025`) and
`DataFactory.sample_new_points()` (`eval/data_factory.py:1027-1104`)

**Correspondence composition pattern** (`eval/data_factory.py:1006-1025`, full method body — this
is the core mechanism D-01 says to reuse rather than reimplement):
```python
def drop_points(
    self,
    dataset: dict[int, zRegPointCloud],
    fraction: float,
    seed: int = 42,
) -> dict[int, zRegPointCloud]:
    torch.manual_seed(seed)
    result: dict[int, zRegPointCloud] = {}
    new_corr: dict[int, torch.Tensor] = {}
    for i, pc in dataset.items():
        n = pc["pos"].shape[0]
        keep = max(1, round(n * (1.0 - fraction)))
        idx = torch.randperm(n, device=pc["pos"].device)[:keep].sort().values
        result[i] = zRegPointCloud(
            pos=pc["pos"][idx],
            label=pc["label"][idx] if pc["label"] is not None else None,
            id=pc["id"][idx] if pc["id"] is not None else None,
        )
        result[i]["fps-idx"] = pc["fps-idx"][idx] if pc["fps-idx"] is not None else None
        # Phase 56 D-03: compose with prior correspondence, or start fresh
        if self._correspondence_idx is not None and i in self._correspondence_idx:
            new_corr[i] = self._correspondence_idx[i][idx]
        else:
            new_corr[i] = idx.clone()
    self._correspondence_idx = new_corr
    return result
```
**Key implication for the new method (CORRECTED — verified against the method body above):**
calling `drop_points()` twice (once for the "source" fraction, once for the "target" fraction) on
the same input dict does **not** simply "overwrite" the first call's correspondence with the
second — see lines 196-200 above: whenever `self._correspondence_idx` is already non-`None` for a
frame key, `drop_points()` **COMPOSES** with it (`new_corr[i] = self._correspondence_idx[i][idx]`),
treating the existing tensor as an index space to gather from. Calling `drop_points()` a second
time on `self` WITHOUT resetting `self._correspondence_idx = None` first is therefore a real bug,
not just a capture-timing nuance: the second call's `idx` is computed over the FULL `base_dataset`
range (e.g. up to index 99 on a 100-point frame), but it gets used to index the FIRST call's
already-reduced-length correspondence tensor (e.g. length 80 after an 0.2 drop fraction) —
`self._correspondence_idx[i][idx]` raises `IndexError: index 80 is out of bounds for dimension 0
with size 80` on any `idx` value >= 80. This happens on every realistic call (any
`source_fraction`/`target_fraction < 1.0`), not an edge case. The new method must therefore reset
`self._correspondence_idx = None` immediately **before EACH** of the two `drop_points()` calls
(so each call composes against a clean slate, i.e. hits the `idx.clone()` branch at line 200), and
also capture each call's resulting index tensor into its own local variable immediately
afterward, e.g.:
```python
self._correspondence_idx = None  # required: drop_points() composes with prior state if present
source_view = self.drop_points(base_dataset, 1.0 - source_fraction, seed=seed)
source_corr = self._correspondence_idx  # capture before the next reset clears it
self._correspondence_idx = None  # required: prevents composing against source_corr's shorter range
target_view = self.drop_points(base_dataset, 1.0 - target_fraction, seed=seed + 1)
target_corr = self._correspondence_idx
```
This is a real risk area analogous to the `_synthetic_target`/`_correspondence_idx` single-slot
state pattern already documented at class level (`eval/data_factory.py:148-154`):
```python
self._transform_spec: dict | None = None
self._preprocessing_stats: dict | None = None
# Phase 56 D-03/D-04: per-frame original-index correspondence map,
# populated by drop_points()/sample_new_points() and consumed by
# get_synthetic_ground_truth() to gather (not truncate) y_true when
# dropout/new-points change point counts. Deliberately reusable by
# Phase 57 (subsample-pair generation has the same need).
self._correspondence_idx: dict[int, torch.Tensor] | None = None
```

**Analog 4 — GT-extraction consumer the new method's output must remain compatible with:**
`DataFactory.get_synthetic_ground_truth()` (`eval/data_factory.py:678-767`)

**Gather-by-correspondence pattern** (`eval/data_factory.py:747-767`, the tail of the method):
```python
field = self.config.ground_truth_field
result: dict[int, torch.Tensor] = {}
for k, pc in self._source_dataset.items():
    field_values = pc[field]
    if field_values is not None:
        base = field_values.to(torch.long)  # D-04 + D-06: cast to torch.long
    else:
        base = torch.arange(pc["pos"].shape[0], dtype=torch.long)  # D-05 + D-06: ordinal fallback

    if self._correspondence_idx is not None and k in self._correspondence_idx:
        # Phase 56 D-03/D-04 (GT-02): gather by tracked correspondence
        # instead of returning positionally — produces a tensor
        # already sized to match the target's actual point count.
        corr = self._correspondence_idx[k]
        gathered = torch.full((corr.shape[0],), -1, dtype=torch.long)
        valid = corr >= 0
        gathered[valid] = base[corr[valid]]
        result[k] = gathered
    else:
        result[k] = base
return result
```
This method reads `self._source_dataset` and `self._correspondence_idx` (single-slot, target-side
only). The new method's output must either (a) call `generate_target`-compatible bookkeeping so
`get_synthetic_ground_truth()` keeps working unmodified when `run_alignment=True`, or (b) the
planner may need a natural extension of this method for the pair-correspondence case (per
canonical_refs) since `subsample_pair` produces correspondence on **both** sides (source view AND
target view), not just target-side as today's single `_correspondence_idx` slot assumes.

---

### `eval/config.py` — confirm-only, no schema change

`EvalConfig.transform_spec` (`eval/config.py:290`) is already:
```python
transform_spec: dict | None = None
```
with docstring at `eval/config.py:202-211`:
```python
transform_spec : dict or None
    Transform specification dict for synthetic mode.  Required (non-None)
    when ``pipeline_mode='synthetic'`` and
    ``DataFactory.generate_target()`` is invoked; ignored otherwise.
    ``ValueError`` is raised at ``generate_target()`` call time (not at
    ``EvalConfig`` construction) per the error-at-use-time pattern (D-08).
    Dict must contain at least one recognised augmentation key beyond the
    ``"type"`` discriminator (e.g. ``{"type": "rigid", "rotation_deg":
    30.0, "rotation_axis": [0, 0, 1]}`` or ``{"type": "noise", "sigma":
    0.1}``).  Default ``None``.
```
No `field_validator` exists for `transform_spec` contents (unlike `alignment_method`,
`swd_variant`, `label_transfer_method`, `device`, which each have a
`@field_validator` — see `eval/config.py:364-472` for that pattern if the planner decides
`type: "subsample_pair"` needs its own key-set validation at construction time; note this would
be a *departure* from the existing "validate at call time, not construction time" convention
used for `transform_spec` itself — D-08 in the docstring above). Docstring at line 202-211 should
be updated to mention `"subsample_pair"` as a third `"type"` value, purely as documentation (no
behavioral schema change, matching D-04's "no new EvalConfig field").

`pipeline_mode` field (`eval/config.py:288`, docstring `192-195`) is unchanged:
```python
pipeline_mode: Literal["paired", "synthetic"] = "paired"
```

---

### `eval/runners/eval_runner.py` — one dispatch branch (controller, request-response)

**Analog:** the existing `pipeline_mode` branch in `run()`

**Imports** (`eval/runners/eval_runner.py:44-61`):
```python
import json
from pathlib import Path
from typing import Any

from zreg.core.dataset import zRegPointCloud

import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.metrics import MetricsEngine
from eval.stages import AlignmentStage, LabelTransferStage
from eval.tracking import export_trajectory
from eval.types import AlignResult, EvalReport, LabelResult, StageMetrics
from eval.viz import plot_metrics, plot_trajectory
```

**Exact call site to modify** (`eval/runners/eval_runner.py:185-194`):
```python
Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)  # D-12

self.factory = DataFactory(self.config)
source = self.factory.load_real()
if self.config.pipeline_mode == "paired":
    target = self.factory.load_target()
else:  # pipeline_mode == "synthetic" — Phase 31 MODE-02
    target = self.factory.generate_target(source, self.config.transform_spec)

result = self._run_single(source, target, self.params)
```
The `type: "subsample_pair"` branch needs to nest inside the `else:` arm (line 191-192), e.g.
`if self.config.transform_spec.get("type") == "subsample_pair": ... else: target =
self.factory.generate_target(...)`. Note `source = self.factory.load_real()` (line 188) already
runs unconditionally **before** this branch — for `subsample_pair`'s "synthesize fresh geometry"
mode (D-02), the new branch would need to override/ignore this eagerly-loaded `source` rather than
being able to skip loading it, unless the new method is designed to accept and re-derive from it.

**Second call site — ground-truth extraction, same file** (`eval/runners/eval_runner.py:327-332`,
inside `_run_single`):
```python
# Ground truth — branches on pipeline_mode (Phase 31 MODE-03)
if self.config.pipeline_mode == "synthetic":
    gt = self.factory.get_synthetic_ground_truth()
else:
    gt = self.factory.get_ground_truth(source)  # {frame_key: id_tensor}
y_true = gt[source_sorted_keys[-1]]
```
This stays `pipeline_mode == "synthetic"` (D-04: no new `pipeline_mode`), so if the new method's
output is made compatible with `get_synthetic_ground_truth()` (per canonical_refs), this call site
needs **no** additional `type == "subsample_pair"` branch at all — it is the strongest argument
for making the new method populate `self._correspondence_idx`/`self._source_dataset` the same way
`generate_target()` does.

---

### `eval/runners/optimizer.py` — THREE call sites (controller/HPO, request-response + batch)

**Imports** (`eval/runners/optimizer.py:73-96`):
```python
import json
import logging
import os
from pathlib import Path
from typing import Any

from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_labels, generate_trajectory

import torch

import optuna

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from eval.metrics import MetricsEngine
from eval.search_strategies import BayesianSearch, GridSearch, PropulateSearch, RandomSearch, SobolSearch
from eval.stages import AlignmentStage, LabelTransferStage
from eval.types import SearchResult, StageMetrics, Trial
```

**Tier-gating module constants to follow for D-07** (`eval/runners/optimizer.py:100-102`, sibling
convention already flagged by CONTEXT.md's code_context section):
```python
SANITY_N_TRIALS: int = 5
DEV_N_TRIALS: int = 20
```
and the analogous gate-and-fallback shape used elsewhere in the codebase for a similar
budget-vs-correctness tradeoff (`eval/search_strategies.py:59,246-253`):
```python
SOBOL_MIN_TRIALS: int = 8
...
        # D-10: small budget fallback to RandomSearch
        if n_trials < SOBOL_MIN_TRIALS:
            _log.debug(
                "SobolSearch: n_trials=%d < 8, using RandomSearch fallback", n_trials
            )
            random.seed(seed)
            return RandomSearch().search(
                search_space, objective_fn, n_trials=n_trials, warm_start=warm_start
            )
```
D-07's "reject/warn when `len(seed) > 1` and `tier != 'full'`" gate should follow this exact
shape: a module-level named constant (if a threshold is needed) plus an early guard/`warnings.warn`
or `raise` at the top of whichever function reads `transform_spec["seed"]`.

**Scratch-factory isolation helper — directly relevant to D-07 seed-list averaging**
(`eval/runners/optimizer.py:105-136`, full function):
```python
def _apply_transform_to_dataset(
    dataset: dict,
    transform_spec: dict,
    config: "EvalConfig",
) -> dict:
    """Apply transform_spec to dataset using a scratch DataFactory (D-11).

    This helper is used by ``_objective`` for the sanity tier in synthetic mode.
    It MUST NOT touch the caller's ``_factory`` instance — using a scratch
    ``DataFactory`` avoids overwriting ``_synthetic_target`` on the main factory
    (Pitfall 3 from Phase 31 RESEARCH.md, D-11 from CONTEXT.md).
    ...
    """
    augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
    scratch_cfg = config.model_copy(update={"augmentation_params": augment_params})
    scratch_factory = DataFactory(scratch_cfg)
    return scratch_factory.augment(dataset)
```
If D-07's multi-seed averaging is implemented inside `_objective`, each of the N seeds' subsample
pairs must be generated the same isolated way (a scratch `DataFactory`, or equivalent), because
`self._factory._correspondence_idx`/`_synthetic_target` are single-slot instance attributes that
would otherwise be clobbered between seeds within the same trial.

**Call site 1 of 3 — `run()`, pre-populate branch** (`eval/runners/optimizer.py:223-237`):
```python
# Synthetic mode: pre-populate _synthetic_target so dev/full tiers can access it.
# Must be done before the tier loop because _objective reads _factory._synthetic_target
# directly for tier_name != "sanity" (it cannot call generate_target() per-trial
# without overwriting shared state — Pitfall 3 from Phase 31 RESEARCH.md).
#
# Gate on a non-sanity ceiling (D-11): when config.tier == "sanity" only the sanity
# tier runs, and it must use the isolated scratch factory (_apply_transform_to_dataset)
# — calling the main factory's generate_target() there would violate sanity isolation.
if (
    self.config.pipeline_mode == "synthetic"
    and self.config.transform_spec is not None
    and self.config.tier != "sanity"
):
    real_source = self._factory.load_real()
    self._factory.generate_target(real_source, self.config.transform_spec)
```
Needs a nested `if self.config.transform_spec.get("type") == "subsample_pair": ... else:
self._factory.generate_target(...)` — the new method call must populate whatever attribute(s)
call sites 2 and 3 below will read for the `subsample_pair` case.

**Call site 2 of 3 — `_objective()`, tier_target branch** (`eval/runners/optimizer.py:401-421`):
```python
# Phase 30/31 source/target dispatch — Pitfall 7 + CONTEXT D-03/D-10/D-11
# Phase 31 MODE-02/MODE-03: synthetic mode branches added here
if self.config.pipeline_mode == "synthetic":
    if tier_name == "sanity":
        # D-11: apply transform locally — NOT via self._factory.generate_target()
        # to avoid overwriting _synthetic_target (Pitfall 3)
        tier_target = _apply_transform_to_dataset(
            tier_dataset, self.config.transform_spec, self.config
        )
    else:  # dev / full
        # D-10: use pre-computed _synthetic_target, sliced to tier keys (Pitfall 7)
        tier_target = {
            k: self._factory._synthetic_target[k]
            for k in tier_dataset
            if k in self._factory._synthetic_target
        }
else:  # paired mode — existing code preserved
    if tier_name == "sanity":
        tier_target = tier_dataset  # Pitfall 7(a) — sanity reuses same dataset
    else:
        tier_target = self._factory.load_target()  # Pitfall 7(b) — dev/full call load_target
```
This is the site with the most branching already (tier × pipeline_mode); a `subsample_pair`
sub-branch nested inside `if self.config.pipeline_mode == "synthetic":` needs its own
sanity-vs-dev/full split mirroring the existing shape, likely re-deriving both source AND target
views per-tier (unlike today's single `tier_target` derivation) since `subsample_pair` produces
a paired (source, target) view rather than one target transformed from a fixed source.

**Call site 3 of 3 — `_objective()`, GT-selection branch** (`eval/runners/optimizer.py:443-464`):
```python
# GT selection — branches on pipeline_mode (Phase 31 MODE-03, D-08/D-09)
if self.config.pipeline_mode == "synthetic":
    if tier_name == "sanity":
        # Sanity toy dataset has labels in pc["label"] (generate_labels contract,
        # Pitfall 5 from Phase 31 RESEARCH.md). get_synthetic_ground_truth() reads
        # _source_dataset (the real dataset), not the toy dataset — use label directly.
        sample_pc = tier_dataset[source_sorted_keys[0]]
        if sample_pc["label"] is not None:
            y_true = tier_dataset[source_sorted_keys[-1]]["label"]
        else:
            n = tier_dataset[source_sorted_keys[-1]]["pos"].shape[0]
            y_true = torch.arange(n, dtype=torch.long)
    else:  # dev / full in synthetic mode — D-09
        y_true = self._factory.get_synthetic_ground_truth()[source_sorted_keys[-1]]
else:
    ...
```
If the new method is made `get_synthetic_ground_truth()`-compatible (recommended, see
`eval_runner.py` section above), this call site needs **no** `type == "subsample_pair"` branch
either — same argument as `eval_runner.py`'s GT call site.

## Shared Patterns

### Config-driven conditional branching (the real "dispatch" pattern to copy)
**Source:** `eval/runners/eval_runner.py:189-192`, `eval/runners/optimizer.py:231-237,403-421,444-456`
**Apply to:** all 4 integration points
```python
if self.config.pipeline_mode == "synthetic":
    ...
else:
    ...
```
This `if/else` on a `self.config.<field>` value, with an inline comment citing the deciding
CONTEXT.md decision (`# D-04`, `# D-11`, etc.), is the established style — copy this shape for
every new `transform_spec.get("type") == "subsample_pair"` check, not a lookup-table/dispatch-dict
style (none exists in this codebase for `transform_spec`).

### Correspondence tracking (Phase 56 machinery, reuse don't reimplement)
**Source:** `eval/data_factory.py:961-1025` (`drop_points`), `1027-1104` (`sample_new_points`),
`678-767` (`get_synthetic_ground_truth`)
**Apply to:** `eval/data_factory.py` new method
Single-slot `self._correspondence_idx: dict[int, torch.Tensor] | None` — when generating two
independent views (source + target) from one base dataset, reset `self._correspondence_idx = None`
immediately **before EACH** `drop_points()`/`sample_new_points()` call (the method COMPOSES with
any prior non-`None` state instead of starting fresh, which raises `IndexError` on a second
un-reset call — see Analog 3 above), and capture the resulting tensor into a local variable
immediately **after** each call, since the attribute is reassigned by every call.

### Try/finally instance-state save-restore
**Source:** `eval/data_factory.py:334-341` (`generate_target`)
**Apply to:** `eval/data_factory.py` new method, if it also temporarily overrides
`self.config.augmentation_params` to call into `augment()`/`generate_target()` for the
`run_alignment=True` transform-layering half of D-03.

### Tier-based budget/correctness gating
**Source:** `eval/search_strategies.py:59,246-253` (`SOBOL_MIN_TRIALS`),
`eval/runners/optimizer.py:100-102` (`SANITY_N_TRIALS`, `DEV_N_TRIALS`)
**Apply to:** D-07's multi-seed-averaging tier gate (module-level constant + early guard,
`_log.debug`/`warnings.warn` message, no dispatch table).

### Frozen-model construction (if a `SubsamplePairResult`-style typed return is chosen)
**Source:** `eval/types.py:356-390` (`TrainingTriple`)
```python
class TrainingTriple(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    source_cloud: zRegPointCloud
    target_cloud: zRegPointCloud
    seed: int
```
If the planner's discretion (exact new-method signature) leads to a typed return value rather than
a plain `dict[int, zRegPointCloud]` tuple, this is the exact frozen-model convention
(`ConfigDict(frozen=True, arbitrary_types_allowed=True)`) already used for the same
"(source, target) pair + seed" shape in this codebase.

## No Analog Found

None — all 4 touched files already contain in-file sibling patterns strong enough to copy
directly (see above). No file requires reaching into RESEARCH.md-style external patterns (and no
RESEARCH.md exists for this phase; research was explicitly skipped per the orchestrator's framing).

## Metadata

**Analog search scope:** `eval/data_factory.py`, `eval/config.py`, `eval/runners/eval_runner.py`,
`eval/runners/optimizer.py`, `eval/search_strategies.py`, `eval/types.py`
**Files scanned:** 6 (4 target files + 2 cross-referenced for the tier-gating and frozen-model
conventions)
**Verification method:** every quoted excerpt was read directly via the `Read` tool at the cited
line numbers in this pass (no re-reads of already-loaded ranges); the "no `type` dispatch exists"
claim was additionally verified via `grep -rn` across `eval/` with zero matches, run in this
session.
**Pattern extraction date:** 2026-08-04
