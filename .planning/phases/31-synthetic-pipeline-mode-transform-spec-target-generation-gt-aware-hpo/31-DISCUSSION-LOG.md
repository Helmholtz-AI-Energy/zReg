# Phase 31: Synthetic Pipeline Mode — Transform-Spec Target Generation & GT-Aware HPO - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 31-synthetic-pipeline-mode-transform-spec-target-generation-gt-aware-hpo
**Areas discussed:** generate_target() strategy, GT label semantics, _objective() synthetic mode

---

## generate_target() strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Delegate to augment() | Reuse existing dispatch + deep-copy semantics via temp augmentation_params override | ✓ |
| Call generators directly | Call apply_rigid()/add_gaussian_noise() from src/zreg/generators/ directly | |

**User's choice:** Delegate to augment()

| Option | Description | Selected |
|--------|-------------|----------|
| Temp override via augmentation_params | Save/restore self.config.augmentation_params around augment() call | ✓ |
| Build a throw-away DataFactory | Construct a temp DataFactory with transform_spec as augmentation_params | |
| Extract augment() logic to a free function | Refactor augment() to accept explicit params | |

**User's choice:** Temp override via augmentation_params

| Option | Description | Selected |
|--------|-------------|----------|
| Store result as instance state | self._synthetic_target + self._source_dataset; RuntimeError guard = _synthetic_target is None | ✓ |
| Recompute on demand | Re-apply transform from stored spec + source on each get_synthetic_ground_truth() call | |

**User's choice:** Store result as instance state

---

## GT label semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Ordinal indices 0..N-1 | torch.arange(n_points, dtype=torch.long) per frame when pc["id"] is None | ✓ |
| Raise RuntimeError | Treat missing source labels as an error | |
| Return None per frame | Signal unavailability; caller handles None | |

**User's choice:** Ordinal indices 0..N-1

| Option | Description | Selected |
|--------|-------------|----------|
| Always torch.long | Consistent with compute_f1 regardless of source dtype | ✓ |
| Match source pc["id"] dtype | Preserves original label dtype; risky if float | |

**User's choice:** Always torch.long

| Option | Description | Selected |
|--------|-------------|----------|
| Just return the label dict | No metadata; transform_spec recoverable from EvalConfig | |
| Store transform_spec as metadata | self._transform_spec stored on DataFactory after generate_target() | ✓ |

**User's choice:** Store transform_spec as metadata
**Notes:** User chose to store transform_spec on the DataFactory instance for debugging/reproducibility.

---

## _objective() synthetic mode

| Option | Description | Selected |
|--------|-------------|----------|
| Check self._config.pipeline_mode | Branch on pipeline_mode == "synthetic" — direct and readable | ✓ |
| Check self._factory._synthetic_target is not None | Implicit detection via DataFactory state | |

**User's choice:** Check self._config.pipeline_mode

**Notes:** User requested elaboration on the "which frame's GT" question — explained that `_tier_dataset` generates three different datasets (sanity = 3-frame toy, dev = real data, full = real data) and that GT needs to be sliced to whichever frame the tier is using.

| Option | Description | Selected |
|--------|-------------|----------|
| Last frame of tier_dataset | y_true = get_synthetic_ground_truth()[tier_sorted_keys[-1]] — mirrors paired mode structure | ✓ |
| Average F1 across all tier frames | Loop + average per tier frame — more representative but structurally diverges | |

**User's choice:** Last frame of tier_dataset

**Notes:** User requested elaboration on sanity tier behavior — explained that sanity tier generates its own 3-frame toy dataset unrelated to the real source. User then chose to have sanity tier also apply the transform to its toy dataset in synthetic mode (rather than keep self-alignment).

| Option | Description | Selected |
|--------|-------------|----------|
| Keep sanity unchanged | Sanity self-aligns toy dataset regardless of pipeline_mode | |
| Apply transform to sanity toy dataset too | Sanity also exercises synthetic transform path in synthetic mode | ✓ |

**User's choice:** Apply transform to sanity toy dataset too
**Notes:** Implementation must NOT call DataFactory.generate_target() for sanity (would overwrite _synthetic_target from real data). Planner must isolate sanity's transform application — e.g., stateless helper or scratch config override.

---

## Claude's Discretion

None — user made explicit decisions on all presented options.

## Deferred Ideas

None — discussion stayed within phase scope.
