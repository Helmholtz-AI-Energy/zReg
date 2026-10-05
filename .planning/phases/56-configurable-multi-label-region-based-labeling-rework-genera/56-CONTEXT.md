# Phase 56: Configurable multi-label region-based labeling - Context

**Gathered:** 2026-07-31
**Status:** Ready for planning
**Source:** In-chat design discussion (evaluation of generate_labels() against user's vision, followed by structured clarifying questions)

<domain>
## Phase Boundary

Rework `generate_labels()` in `src/zreg/data_generation/labels.py` so it becomes the single, config-driven entry point for labeling a point cloud trajectory: any number of labels (`n_labels`, renamed from `n_classes`), each defined by one or more region components (voronoi / gaussian blob / gaussian cone), assigned either deterministically (argmax) or probabilistically (sampled), with region centers fixed once per trajectory so label change is spatially traceable rather than random per-frame noise. `assign_cap_labels`/`assign_gaussian_labels` (Phase 55) are deleted entirely and absorbed as the cone shape's special case. Config wiring (new `EvalConfig` field) is in scope so scenario YAML can declare a label spec instead of hardcoded Python literals.

Explicitly out of scope for this phase: animated/moving region centers over the trajectory (interpolated/per-frame keyframes) — deferred to a future phase. Anisotropic/full-covariance blobs and off-origin/arbitrary-apex cones are also deferred — see Claude's Discretion below for the locked simpler forms.

</domain>

<decisions>
## Implementation Decisions

### Investigation findings that motivate this phase
- **D-01:** `generate_labels()` currently redraws Voronoi seed points with `torch.randn` *inside* the per-frame loop (`src/zreg/data_generation/labels.py` lines ~76-81) — every frame gets a fresh, uncorrelated random partition. This is the root cause of labels having zero temporal correlation across a trajectory and must be fixed as part of this rework (see D-07).
- **D-02:** `generate_trajectory()` itself draws each frame as an independent fresh `N(0,I)` sample with no persistent per-point identity — this is a separate, out-of-scope limitation of the base synthetic generator, not something this phase needs to fix. This phase's fix (D-07) makes region *centers* stable in space; it does not and cannot make individual point identity traceable through `generate_trajectory()`'s output. Real/semi-synthetic trajectories (from `DataFactory`'s augmentation pipeline, or real tracked-cell datasets) already carry per-point correspondence, and static centers will produce genuinely traceable per-point label changes there.
- **D-03:** No existing code path wires label generation to config at all today — `generate_labels`/`assign_cap_labels`/`assign_gaussian_labels` are called with hardcoded literal args directly in Python (`eval/data_factory.py`, `eval/runners/optimizer.py`). This phase introduces the first config-driven path.

### Naming and API surface
- **D-04:** Rename `n_classes` → `n_labels` everywhere in the label-generation surface: `generate_labels`'s own parameter, `DataFactory.generate_training_triple`'s `n_classes` param ([eval/data_factory.py:344](eval/data_factory.py:344)), and the literal call site in [eval/runners/optimizer.py:521](eval/runners/optimizer.py:521).
- **D-05:** `generate_labels()` keeps a simple path — pass `n_labels: int` for auto-generated random Voronoi components (drop-in replacement for today's usage) — alongside a full path — pass explicit `label_specs: list[LabelSpec]` for custom shapes/multi-region/multi-label control. Exactly one of `n_labels`/`label_specs` is required.
- **D-06 (Phase 55 removal):** Delete `assign_cap_labels`, `assign_gaussian_labels`, and their ~13 tests entirely. Do not keep back-compat wrappers or deprecated shims — this is dead-clean removal, not a soft deprecation. Cone shape (D-09) absorbs their functionality.

### Region/component model
- **D-07 (critical bug fix):** Region component centers (for all shapes: voronoi/blob/cone) MUST be sampled/resolved once per `generate_labels()` call, before the per-frame loop, then reused unchanged across every frame of the trajectory. Centers are static in space for the whole trajectory — no per-frame animation or interpolation in this phase (explicitly deferred).
- **D-08 (blob shape):** Isotropic sigma only — a single scalar Euclidean sigma per blob component. No anisotropic/full covariance matrix, no per-axis diagonal covariance.
- **D-09 (cone shape):** Origin-anchored pole direction + `sigma_deg`, exactly matching the geometry `assign_cap_labels`/`assign_gaussian_labels` already used (reuse the existing `_angular_distance_deg` helper math, generalized to be called per cone component rather than hardcoded to a single pole). No arbitrary apex position in this phase.
- **D-10 (voronoi shape):** Deterministic mode = nearest-center by Euclidean distance (unchanged from current behavior). Probabilistic mode = softmax over negative squared distance with a required temperature parameter (voronoi has no natural probability scale on its own, unlike blob/cone which have `sigma`).
- **D-11 (mixture within a label):** A label may have multiple components (mix of shapes allowed, e.g. one label = one blob + one cone). Combine via `logsumexp` of (weighted) component scores — a `weight` field on each component acts as the mixing coefficient/prior.
- **D-12 (assignment mode is a top-level switch):** `mode: "deterministic" | "probabilistic"` applies uniformly across all labels in one `generate_labels()` call — argmax over per-label mixture scores (deterministic) or softmax-over-labels + `torch.multinomial` categorical sample per point (probabilistic).

### Config wiring
- **D-13:** Add a new `EvalConfig` field (e.g. `label_generation`) holding the label spec (list of `LabelSpec`, each with `components: list[LabelComponentSpec]`), plus `mode` and `seed`, so scenario YAML can declare labeling declaratively. Wire it into `optimizer.py`'s sanity tier and `DataFactory.generate_training_triple`, replacing their current hardcoded `n_classes=4`/`6` literals.

### Suggested function decomposition (non-binding — planner/executor may refine)
- `LabelComponentSpec`/`LabelSpec` pydantic models (fields: `shape: Literal["voronoi","blob","cone"]`, `center`, `sigma`, `weight`, `label_id`)
- `_component_score(pos, component) -> Tensor(N,)` — per-shape score function
- `_label_scores(pos, label_spec) -> Tensor(N,)` — mixture-within-label aggregation via `logsumexp`
- `_assign_deterministic(scores) -> Tensor(N,)` — argmax over labels
- `_assign_probabilistic(scores, seed) -> Tensor(N,)` — softmax + `torch.multinomial` per point
- `generate_labels(trajectory, n_labels=None, label_specs=None, mode="deterministic", seed=42) -> trajectory` — public orchestrator

### Suggested plan breakdown (non-binding — planner decides final split)
1. Core `LabelComponentSpec`/`LabelSpec` + per-shape scoring (`_component_score` for voronoi/blob/cone) + mixture math (`_label_scores`)
2. Assignment modes (`_assign_deterministic`/`_assign_probabilistic`) + reworked `generate_labels()` orchestrator (simple + full path, once-per-trajectory center fix) + deletion of Phase 55 functions and tests
3. `EvalConfig.label_generation` field + validation + call-site wiring/rename (`optimizer.py`, `DataFactory.generate_training_triple`) + example scenario YAML
4. Comprehensive tests: per-shape correctness, deterministic-vs-probabilistic mode behavior, and an explicit regression test proving labels are temporally correlated across frames when centers are static (this is the test that proves D-07's bug fix)

### Claude's Discretion
- Exact module layout: whether the new pydantic models/scoring functions live in a new `label_spec.py` or stay inside `labels.py`.
- Exact `EvalConfig` field/model naming (`label_generation` is a working name, not locked).
- Whether `weight` needs formal normalization (softmax-normalized priors) or is just a relative multiplier into the mixture — resolve during Plan 1 implementation.
- Final plan count/split (the 4-plan breakdown above is a starting suggestion, not a requirement).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Current implementation being replaced
- `src/zreg/data_generation/labels.py` — current `generate_labels`, `remove_labels`, `assign_cap_labels`, `assign_gaussian_labels`, `_angular_distance_deg` (lines 25-256)
- `src/zreg/data_generation/__init__.py` — current label-function exports
- `tests/test_generators.py` — `TestSphericalLabelGenerators` (13 tests to be deleted along with `assign_cap_labels`/`assign_gaussian_labels`)

### Call sites requiring rename/rewiring
- `eval/data_factory.py` lines ~330-430 (`DataFactory.generate_training_triple`, its `n_classes` param, and its `generate_labels(base, n_classes=n_classes, seed=seed)` call at line 420)
- `eval/runners/optimizer.py` lines ~61, 83, 435, 447, 515, 521 (`generate_labels(traj, n_classes=4, seed=42)` sanity-tier call and its surrounding comments referencing the `generate_labels` contract)
- `eval/config.py` (`EvalConfig` class, ~line 98) — where the new `label_generation` field is added; note existing `label_names: dict[int, str] | None` field (line 274) is a display-name mapping, unrelated to this new field but likely referenced together in docs

### Prior phase for context (not to be re-read in full, but informs the removal)
- `.planning/phases/55-spherical-cap-and-gaussian-label-generators-for-zreg-data-ge/55-01-SUMMARY.md` — documents exactly what Phase 55 built and why (for understanding what's being removed and why cone absorbs it)

No external specs/ADRs — requirements fully captured in decisions above. This phase has no formal REQUIREMENTS.md entries (consistent with Phase 55's "Requirements: none mapped" precedent — REQUIREMENTS.md is scoped to the separate v1.6 HoreKa Cluster milestone).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_angular_distance_deg(pos, pole)` in `labels.py` (lines 111-138) — pole-normalization, eps-guarded norm, clamped cosine, `rad2deg(acos(...))` — this math is correct and should be reused/generalized for the cone shape's per-component scoring, not rewritten.
- The deep-copy contract (`copy.deepcopy(trajectory)` at the top of every label function, D-03 project-wide convention) must be preserved in the reworked `generate_labels()`.

### Established Patterns
- All label tensors are `torch.long`, shape `(N,)`, stored in `pc["label"]` — this output contract is unchanged by the rework.
- `seed: int | None` contract: `int` calls `torch.manual_seed` once at function entry; `None` means caller controls RNG state. Preserve this for the reworked `generate_labels()`.
- Empty frames (`N == 0`) are handled transparently by tensor ops without special-casing — preserve this.

### Integration Points
- `DataFactory.generate_training_triple` (single-frame base cloud → `generate_labels` → `generate_target` which propagates `label` through transforms) — the rename and new API must not break this label-then-transform pipeline.
- `optimizer.py`'s sanity tier (`generate_labels(generate_trajectory(...))`, multi-frame) — this is the actual trajectory-labeling call site where D-07's center-stability fix will be directly observable/testable.

</code_context>

<specifics>
## Specific Ideas

- User's original vision, verbatim intent: "I want the functionality of generate_labels to include setting labels for a whole point cloud trajectory, in which the labels change over the course of the trajectory. The way one individual point cloud should be labeled should be defined in the config... labeled using gaussian cones or blobs and then choosing whether the points should be labeled in a deterministic way (what gaussian center of which label is closest) or in a probabilistic way (which label is the point most likely to have with the gaussians for the labels). It should be possible to define one or multiple centers/distributions and regions for each label and it should be possible to choose the number of labels freely."
- Confirmed via investigation: today neither axis (distribution shape x assignment mode) is implemented as a real matrix — see D-01/D-02/D-03 above and the deleted functions' narrow binary-only scope.

</specifics>

<deferred>
## Deferred Ideas

- **Animated/moving region centers** (per-frame or interpolated keyframe positions) — explicitly deferred; this phase locks centers as static-per-trajectory (D-07). A future phase could add this on top once the static-center system is proven.
- **Anisotropic/full-covariance blobs** — deferred; isotropic sigma only for now (D-08).
- **Arbitrary-apex cones** (not anchored at the origin) — deferred; origin-anchored pole only for now (D-09).
- **Fixing `generate_trajectory()`'s lack of per-point identity** — out of scope for this phase entirely (D-02); this phase only fixes region-center stability, not the base synthetic generator's point correspondence.

None — discussion stayed within phase scope otherwise.

</deferred>

---

*Phase: 56-configurable-multi-label-region-based-labeling-rework-genera*
*Context gathered: 2026-07-31*
