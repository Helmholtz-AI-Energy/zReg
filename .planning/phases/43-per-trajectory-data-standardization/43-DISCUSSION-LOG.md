# Phase 43: Per-Trajectory Data Standardization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-30
**Phase:** 43-per-trajectory-data-standardization
**Areas discussed:** Default on vs opt-in, Source-target coupling, Robust scaler semantics, Which fields get scaled

---

## Default on vs opt-in

| Option | Description | Selected |
|--------|-------------|----------|
| Opt-in only (None by default) | `data_preprocessing: ... \| None = None` — same as Phase 41. Existing configs untouched. | |
| On by default (standardize) | Defaults to `DataPreprocessingConfig(method="standardize")`. Existing configs silently inherit z-score. | ✓ |
| On by default but disable-able | Defaults to standardize; users opt out with `data_preprocessing: null`. | |

**User's choice:** On by default — standardize applied to all configs including existing ones.

**Follow-up — existing configs:**

| Option | Description | Selected |
|--------|-------------|----------|
| Intentional — let them inherit standardize | No changes to configs/*.yaml needed. | ✓ |
| Preserve old behavior — add null to existing configs | Explicitly opt out existing configs. | |

**User's choice:** Intentional — existing scenario configs inherit standardize as the new default.

**Notes:** This is a behavioral change for all existing runs. Accepted intentionally per DATA-02-01 ("standardization as default").

---

## Source-target coupling

| Option | Description | Selected |
|--------|-------------|----------|
| Independent (each uses own stats) | `load_real()` and `load_target()` each compute their own mean/std. | |
| Target uses source stats | `load_real()` computes and caches stats; `load_target()` reuses them. | ✓ |
| You decide | Leave to planner. | |

**User's choice:** Target uses source statistics.

**Follow-up — storage mechanism:**

| Option | Description | Selected |
|--------|-------------|----------|
| Private attribute on DataFactory | `self._preprocessing_stats: dict \| None` — follows existing cache pattern. | ✓ |
| Returned alongside dataset | `load_real()` returns `(dataset, stats)` tuple — changes public API. | |

**User's choice:** Private attribute `self._preprocessing_stats` on DataFactory.

**Notes:** If `load_target()` is called before `load_real()` (no cached stats), it computes its own — edge case handled gracefully.

---

## Robust scaler semantics

**Initial clarification:** User asked which approach is cleaner. Claude recommended Option A (separate `method: robust`) because:
- `robust_outlier_threshold` logically belongs to the robust method, not to z-score
- Follows `AlignmentPreprocessingConfig` precedent: `method` selects algorithm, other fields are method-specific
- Avoids `robust_outlier_threshold` being silently ignored on non-robust methods

| Option | Description | Selected |
|--------|-------------|----------|
| Separate method: median + IQR | `method: robust` uses median+IQR. `robust_outlier_threshold` clips at ±N×IQR. | ✓ (cleaner) |
| Outlier clip on top of z-score | `robust_outlier_threshold` modifies `method: standardize`. | |

**User's choice:** Separate `method: robust` with `robust_outlier_threshold` as clip threshold.

**Follow-up — clip semantics:**

| Option | Description | Selected |
|--------|-------------|----------|
| Clip at median ± N×IQR | After median+IQR scaling, clip values outside ±robust_outlier_threshold. | ✓ |
| Standard sklearn RobustScaler (no clip) | Subtract median / divide by IQR only; robust_outlier_threshold unused. | |
| You decide | Leave to planner. | |

**User's choice:** Clip at median ±N×IQR after scaling.

**Notes:** DATA-02-04 example showed `method: standardize, robust_outlier_threshold: 3` in the same dict — this was interpreted as a documentation shortcut. In code, `robust_outlier_threshold` belongs to `method: robust` only.

---

## Which fields get scaled

| Option | Description | Selected |
|--------|-------------|----------|
| Only 'pos' | Standardize only 3D spatial coordinates. Labels, ids, fps-idx untouched. | ✓ |
| All numeric tensors | Apply to every torch.Tensor. Risk: corrupts label IDs and indices. | |
| You decide | Leave to planner. | |

**User's choice:** Only `pos`.

**Follow-up — per-trajectory scope:**

| Option | Description | Selected |
|--------|-------------|----------|
| Global trajectory stats | Concatenate all frames' pos tensors; compute one mean/std per XYZ across all frames. | ✓ |
| Per-frame independently | Each frame gets its own mean/std; destroys inter-frame scale relationships. | |

**User's choice:** Global trajectory statistics (cross-frame concatenation).

**Notes:** "Per-trajectory, not per-dataset" from DATA-02-03 confirmed to mean: each trajectory (source or target) gets its own global statistics computed across all its frames.

---

## Claude's Discretion

- Whether `DataPreprocessingConfig` lives in `eval/config.py` alongside `AlignmentPreprocessingConfig` or in a separate module
- Whether to log preprocessing method and computed stats at `logging.debug`
- Numeric stability guard for zero-std / zero-IQR (eps=1e-8 recommended over ValueError)

## Deferred Ideas

None — discussion stayed within phase scope.
