# Phase 53: GPU Acceleration - Context

**Gathered:** 2026-07-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 53 threads a configurable `device` field through `EvalConfig` → `DataFactory` → `AlignmentStage`/CPD so all tensor operations execute on the configured device. Three concrete deliverables:

1. **`EvalConfig.device`** — a validated string field controlling where tensors are loaded and computed (GPU-01)
2. **`DataFactory` device threading** — replaces 6 hardcoded `device="cpu"` calls with `self.config.device`; raises loudly at `__init__` if the requested device isn't available (GPU-02)
3. **Verification** — cluster configs updated with `device: "cuda"`; DataFactory logs the resolved device so operators can confirm GPU tensors in SLURM output (GPU-03)

No changes to CPD internals — CPD is already device-transparent via tensor operations. No changes to DTW, metrics, or visualisation.

</domain>

<decisions>
## Implementation Decisions

### Device Field Shape (GPU-01)
- **D-01:** `EvalConfig.device` is a `str` field with default `"cpu"`. Accepted values: `{"cpu", "cuda", "cuda:0", "cuda:1", "mps"}`. Validated via `@field_validator` at EvalConfig construction (YAML load time), consistent with `alignment_method`, `label_transfer_method`, and `swd_variant` validators already in `eval/config.py`.
- **D-02:** Unknown device strings (e.g., `"gpu"`, `"cuda:7"`) are rejected immediately with a clear `EvalConfigError` message — not passed through to PyTorch where the error would be delayed and opaque.

### CUDA Unavailability Behavior (GPU-02)
- **D-03:** `DataFactory.__init__` checks device availability immediately after construction: if `config.device != "cpu"` and the requested backend is unavailable (`torch.cuda.is_available()` for `"cuda"` / `"cuda:N"`; `torch.backends.mps.is_available()` for `"mps"`), raise `RuntimeError` with a message naming the configured device and confirming the backend is not available. Fails fast before any disk I/O or MPI rank setup.
- **D-04:** `dataset.py`'s existing silent CPU fallback (lines 118-120) is never reached — DataFactory's `__init__` raise intercepts the unavailable-device case before any loader is called. Do not modify `dataset.py`.

### ICP + GPU Incompatibility Guard (GPU-03)
- **D-05:** `EvalConfig` gains a `@model_validator(mode='after')` that raises `EvalConfigError` when `device != "cpu"` and `alignment_method == "icp"`. Error message: `"device='{device}' is incompatible with alignment_method='icp': Open3D ICP requires CPU-resident tensors (explicit .cpu().numpy() round-trips in icp.py:114-115, 196-197)"`. Mirrors the existing `validate_swd_variant` cross-field pattern.
- **D-06:** ICP's CPU round-trips are an explicit architectural constraint of Open3D, not a silent fallback. No changes to `icp.py`. The model_validator documents the constraint at parse time rather than burying it in code comments.

### Verification Artifacts (GPU-03)
- **D-07:** `DataFactory.load_real()` (and `load_target()`) logs the resolved device after the dataset is loaded: `logger.info("Loaded %s dataset on %s (%d frames)", 'source'/'target', self.config.device, len(dataset))`. Runs inside `srun` on compute nodes — appears in SLURM output for operator inspection.
- **D-08:** Cluster configs in `baseline_experiments/configs_horeka/` (all 7 files) are updated to include `device: "cuda"`. This is the mechanism that activates GPU tensors on HoreKa — no SLURM script changes needed beyond the config update.
- **D-09:** The existing `[VERIFY ON HOREKA]` convention (from Phase 51/52) is extended to a comment in the relevant sbatch: `# [VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output to confirm GPU tensors`.

### Claude's Discretion
- Logger name / level inside DataFactory (use existing logging setup in the file; `INFO` level)
- Whether to add `device` to the `__repr__` / `__str__` of DataFactory (probably not needed — the log line is sufficient)
- Exact wording of `EvalConfigError` messages (be specific: include the configured value and what was expected)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Implementation Spec
- `baseline_experiments/HOREKA_PLAN.md` §"Phase 2 — Real GPU acceleration" (§2.1–2.4) — MUST READ: the authoritative implementation spec for this phase. Covers `device` field addition (§2.1), DataFactory threading (§2.2), downstream stage device-consistency (§2.3), and Open3D CUDA caveat (§2.4).

### Files to Modify
- `eval/config.py` — EvalConfig: add `device` field (D-01/D-02) + `@model_validator` ICP guard (D-05). Read in full before editing — understand existing `@field_validator` patterns for `alignment_method`, `label_transfer_method`, `swd_variant`.
- `eval/data_factory.py` — DataFactory: add `__init__` CUDA availability check (D-03) + replace 6 hardcoded `device="cpu"` calls (lines 158, 161, 224, 227, 640, 642) with `self.config.device` (D-04) + add device logging (D-07).
- `baseline_experiments/configs_horeka/*.yaml` — all 7 cluster configs: add `device: "cuda"` (D-08).

### Reference Files (read-only)
- `src/zreg/dataset.py` lines 55–162 — `load_data_from_tracklets`: already accepts `device` param; silent CPU fallback at lines 118-120 (do NOT modify — DataFactory.__init__ guard makes it unreachable).
- `src/zreg/cpd/base.py` — CPD base: stores `use_cuda` but device follows input tensor; no changes needed.
- `src/zreg/registration/icp.py` lines 114-115, 196-197 — explicit `.cpu().numpy()` round-trips; reason for D-05 model_validator guard.
- `eval/stages/alignment.py` lines 440-441 — `"device": matched_source_frame["pos"].device` shows CPD already follows tensor device; no changes needed.

### Requirements
- `.planning/REQUIREMENTS.md` §"GPU Acceleration" — GPU-01 (EvalConfig.device), GPU-02 (DataFactory device loading), GPU-03 (verified GPU execution, no silent fallback)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/config.py` `@field_validator` pattern (used by `alignment_method`, `label_transfer_method`, `swd_variant`) — copy the validator structure for `device` field (D-01)
- `eval/config.py` `@model_validator(mode='after')` pattern (used by `validate_swd_variant` cross-field check when `alignment_method='swd'`) — copy this pattern for the ICP guard (D-05)
- Existing logging setup in `eval/data_factory.py` — use whatever `logger` is already established there (D-07)

### Established Patterns
- **Error-at-use-time** (Phase 30, target_data_path): used for optional fields that are None by default. `device` is always-present (default "cpu"), so `@field_validator` at construction is correct — not error-at-use-time.
- **Defensive import with fallback** (Phase 52, MPI): `try: from mpi4py import MPI; RANK = ... except ImportError: RANK = 0`. The CUDA availability check at DataFactory.__init__ is analogous — check early, fail loud if configured device isn't available.
- **`extra="forbid"` on EvalConfig** — unknown YAML keys already raise EvalConfigError. Adding `device` as a new field is backward-compatible: existing configs without `device:` will use the `"cpu"` default.

### Integration Points
- `DataFactory.__init__` (line ~50 area) — where the CUDA availability check is inserted (D-03)
- `DataFactory.load_real()` lines 158, 161 — two of the 6 `device="cpu"` → `self.config.device` substitutions (D-04)
- `DataFactory.load_target()` lines 224, 227 — two more substitutions
- `DataFactory.get_ground_truth()` lines 640, 642 — remaining two substitutions
- `AlignmentStage` — no changes; CPD path uses `device=cloud["pos"].device` which follows tensor device automatically once data loads onto GPU

</code_context>

<specifics>
## Specific Ideas

- The `@model_validator` ICP guard error message should reference the specific source lines (`icp.py:114-115, 196-197`) — this is unusually specific but correct here because it answers the "why can't I use ICP on GPU?" question without making the user dig through code.
- `[VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output` as the concrete verification instruction (D-09).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 53-gpu-acceleration*
*Context gathered: 2026-07-15*
