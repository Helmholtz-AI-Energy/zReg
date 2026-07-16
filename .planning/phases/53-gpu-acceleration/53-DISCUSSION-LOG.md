# Phase 53: GPU Acceleration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-15
**Phase:** 53-gpu-acceleration
**Areas discussed:** Device field shape, CUDA unavailability behavior, ICP scope, Verification artifact (GPU-03)

---

## Device Field Shape

| Option | Description | Selected |
|--------|-------------|----------|
| str, limited set | `{"cpu", "cuda", "cuda:0", "cuda:1", "mps"}` — validated via `@field_validator` at EvalConfig construction | ✓ |
| str, pass-through | Accept any string, fail at tensor creation time | |
| str, 'cpu' and 'cuda' only | Simpler validator, drops MPS | |
| You decide | Claude picks | |

**User's choice:** str, limited set

**Sub-question — where should validation happen?**

| Option | Description | Selected |
|--------|-------------|----------|
| At EvalConfig construction (field_validator) | Immediate rejection at YAML load, consistent with existing validators | ✓ |
| At DataFactory.load_real() call time | Error-at-use-time pattern | |
| Both — validator + CUDA-availability check at load | Two-stage | |

**User's choice:** field_validator at construction (after requesting Claude weigh the options).

**Notes:** User asked Claude to weigh the options. Claude recommended field_validator because `device` is an always-present field (not optional like `target_data_path`), and the error-at-use-time pattern exists to handle optional fields where the mode context isn't known at parse time. Consistent with `alignment_method`, `label_transfer_method`, `swd_variant` validators.

---

## CUDA Unavailability Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| RuntimeError in DataFactory.__init__ | Raises before any I/O if device unavailable | ✓ |
| Warning + silent CPU fallback | Logs warning, continues on CPU | |
| Leave dataset.py silent fallback as-is | No extra code | |

**User's choice:** RuntimeError in DataFactory.__init__

**Sub-question — where exactly does the check live?**

| Option | Description | Selected |
|--------|-------------|----------|
| DataFactory.__init__ | Check once at construction, fails fast | ✓ |
| DataFactory.load_real() only | Later, after construction | |
| Each of 6 call sites | Most defensive, most repetitive | |

**User's choice:** DataFactory.__init__

---

## ICP Scope

| Option | Description | Selected |
|--------|-------------|----------|
| CPD-only, document ICP as known CPU layer | No validation, cluster configs use cpd anyway | |
| Raise at EvalConfig parse (loudest) | @model_validator raises when device != cpu + alignment_method == icp | ✓ |
| Warn not raise | Warning at AlignmentStage construction | |

**User's choice:** Raise at EvalConfig parse (after requesting Claude weigh the options and explain which fails loudest).

**Notes:** User asked which option "fails loudest." Claude explained Option 2 fails loudest — fires at YAML load time via `@model_validator(mode='after')`, before any computation or disk I/O. Option 1 doesn't fail at all (silent). Option 3 warns late (AlignmentStage construction, which runs inside the job). Claude noted Option 2 is also clean to implement — mirrors the existing `validate_swd_variant` cross-field pattern.

---

## Verification Artifact (GPU-03)

**First question — verification form:**

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone verify_gpu_path.py script | Separate script asserting tensor devices | |
| Inline assertions in existing smoke sbatch | Piggyback on Phase 52 test job | ✓ |
| DataFactory logging only (passive) | Logger line, operator checks SLURM output | |

**User's choice:** Inline assertions in existing smoke sbatch

**Second question — assertion form (user asked Claude to weigh options):**

| Option | Description | Selected |
|--------|-------------|----------|
| Python one-liner pre-flight | Runs outside srun — login node has no GPU, would fail for wrong reason | |
| torch.cuda check only | Redundant with DataFactory.__init__ raise already locked | |
| DataFactory device logging | logger.info line, runs inside srun, appears in SLURM output | ✓ |

**User's choice:** DataFactory device logging (Option 3)

**Notes:** User asked "does the first option already use GPU time?" — Claude confirmed it would but negligibly. Deeper issue: the one-liner pre-flight runs on the login/batch node (no GPU), so it would fail for the wrong reason. The correct form is passive logging inside srun. Hard guarantee already comes from DataFactory.__init__ raise + the code change itself.

---

## Claude's Discretion

- Logger name / level inside DataFactory (use existing setup, INFO level)
- Whether to add device to DataFactory repr (decided: no — log line is sufficient)
- Exact wording of EvalConfigError messages (be specific about configured value and expected values)

## Deferred Ideas

None — discussion stayed within phase scope.
