# Phase 28: Script Integration — generate_datasets uses DataFactory - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 28-script-integration-generate-datasets-uses-datafactory
**Areas discussed:** DataFactory wiring, Dropout RNG parity

---

## DataFactory wiring

User asked for pros/cons before deciding.

| Option | Description | Selected |
|--------|-------------|----------|
| augment() via minimal EvalConfig | `DataFactory(EvalConfig(data_path="", augmentation_params={...})).augment(dataset)` per call. Routes through canonical dispatch. data_path="" safe — no I/O on construction. | ✓ |
| Direct instance methods | One shared dummy DataFactory; call df.scale(), df.drop_points() directly; noise via add_gaussian_noise(). Mixes DF methods with raw zreg calls. | |

**User's choice:** Option A — augment() via minimal EvalConfig
**Notes:** User requested pros/cons discussion before deciding. Chose canonical augment() path for spec alignment. Noted that data_path="" feels slightly unclean but is safe per DataFactory D-08.

---

## Dropout RNG parity

| Option | Description | Selected |
|--------|-------------|----------|
| Accept the divergence | Dropout CSVs differ (NumPy → PyTorch RNG). Relax criterion 3 to bit-identical for scaling/noise; dropout reproducible but not matching. Document in plan. | ✓ |
| Workaround: regenerate baseline first | Delete existing dropout CSVs before refactor so new RNG becomes the baseline. Criterion 3 vacuously satisfied. | |
| Strict: keep NumPy path | Don't replace _augment_dropout in this phase — partial refactor only. | |

**User's choice:** Accept the divergence (recommended)
**Notes:** No workaround needed. Plan must document that dropout CSV content changes are expected due to RNG implementation change, not a bug.

---

## Claude's Discretion

- **Import order (D-06):** Ordering of new imports (`eval.data_factory`, `eval.config`) relative to existing zreg/torch imports — Claude to handle per established libomp guard pattern.
- **AUGMENTATION_GRID key mapping:** Mapping `"factor"` → `"scale_factor"` and `"fraction"` → `"dropout_fraction"` in the refactored call site — Claude to implement.

## Deferred Ideas

None — discussion stayed within phase scope.
