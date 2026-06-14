---
phase: 32
name: "Heterogeneous Paired Evaluation — target_data_format"
date: 2026-06-14
slug: heterogeneous-paired-evaluation-target-data-format
---

# Phase 32 Context

## Domain

Enable paired evaluation across datasets with **different file formats** — e.g. Kobitski `.tracklets` as source, Shah `.csv` as target (or the reverse). The blocker is that `DataFactory.load_target()` always reads `config.data_format`, the same field used for the source. Adding `target_data_format` (with a `None` fallback to `data_format`) lifts that constraint without breaking any existing config.

## Canonical Refs

- `eval/config.py` — `EvalConfig` pydantic model; `extra="forbid"` requires explicit field declaration
- `eval/data_factory.py` — `DataFactory.load_target()` lines 122–172; format dispatch via `if self.config.data_format == "tracklets"` / `elif == "csv"`
- `configs/paired_alignment.yaml` — existing paired config (no `target_data_format`); must remain valid after this phase (backward compat guard)
- `.planning/phases/32-heterogeneous-paired-evaluation-target-data-format/prompts/phase-32-target-data-format.md` — original prompt (full acceptance criteria and data file paths)

## Decisions

### EvalConfig field

- Field name: `target_data_format: str | None = None`
- Accepted values: `"tracklets"`, `"csv"`, `None`
- Default `None` means "inherit from `data_format`" — backward compatible; all existing YAMLs without this key continue to work
- No `Literal` type constraint on this field (keep it `str | None`) to stay consistent with `data_format: str`; validation happens at `load_target()` call time via the same `unknown data_format` error path that already exists
- Position: after `transform_spec` (last field in the model)
- Docstring: document the fallback behaviour and accepted values

### DataFactory.load_target() change

- Replace the single `self.config.data_format` lookup with:
  ```python
  fmt = self.config.target_data_format or self.config.data_format
  ```
- Use `fmt` in the existing `if/elif` dispatch block — no other logic changes
- The `unknown data_format` error branch already handles invalid values correctly

### What does NOT change

- `EvaluationRunner.run()` — already calls `factory.load_real()` and `factory.load_target()`
- `AlignmentStage`, `LabelTransferStage` — format-agnostic after loading
- `MetricsEngine`, `EvalReport`, `viz.py`, `HyperparamOptimizer`
- `run_eval.py` CLI

### Scenario configs

Three new configs under `configs/`:

1. `kobitski_vs_shah.yaml` — Kobitski ew06 tracklets as source, Shah sample-1 CSV as target
2. `shah_vs_kobitski.yaml` — Shah sample-1 CSV as source, Kobitski ew06 tracklets as target
3. `kobitski_vs_kobitski_cross.yaml` — Kobitski ew06 as source, Kobitski ew08 as target (same format, different embryos — sanity check before the format difference)

All three: `tier: sanity`, `n_trials: 3`, `run_alignment: true`, `run_label_transfer: true`

Data file paths (absolute, already on disk):
- Kobitski ew06: `/Users/valeriekieslinger/Documents/Hiwi/BA/data/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets`
- Kobitski ew08: `/Users/valeriekieslinger/Documents/Hiwi/BA/data/kobitski_data/12_11_27_embryo_ew_08_Cleaned_BackTracked_Oriented.tracklets`
- Shah sample-1: `/Users/valeriekieslinger/Documents/Hiwi/BA/data/shah_data/sample-1/sample-1-cell-tracks.csv`

### Tests

- `TestEvalConfigTargetDataFormat` — in `tests/test_data_factory.py`:
  - `target_data_format: "csv"` accepted without error
  - `target_data_format: None` (default) is backward compatible
  - Unknown value for `target_data_format` does not raise at config construction (validated at use time)
- `TestLoadTargetFormatDispatch` — in `tests/test_data_factory.py`:
  - When `target_data_format="csv"`, `load_target()` calls the CSV loader
  - When `target_data_format="tracklets"`, `load_target()` calls the tracklets loader
  - When `target_data_format=None`, `load_target()` falls back to `data_format`
  - Implemented via mocks (no real data files required in unit tests)
- Smoke tests in `tests/test_cli.py` (`TestScenarioConfigs`):
  - `EvalConfig.from_yaml("configs/kobitski_vs_shah.yaml")` parses without error
  - `EvalConfig.from_yaml("configs/shah_vs_kobitski.yaml")` parses without error
  - `EvalConfig.from_yaml("configs/kobitski_vs_kobitski_cross.yaml")` parses without error
  - Guard with `@pytest.mark.skipif(not Path(data_path).exists(), ...)` so tests are skipped cleanly on machines without the data

## Code Context

Reusable assets:
- `DataFactory.load_real()` (lines 86–118 in `eval/data_factory.py`) — identical tracklets/CSV dispatch pattern; `load_target()` should mirror it exactly after the `fmt` variable is resolved
- `TestLoadTargetRequiresPairedMode` and related classes in `tests/test_data_factory.py` — test fixture patterns for mocking `load_data_from_tracklets` and `load_shah_from_csv`
- `TestScenarioConfigs` in `tests/test_cli.py` — existing parametrised pattern with `@pytest.mark.skipif` guards for path-dependent configs

## Deferred Ideas

None — scope is fully defined by the prompt.
