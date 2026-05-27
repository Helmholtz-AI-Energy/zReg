# Phase 17: Framework Config & DataFactory - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Create `eval/config.py` (`EvalConfig` pydantic model with YAML loading/validation and `EvalConfigError`) and `eval/data_factory.py` (`DataFactory` class wrapping existing `load_data_from_tracklets`, `load_shah_from_csv`, generators, and corruption functions). Both are standalone modules in the `eval/` namespace directory (no `eval/__init__.py`). Unit tests in `tests/test_data_factory.py` are included in this phase.

Phase 18 (MetricsEngine & Result Types) and all downstream phases are out of scope.

</domain>

<decisions>
## Implementation Decisions

### EvalConfig — YAML Parsing & Validation
- **D-01:** `EvalConfig` is a **pydantic `BaseModel`** (not a plain dataclass). pydantic handles type coercion, validation, and field defaults.
- **D-02:** Only `data_path` is required. All other FRAME-01 fields (`data_format`, `ground_truth_path`, `n_synthetic`, `transform_degree`, `augmentation_params`, `run_alignment`, `run_label_transfer`, `search_space`, `search_strategy`, `tier`, `n_trials`, `output_dir`, `save_plots`, `verbose`, `val_split`) have sensible defaults.
- **D-03:** `EvalConfigError(ValueError)` is raised with a plain human-readable message (e.g., `"EvalConfig: 'data_path' is required"`). pydantic's `ValidationError` must be caught internally and re-raised as `EvalConfigError` — no raw pydantic errors leak to callers. `run_eval.py` (Phase 23) will catch `EvalConfigError` and print the message without stacktrace.
- **D-04:** `EvalConfig.from_yaml(path: str | Path) -> EvalConfig` is the canonical class method for loading. Internally: `pyyaml.safe_load()` → `EvalConfig(**data)`, with `ValidationError` caught and wrapped in `EvalConfigError`.

### prepare_split() — Train/Val Split
- **D-05:** `prepare_split(dataset: dict[int, zRegPointCloud]) -> tuple[dict[int, zRegPointCloud], dict[int, zRegPointCloud]]`. Uses `random.sample` to randomly select which frame indices go into the val set (size = `int(len(keys) * config.val_split)`). Both returned dicts have keys in ascending sorted order — temporal ordering is preserved within each subset.
- **D-06:** Split ratio comes from `EvalConfig` field `val_split: float = 0.2`.
- **D-07:** If the dataset has only 1 frame (too small to split), `prepare_split()` returns `(dataset, {})` silently — no exception raised.

### DataFactory — Construction & State
- **D-08:** `DataFactory.__init__(config: EvalConfig)` stores the config and does NOT load data eagerly. Data is loaded lazily on first method call.
- **D-09:** `DataFactory` caches data after first load. `load_real()` stores the result in `self._real_dataset`; `generate_synthetic()` stores the result in `self._synthetic_dataset`. Subsequent calls return the cached result without reloading.

### get_ground_truth() — GT Extraction
- **D-10:** Both `tracklets` and CSV formats load `id` into `zRegPointCloud['id']`. `get_ground_truth()` extracts `pc['id']` from the loaded dataset by default. If `config.ground_truth_path` is explicitly set (not `None`), it reads from the external file using the same loader as `config.data_format`.
- **D-11:** `get_ground_truth()` returns `dict[int, torch.Tensor]` — one id tensor per frame, keyed by frame index. Mirrors the structure of the dataset itself.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §FRAME-01 — Full EvalConfig field list: `data_path`, `data_format`, `ground_truth_path`, `n_synthetic`, `transform_degree`, `augmentation_params`, `run_alignment`, `run_label_transfer`, `search_space`, `search_strategy`, `tier`, `n_trials`, `output_dir`, `save_plots`, `verbose`
- `.planning/REQUIREMENTS.md` §FRAME-02 — Full DataFactory spec: `load_real()`, `generate_synthetic()`, `augment()`, `prepare_split()`, `get_ground_truth()`; test gate criteria

### Existing Data Loaders
- `src/zreg/dataset.py` §`load_data_from_tracklets` (line 55) — signature: `(filepath: str, device: str = "cpu") -> tuple[dict[int, zRegPointCloud], dict[int, dict]]`; MATLAB tracklets loader
- `src/zreg/dataset.py` §`load_shah_from_csv` (line 288) — signature: `(filepath: str | Path, device: str | torch.device) -> dict[int, zRegPointCloud]`; CSV loader with `id` field populated
- `src/zreg/dataset.py` §`zRegPointCloud` (top) — dict subclass with fields `pos`, `color`, `id`, `fps-idx`; understand constructor before building DataFactory

### Existing Generators & Corruption Functions
- `src/zreg/generators/` — installed package; 7 public symbols: `generate_trajectory`, `apply_rigid`, `apply_affine`, `add_gaussian_noise`, `add_outliers`, `generate_labels`, `remove_labels`; all accept `seed: int | None = 42` and return/accept `dict[int, zRegPointCloud]`

### eval/ Patterns (from prior phases)
- `eval/tracking/__init__.py` — pattern for importable eval subpackage: `__all__` export, re-exports from inner module
- `eval/run_real.py` — pattern for standalone eval script: `sys.path.insert(0, _repo_root)` at top; `from zreg.dataset import load_data_from_tracklets` works after path insertion
- `tests/conftest.py` — `sys.path.insert(0, repo_root)` already present (Phase 14); `from eval.* import ...` works in tests without changes

### Testing
- `tests/test_data_factory.py` — new file this phase; gate criteria per FRAME-02: EvalConfig-from-YAML and split-shape gates

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `load_data_from_tracklets(filepath, device)` — call directly from `DataFactory.load_real()` when `config.data_format == "tracklets"`; discard the raw tracklets second return value
- `load_shah_from_csv(filepath, device)` — call directly from `DataFactory.load_real()` when `config.data_format == "csv"`
- `src/zreg/generators/` installed package — `DataFactory.generate_synthetic()` imports from here; wrap with `config.n_synthetic`, `config.transform_degree`, `config.augmentation_params` as parameters
- `AffineTransformation()` default has `t=[1,1,1]` (NOT identity) — tests using `apply_affine` with default params must use `AffineTransformation(t=torch.zeros(3))` for identity verification (Phase 14 landmine)

### Established Patterns
- `eval/` is a namespace directory — `eval/config.py` and `eval/data_factory.py` are standalone modules; no `eval/__init__.py` needed or allowed
- `__all__` in every subpackage `__init__.py` — if any `eval/` subpackage `__init__.py` is created, it must define `__all__`
- NumPy-style docstrings on all public functions and classes — `Parameters`, `Returns`, `Raises` sections
- `Path(output_dir).mkdir(parents=True, exist_ok=True)` pattern from Phase 15 for auto-creating output dirs
- pydantic BaseModel: use `model_config = ConfigDict(extra="forbid")` to reject unknown YAML keys with a clear error

### Integration Points
- All downstream phases (18–23) depend on `EvalConfig` and `DataFactory` — this is the data foundation layer; API must be stable
- Phase 23 CLI (`run_eval.py`) will call `EvalConfig.from_yaml(args.config)` and `DataFactory(config)` — keep construction clean
- Generators at `src/zreg/generators/` are the installed package (Phase 14); import as `from zreg.generators import generate_trajectory, add_gaussian_noise, ...`

</code_context>

<specifics>
## Specific Ideas

- `EvalConfig.from_yaml(path)` classmethod is the public entry point for YAML loading — `EvalConfig(**data)` construction is an implementation detail
- pydantic's `model_config = ConfigDict(extra="forbid")` will catch typos in YAML keys and raise descriptive errors (wrapped to `EvalConfigError`)
- `random.sample(sorted(dataset.keys()), k=val_count)` then sort both resulting key sets before constructing the two dicts
- `DataFactory._real_dataset: dict[int, zRegPointCloud] | None = None` and `DataFactory._synthetic_dataset: dict[int, zRegPointCloud] | None = None` — None means not yet loaded

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 17-Framework Config & DataFactory*
*Context gathered: 2026-05-27*
