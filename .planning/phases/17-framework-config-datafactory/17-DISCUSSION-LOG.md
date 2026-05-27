# Phase 17: Framework Config & DataFactory - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 17-Framework Config & DataFactory
**Areas discussed:** YAML parsing & validation, prepare_split() strategy, DataFactory construction, get_ground_truth() paths

---

## YAML Parsing & Validation

| Option | Description | Selected |
|--------|-------------|----------|
| pyyaml + dataclass | pyyaml.safe_load() → manual dataclass construction, raise EvalConfigError on missing/bad fields | |
| pydantic BaseModel | Replace dataclass with pydantic.BaseModel for type coercion, validators, descriptive errors | ✓ |
| dacite | pyyaml.safe_load() then dacite.from_dict(EvalConfig, data) | |

**User's choice:** pydantic BaseModel

| Option | Description | Selected |
|--------|-------------|----------|
| Only data_path required | data_path is the one required field; everything else has defaults | ✓ |
| data_path + data_format required | Both required since format drives loader selection | |
| You decide | Planner picks required vs optional fields | |

**User's choice:** Only data_path required

| Option | Description | Selected |
|--------|-------------|----------|
| Human-readable string, no stacktrace | EvalConfigError(ValueError) with plain message; pydantic ValidationError caught internally | ✓ |
| Re-raise pydantic ValidationError | Let pydantic's detailed field-by-field error propagate | |

**User's choice:** Human-readable string, no stacktrace

---

## prepare_split() Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Frame-index split | First N% of frame indices → train; preserves temporal order, no shuffling | |
| Random frame split | Randomly shuffle frame indices, then split by ratio | ✓ |
| You decide | Planner picks split strategy | |

**Notes:** User clarified via free text: "take random frames from the trajectory but should not input the shuffled frames. The frames need to stay ordered after removing frames." Final decision: randomly select which indices go to val (`random.sample`), but return both dicts with keys in ascending sorted order.

| Option | Description | Selected |
|--------|-------------|----------|
| EvalConfig field | Add val_split: float = 0.2 to EvalConfig; config-file-driven | ✓ |
| prepare_split() parameter | def prepare_split(dataset, val_ratio=0.2); no EvalConfig field | |
| You decide | Planner decides | |

**User's choice:** EvalConfig field

| Option | Description | Selected |
|--------|-------------|----------|
| Raise ValueError | Fast failure on too-small dataset | |
| Return (dataset, {}) silently | All frames to train, empty dict for val | ✓ |
| You decide | Planner picks edge-case behavior | |

**User's choice:** Return (dataset, {}) silently

---

## DataFactory Construction

| Option | Description | Selected |
|--------|-------------|----------|
| Takes EvalConfig, loads lazily | DataFactory(config: EvalConfig); stores config, no I/O on init | ✓ |
| Takes EvalConfig, loads eagerly | DataFactory immediately loads data on construction | |
| Stateless class methods | All methods are @classmethods or @staticmethods | |

**User's choice:** Takes EvalConfig, loads lazily

| Option | Description | Selected |
|--------|-------------|----------|
| No caching, reload each call | Each call to load_real() reloads from source | |
| Cache after first load | Store result in self._dataset; second call returns cached | ✓ |
| You decide | Planner picks caching strategy | |

**User's choice:** Cache after first load

---

## get_ground_truth() Paths

| Option | Description | Selected |
|--------|-------------|----------|
| ground_truth_path=None → extract from pc['id'] | Config-driven: None = extract from point cloud; set = read external file | |
| data_format drives the decision | tracklets always has id in pc; CSV requires external GT file | ✓ |
| You decide | Planner decides switching logic | |

**Initial choice:** data_format drives the decision

**Follow-up clarification:** Both tracklets and CSV loaders already populate `pc['id']`. Result: `get_ground_truth()` always extracts `pc['id']` unless `config.ground_truth_path` is explicitly set as an override (uses same loader as data_format for the external file).

| Option | Description | Selected |
|--------|-------------|----------|
| dict[int, Tensor] — same structure as dataset | {frame_id: id_tensor}; consistent with dataset structure | ✓ |
| Single concatenated Tensor | Flat tensor of all frame ids concatenated | |

**User's choice:** dict[int, Tensor]

---

## Claude's Discretion

None — all gray areas had explicit user choices.

## Deferred Ideas

None — discussion stayed within phase scope.
