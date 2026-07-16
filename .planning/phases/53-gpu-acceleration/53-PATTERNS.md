# Phase 53: GPU Acceleration - Pattern Map

**Mapped:** 2026-07-16
**Files analyzed:** 6 files (2 Python modifications, 8 YAML modifications, 1 sbatch modification, 2 test modifications)
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `eval/config.py` | config/model | request-response (parse-time validation) | `eval/config.py` itself (existing `@field_validator` + `@model_validator` patterns) | exact |
| `eval/data_factory.py` | service/factory | CRUD + I/O | `eval/data_factory.py` itself (existing `__init__`, `load_real`, `load_target`) | exact |
| `baseline_experiments/configs_horeka/**/*.yaml` (8 files) | config | N/A | `baseline_experiments/configs_horeka/selfcal/kobitski_ew06_alignment.yaml` | exact |
| `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` | config/script | N/A | `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` itself (existing `[VERIFY ON HOREKA]` pattern) | exact |
| `tests/test_eval_config.py` | test | request-response | `tests/test_eval_config.py` — `TestEvalConfigAlignmentMethodValidation` (lines 50–83), `TestEvalConfigLabelTransferMethodValidation` (lines 86–144) | exact |
| `tests/test_data_factory.py` | test | CRUD + mock | `tests/test_data_factory.py` — `TestLoadReal` (lines 207–243), `TestLoadTarget` (lines 251–294) | exact |

---

## Pattern Assignments

### `eval/config.py` — add `device` field + `@field_validator` + `@model_validator`

**Analog:** `eval/config.py` existing validator patterns (self-referential — copy internal patterns)

**Current pydantic import** (line 14):
```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
```
Required change — add `model_validator`:
```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
```

**Field declaration pattern** (copy from `alignment_method` at lines 280–281 and `swd_variant` at lines 281–284):
```python
# Insert near other string fields, after label_transfer_method block (~line 288)
device: str = Field(
    default="cpu",
    description="Compute device: 'cpu', 'cuda', 'cuda:0', 'cuda:1', or 'mps'"
)
```

**`@field_validator` pattern to copy** (lines 347–369 — `validate_alignment_method`):
```python
@field_validator("alignment_method")
@classmethod
def validate_alignment_method(cls, v: str) -> str:
    """Validate that alignment_method is 'cpd', 'icp', or 'swd'."""
    if v not in ("cpd", "icp", "swd"):
        raise ValueError(f"alignment_method must be 'cpd', 'icp', or 'swd'; got {v!r}")
    return v
```
New validator copies this structure exactly:
```python
@field_validator("device")
@classmethod
def validate_device(cls, v: str) -> str:
    if v not in ("cpu", "cuda", "cuda:0", "cuda:1", "mps"):
        raise ValueError(
            f"device must be one of {{'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'}}; got {v!r}"
        )
    return v
```

**`@model_validator(mode='after')` pattern** — new in this file; mirrors the cross-field `validate_swd_variant` concept (lines 371–400) but uses `model_validator` because both fields must already be resolved:
```python
@model_validator(mode='after')
def validate_device_icp_compat(self) -> "EvalConfig":
    if self.device != "cpu" and self.alignment_method == "icp":
        raise ValueError(
            f"device='{self.device}' is incompatible with alignment_method='icp': "
            "Open3D ICP requires CPU-resident tensors "
            "(explicit .cpu().numpy() round-trips in icp.py:114-115, 196-197)"
        )
    return self
```

**Error wrapping chain** (lines 334–345 — `from_yaml`):
```python
except ValidationError as e:
    first = e.errors()[0]
    field = ".".join(str(x) for x in first["loc"])
    raise EvalConfigError(f"EvalConfig: field '{field}': {first['msg']}") from e
```
Both `@field_validator` and `@model_validator` raise `ValueError` internally; pydantic wraps to `ValidationError`; `from_yaml` wraps to `EvalConfigError`. Do not raise `EvalConfigError` inside the validators.

---

### `eval/data_factory.py` — `__init__` guard + 6 device substitutions + logging

**Analog:** `eval/data_factory.py` itself — `__init__` (lines 117–133), `load_real` (lines 135–171), `load_target` (lines 173–237), `get_ground_truth` (lines 638–643)

**`__init__` structure** (lines 117–133) — guard inserted immediately after `self.config = config` at line 126:
```python
def __init__(self, config: EvalConfig) -> None:
    self.config = config
    # --- insert D-03 guard here, after self.config = config ---
    self._real_dataset: dict[int, zRegPointCloud] | None = None
    self._synthetic_dataset: dict[int, zRegPointCloud] | None = None
    self._target_dataset: dict[int, zRegPointCloud] | None = None
    ...
```
Guard pattern (insert at line 127, after `self.config = config`):
```python
if config.device != "cpu":
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"DataFactory: config.device={config.device!r} requested but "
            "torch.cuda.is_available() is False"
        )
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"DataFactory: config.device={config.device!r} requested but "
            "torch.backends.mps.is_available() is False"
        )
```
Note: `torch` is already imported at line 40 (`import torch`). No new import needed.

**6 device substitution sites** — lines 158, 161, 224, 227, 640, 642:
```python
# BEFORE (all 6 sites):
device="cpu"

# AFTER (all 6 sites):
device=self.config.device
```
Specific call sites:
- Line 158: `load_data_from_tracklets(self.config.data_path, device="cpu")` in `load_real()`
- Line 161: `load_shah_from_csv(self.config.data_path, device="cpu")` in `load_real()`
- Line 224: `load_data_from_tracklets(self.config.target_data_path, device="cpu")` in `load_target()`
- Line 227: `load_shah_from_csv(self.config.target_data_path, device="cpu")` in `load_target()`
- Line 640: `load_data_from_tracklets(self.config.ground_truth_path, device="cpu")` in `get_ground_truth()`
- Line 642: `load_shah_from_csv(self.config.ground_truth_path, device="cpu")` in `get_ground_truth()`

**Logging pattern** (analog: existing bare `logging.debug` at line 840):
```python
# line 840 — only existing log call in the file (bare logging module, no named logger):
logging.debug(
    "DataFactory._standardize: method=%s mean=%s", cfg.method, stats["mean"]
)
```
`logging` is already imported at line 15. New D-07 calls use `logging.info` at the same style:
```python
# In load_real(), after dataset = self._standardize(dataset) (line 169), before self._real_dataset = dataset:
logging.info("Loaded %s dataset on %s (%d frames)", "source", self.config.device, len(dataset))

# In load_target(), after dataset = self._standardize(dataset, ...) (line 235), before self._target_dataset = dataset:
logging.info("Loaded %s dataset on %s (%d frames)", "target", self.config.device, len(dataset))
```
No log call in `get_ground_truth()` — D-07 specifies only `load_real()` and `load_target()`.

---

### `baseline_experiments/configs_horeka/**/*.yaml` (8 files)

**Analog:** `baseline_experiments/configs_horeka/selfcal/kobitski_ew06_alignment.yaml` (full file read above)

**YAML field addition pattern** — add `device: "cuda"` to each file. Placement: after `data_format` / `max_points_per_frame` block, before `run_alignment`. Same placement as other top-level scalar fields:
```yaml
data_path: <existing>
data_format: <existing>
max_points_per_frame: <existing if present>
device: "cuda"             # ADD — GPU-03: GPU tensor placement for HoreKa runs
run_alignment: <existing>
```

**All 8 target files** (none currently has a `device:` field):
1. `baseline_experiments/configs_horeka/baseline_no_hpo/ew06_vs_shah.yaml`
2. `baseline_experiments/configs_horeka/baseline_with_selfcal/ew06_vs_shah.yaml`
3. `baseline_experiments/configs_horeka/ground_truth/kobitski_ew06.yaml`
4. `baseline_experiments/configs_horeka/ground_truth/shah_sample1.yaml`
5. `baseline_experiments/configs_horeka/selfcal/kobitski_ew06_alignment.yaml`
6. `baseline_experiments/configs_horeka/selfcal/shah_alignment.yaml`
7. `baseline_experiments/configs_horeka/selfcal/shah_label_transfer.yaml`
8. `baseline_experiments/configs_horeka/smoke/kobitski_ew06_alignment.yaml`

Note: CONTEXT.md D-08 states "all 7 files" but RESEARCH.md confirms 8 files. The smoke config must be included.

---

### `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` — GPU-03 verification comment

**Analog:** Existing `[VERIFY ON HOREKA]` convention in the same file (lines 19, 20, 25, 27–31, 36)

**Existing pattern** (lines 62–65):
```bash
# After job completes, verify passing criteria:
#   grep "rank" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out | head -20
#   find baseline_experiments/experiments/smoke/kobitski_ew06_alignment -name "best_params.json" | wc -l
#   grep -i "traceback\|error\|exception" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out | head -20
```
New GPU-03 comment appended after line 65 (end of file):
```bash
# [VERIFY ON HOREKA] GPU-03: grep 'Loaded source dataset on cuda' in SLURM output to confirm GPU tensors
#   grep "Loaded source dataset on cuda" baseline_experiments/logs/slurm-${SLURM_JOB_ID}.out
```

---

### `tests/test_eval_config.py` — add `TestEvalConfigDevice` + `TestEvalConfigDeviceICPGuard`

**Analog:** `TestEvalConfigAlignmentMethodValidation` (lines 50–83) — copy class structure, docstring style, `pytest.raises(ValueError, match=...)` pattern

**Copy pattern from `TestEvalConfigAlignmentMethodValidation`** (lines 50–83):
```python
class TestEvalConfigAlignmentMethodValidation:
    """Tests for alignment_method field validation."""

    def test_alignment_method_invalid_value_raises(self, tmp_path):
        """Test that invalid alignment_method raises ValueError."""
        with pytest.raises(ValueError, match="alignment_method must be 'cpd', 'icp', or 'swd'"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="invalid_method",
            )

    def test_alignment_method_case_sensitive(self, tmp_path):
        with pytest.raises(ValueError, match="alignment_method must be 'cpd', 'icp', or 'swd'"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                alignment_method="ICP",
            )
```

New `TestEvalConfigDevice` follows same structure:
```python
class TestEvalConfigDevice:
    """Tests for device field validation (GPU-01)."""

    def test_device_default_is_cpu(self, tmp_path):
        config = EvalConfig(data_path=str(tmp_path / "data.mat"))
        assert config.device == "cpu"

    def test_device_cpu_explicit(self, tmp_path):
        config = EvalConfig(data_path=str(tmp_path / "data.mat"), device="cpu")
        assert config.device == "cpu"

    def test_device_cuda_accepted(self, tmp_path):
        config = EvalConfig(data_path=str(tmp_path / "data.mat"), device="cuda")
        assert config.device == "cuda"

    def test_device_cuda_0_accepted(self, tmp_path):
        config = EvalConfig(data_path=str(tmp_path / "data.mat"), device="cuda:0")
        assert config.device == "cuda:0"

    def test_device_cuda_1_accepted(self, tmp_path):
        config = EvalConfig(data_path=str(tmp_path / "data.mat"), device="cuda:1")
        assert config.device == "cuda:1"

    def test_device_mps_accepted(self, tmp_path):
        config = EvalConfig(data_path=str(tmp_path / "data.mat"), device="mps")
        assert config.device == "mps"

    def test_device_invalid_raises(self, tmp_path):
        with pytest.raises(ValueError, match="device must be one of"):
            EvalConfig(data_path=str(tmp_path / "data.mat"), device="gpu")

    def test_device_cuda_7_raises(self, tmp_path):
        """cuda:7 is outside the whitelisted set and must raise."""
        with pytest.raises(ValueError, match="device must be one of"):
            EvalConfig(data_path=str(tmp_path / "data.mat"), device="cuda:7")

    def test_device_yaml_round_trip(self, tmp_path):
        """GPU-03: YAML with device: cuda parses correctly."""
        p = tmp_path / "cfg.yaml"
        p.write_text("data_path: x.mat\ndevice: 'cuda'\n")
        config = EvalConfig.from_yaml(str(p))
        assert config.device == "cuda"
```

New `TestEvalConfigDeviceICPGuard` copies the cross-field validation pattern from `TestEvalConfigAlignmentMethodValidation`:
```python
class TestEvalConfigDeviceICPGuard:
    """Tests for device+icp model_validator cross-field guard (GPU-01, D-05)."""

    def test_device_cuda_plus_icp_raises(self, tmp_path):
        with pytest.raises(ValueError, match="incompatible with alignment_method='icp'"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                device="cuda",
                alignment_method="icp",
            )

    def test_device_mps_plus_icp_raises(self, tmp_path):
        with pytest.raises(ValueError, match="incompatible with alignment_method='icp'"):
            EvalConfig(
                data_path=str(tmp_path / "data.mat"),
                device="mps",
                alignment_method="icp",
            )

    def test_device_cuda_plus_cpd_succeeds(self, tmp_path):
        """cuda + cpd is a valid combination — no guard should fire."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            device="cuda",
            alignment_method="cpd",
        )
        assert config.device == "cuda"
        assert config.alignment_method == "cpd"

    def test_device_cpu_plus_icp_succeeds(self, tmp_path):
        """cpu + icp is the normal path — no guard should fire."""
        config = EvalConfig(
            data_path=str(tmp_path / "data.mat"),
            device="cpu",
            alignment_method="icp",
        )
        assert config.device == "cpu"
```

---

### `tests/test_data_factory.py` — `TestDataFactoryDeviceGuard` + update dispatch tests

**Analog:** `TestLoadReal` (lines 207–243) and `TestLoadTarget` (lines 251–294) — copy `patch`, `_make_mock_ds`, and `assert_called_once_with` patterns

**Mock setup pattern** (from `TestLoadReal._make_mock_ds`, lines 210–212):
```python
from unittest.mock import MagicMock, patch   # already imported at line 19
import torch                                  # already imported at line 30

def _make_mock_ds(self):
    return {0: zRegPointCloud(pos=torch.zeros(3, 3), label=None, id=torch.arange(3))}
```

**`patch` + `assert_called_once_with` pattern** (from `TestLoadReal.test_dispatches_tracklets`, lines 214–223):
```python
def test_dispatches_tracklets(self):
    cfg = EvalConfig(data_path="x.mat", data_format="tracklets")
    factory = DataFactory(cfg)
    mock_ds = self._make_mock_ds()
    with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
        result = factory.load_real()
    m.assert_called_once_with("x.mat", device="cpu")   # <-- update to device=cfg.device
```

**Updated dispatch tests** — change `device="cpu"` assertion to `device=cfg.device` (or `device="cpu"` when cfg uses default):
```python
# TestLoadReal.test_dispatches_tracklets — update assertion at line 221:
m.assert_called_once_with("x.mat", device="cpu")
# becomes:
m.assert_called_once_with("x.mat", device=cfg.device)  # device defaults to "cpu" — still passes

# TestLoadReal.test_dispatches_csv — update assertion at line 232:
m.assert_called_once_with("x.csv", device="cpu")
# becomes:
m.assert_called_once_with("x.csv", device=cfg.device)
```
Same update pattern for `TestLoadTarget.test_dispatches_tracklets` (line 265) and `TestLoadTarget.test_dispatches_csv` (line 276).

**New `TestDataFactoryDeviceGuard`** — uses `unittest.mock.patch` to mock torch availability (Pitfall 6):
```python
class TestDataFactoryDeviceGuard:
    """GPU-02: DataFactory.__init__ raises RuntimeError for unavailable devices."""

    def test_cuda_unavailable_raises_runtime_error(self):
        """When cuda requested but torch.cuda.is_available() is False, raises RuntimeError."""
        cfg = EvalConfig(data_path="x.mat", device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="torch.cuda.is_available\\(\\) is False"):
                DataFactory(cfg)

    def test_cuda_0_unavailable_raises_runtime_error(self):
        """cuda:0 uses startswith('cuda') check — must raise when CUDA unavailable."""
        cfg = EvalConfig(data_path="x.mat", device="cuda:0")
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="torch.cuda.is_available\\(\\) is False"):
                DataFactory(cfg)

    def test_mps_unavailable_raises_runtime_error(self):
        """When mps requested but torch.backends.mps.is_available() is False, raises RuntimeError."""
        cfg = EvalConfig(data_path="x.mat", device="mps")
        with patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="torch.backends.mps.is_available\\(\\) is False"):
                DataFactory(cfg)

    def test_cpu_skips_guard(self):
        """device='cpu' never enters the guard — no RuntimeError regardless of CUDA availability."""
        cfg = EvalConfig(data_path="x.mat", device="cpu")
        with patch("torch.cuda.is_available", return_value=False):
            factory = DataFactory(cfg)  # must not raise
        assert factory.config.device == "cpu"

    def test_cuda_available_constructs_successfully(self):
        """When CUDA is available, cuda device constructs without error."""
        cfg = EvalConfig(data_path="x.mat", device="cuda")
        with patch("torch.cuda.is_available", return_value=True):
            factory = DataFactory(cfg)  # must not raise
        assert factory.config.device == "cuda"
```

**New device threading tests** — add to existing `TestLoadReal` or a new subclass:
```python
def test_dispatches_tracklets_with_custom_device(self):
    """D-04: config.device is threaded to loader — non-cpu device forwarded correctly."""
    cfg = EvalConfig(data_path="x.mat", data_format="tracklets", device="cuda")
    mock_ds = self._make_mock_ds()
    with patch("torch.cuda.is_available", return_value=True):
        factory = DataFactory(cfg)
    with patch("eval.data_factory.load_data_from_tracklets", return_value=(mock_ds, {})) as m:
        factory.load_real()
    m.assert_called_once_with("x.mat", device="cuda")
```

---

## Shared Patterns

### Pydantic `@field_validator` structure
**Source:** `eval/config.py` lines 347–369 (`validate_alignment_method`) and lines 402–428 (`validate_label_transfer_method`)
**Apply to:** New `validate_device` in `eval/config.py`
```python
@field_validator("<field_name>")
@classmethod
def validate_<field_name>(cls, v: str) -> str:
    if v not in (<allowed_set>):
        raise ValueError(f"<field_name> must be one of <allowed_set>; got {v!r}")
    return v
```

### Pydantic `@model_validator(mode='after')` for cross-field guards
**Source:** Pydantic v2 pattern; `model_validator` not yet in `eval/config.py` line 14 (must be added to import)
**Apply to:** New `validate_device_icp_compat` in `eval/config.py`
**Key rule:** Raise `ValueError` (not `EvalConfigError`) inside the validator body — pydantic wraps it; `from_yaml` rewraps as `EvalConfigError`.

### `unittest.mock.patch` for hardware availability mocking
**Source:** `tests/test_data_factory.py` lines 219, 230, 263, 274 — existing `patch("eval.data_factory.load_data_from_tracklets", ...)` pattern
**Apply to:** New `TestDataFactoryDeviceGuard` tests — patch `torch.cuda.is_available` and `torch.backends.mps.is_available`
```python
with patch("torch.cuda.is_available", return_value=False):
    with pytest.raises(RuntimeError, match="..."):
        DataFactory(cfg)
```

### Bare `logging` module (no named logger)
**Source:** `eval/data_factory.py` lines 15 (import) and 840 (usage — `logging.debug(...)`)
**Apply to:** D-07 `logging.info(...)` calls in `load_real()` and `load_target()`
**Note:** Do not introduce `logging.getLogger(__name__)` — the file uses bare `logging` throughout.

### `[VERIFY ON HOREKA]` sbatch comment convention
**Source:** `baseline_experiments/scripts/launch_horeka_multirank_test.sbatch` lines 19–31, 36
**Apply to:** GPU-03 verification comment appended at end of same file
**Format:** `# [VERIFY ON HOREKA] <REQUIREMENT-ID>: <verification instruction>`

---

## No Analog Found

All files have close analogs within the codebase. No entries.

---

## Metadata

**Analog search scope:** `eval/`, `tests/`, `baseline_experiments/configs_horeka/`, `baseline_experiments/scripts/`
**Files scanned:** 6 source files read in full (config.py, data_factory.py lines 1–270 + 625–660 + 835–855, test_eval_config.py lines 1–360, test_data_factory.py lines 1–310, kobitski_ew06_alignment.yaml, launch_horeka_multirank_test.sbatch)
**Pattern extraction date:** 2026-07-16
