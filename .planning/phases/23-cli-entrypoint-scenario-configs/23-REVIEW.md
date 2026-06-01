---
phase: 23-cli-entrypoint-scenario-configs
reviewed: 2026-06-01T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - run_eval.py
  - tests/test_cli.py
  - configs/alignment_sanity.yaml
  - configs/alignment_dev.yaml
  - configs/label_transfer_sanity.yaml
  - configs/label_transfer_dev.yaml
  - configs/combined_full.yaml
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 23: Code Review Report

**Reviewed:** 2026-06-01T00:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed the CLI entrypoint (`run_eval.py`), its test suite (`tests/test_cli.py`), and five
scenario YAML configs (`configs/`). The YAML configs are structurally correct and consistent with
their D-04 specification. The CLI is well-structured overall, but one security gap, two test
reliability defects, and several quality issues are present.

---

## Critical Issues

### CR-01: `shutil.copy` source path is unsanitised — path traversal via `--config`

**File:** `run_eval.py:103`

**Issue:** `_write_run_config` calls `shutil.copy(config_path, dst)` where `config_path` is the
raw string from `args.config` — it is passed directly without any canonicalisation or containment
check. `Path(config.output_dir).resolve()` is applied to the *destination* only (line 162), not to
the source. A caller who supplies `--config ../../../../etc/passwd` (or any path outside the
project) will have that file copied verbatim into `output_dir/run_config.yaml`. Because this copy
happens unconditionally — before any mode check — the sensitive file is exfiltrated to whatever
`output_dir` the attacker also controls via `--output-dir`. This is a directory-traversal /
information-disclosure vulnerability.

Additionally, `EvalConfig.from_yaml` opens `args.config` directly (line 146), so an attacker can
read any world-readable file through the YAML parser as well.

**Fix:**
```python
def _write_run_config(config_path: str, output_dir: str) -> None:
    src = Path(config_path).resolve()
    repo_root = Path(__file__).parent.resolve()
    if not str(src).startswith(str(repo_root)):
        raise EvalConfigError(
            f"EvalConfig: --config path must be inside the project root: {src}"
        )
    dst = Path(output_dir) / "run_config.yaml"
    shutil.copy(src, dst)
```
Apply the same containment check in `main()` before `EvalConfig.from_yaml` is called.

---

## Warnings

### WR-01: `logging.basicConfig` is a no-op in pytest — param-source log tests are unreliable

**File:** `tests/test_cli.py:213,236`

**Issue:** `TestCLIParamSource` tests use `caplog.at_level(logging.INFO, logger="run_eval")` to
intercept log records, but `run_eval.main` calls `logging.basicConfig(level=logging.INFO)` only
when `--verbose` is supplied (line 158–159). `logging.basicConfig` is explicitly documented as a
no-op when the root logger already has handlers, which is always true inside pytest (pytest
installs its own root handler). As a result, `logging.basicConfig` inside `main()` never actually
re-configures the logger, and the `caplog` context manager is relied upon entirely. The tests
happen to pass because `caplog` captures at the requested level independently — but the production
code path is not exercised: when `run_eval.py` is used as a script (not under pytest), if the root
logger has no handlers, `logging.basicConfig` would be the only configuration, and its level would
need to be set correctly. The real defect is that `main()` configures logging *after* the
`EvalConfig.from_yaml` call and the `model_copy` calls, so any `log.info(...)` that could appear
during config loading would be silently dropped even in verbose mode.

**Fix:** Move `logging.basicConfig(level=logging.INFO)` to immediately after argument parsing,
before any other call, so early log statements are not lost:

```python
def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # Configure logging early so no messages are dropped
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    try:
        config = EvalConfig.from_yaml(args.config)
    ...
```

Using `force=True` ensures pytest's pre-installed root handler is replaced, making the
`logging.basicConfig` call effective in both production and test contexts.

### WR-02: `test_config_error_clean_message` asserts single-line output but the assertion is fragile

**File:** `tests/test_cli.py:156`

**Issue:** The test asserts `"\n" not in captured.err.strip()` to verify "single line output." This
assertion passes if the error message itself contains no embedded newlines, but it silently
accepts multi-line output if the second line is all whitespace (since `.strip()` removes leading
and trailing whitespace but does not collapse internal whitespace). More importantly, the assertion
checks `captured.err.strip()` — if `print(..., file=sys.stderr)` adds a trailing newline (which it
always does), that is stripped away, so the assertion is vacuously true even for two-line output
where the second line is blank. The intent (no stacktrace, single line) is not fully enforced.

**Fix:**
```python
err_lines = [ln for ln in captured.err.splitlines() if ln.strip()]
assert len(err_lines) == 1, f"Expected exactly one non-empty error line, got: {captured.err!r}"
assert err_lines[0].startswith("Error: EvalConfig:")
assert "Traceback" not in captured.err
```

### WR-03: `test_full_mode_calls_optimizer_then_runner` does not verify call ordering

**File:** `tests/test_cli.py:109-138`

**Issue:** The test checks that both `mock_opt_instance.run` and `mock_runner_instance.run` were
called once each, and that the runner received the correct params dict — but it does not assert
that the optimizer ran *before* the runner. The comment "Proves runner saw the params written by
optimizer (Pitfall 5 ordering)" is aspirational: because `mock_opt_instance.run.side_effect`
writes the file, if the two calls were reversed the side-effect would still have fired before the
runner read the file (the mock write happens when `.run()` is called regardless of ordering). A
genuine ordering regression (optimizer and runner swapped in source) would not be caught.

**Fix:** Use `unittest.mock.call` or `Mock.assert_has_calls` with `any_order=False`:
```python
from unittest.mock import call

manager = MagicMock()
manager.attach_mock(mock_opt_instance, "opt")
manager.attach_mock(mock_runner_instance, "runner")
# ... run main() ...
manager.assert_has_calls([call.opt.run(), call.runner.run()], any_order=False)
```

---

## Info

### IN-01: `_write_run_config` does not handle `FileNotFoundError` when source config is gone between load and copy

**File:** `run_eval.py:103`

**Issue:** Between line 146 (`EvalConfig.from_yaml(args.config)`) and line 168
(`_write_run_config(args.config, config.output_dir)`), the source config file could in theory be
deleted by a concurrent process. `shutil.copy` would then raise an unhandled `FileNotFoundError`,
which propagates as an unhandled exception rather than a clean `EvalConfigError` message. While
this is a very unlikely race condition, it violates the stated error-handling contract
("Only `EvalConfigError` is caught; all other exceptions indicate bugs"). A transient filesystem
event is not a bug in the caller.

**Fix:** Wrap `shutil.copy` in a try/except and re-raise as `EvalConfigError` with a descriptive
message, or document explicitly that this exception is intentionally allowed to propagate.

### IN-02: Scenario YAML configs omit `output_dir` — all five runs share the same default directory

**File:** `configs/alignment_sanity.yaml:1-7`, `configs/alignment_dev.yaml:1-7`, `configs/label_transfer_sanity.yaml:1-7`, `configs/label_transfer_dev.yaml:1-7`, `configs/combined_full.yaml:1-7`

**Issue:** None of the five scenario configs sets `output_dir`. They all fall back to the
`EvalConfig` default `"experiments/runs"`. Running two scenarios sequentially without specifying
`--output-dir` at the CLI will cause them to share the same output directory, with later runs
overwriting `best_params.json`, `search_history.json`, `eval_report.json`, and `run_config.yaml`
from earlier runs. The `combined_full` scenario (`tier: full`, 20 trials) is particularly
dangerous as it produces the most output and would silently clobber sanity or dev artefacts.

**Fix:** Add scenario-specific `output_dir` defaults to each config, e.g.:
```yaml
# alignment_sanity.yaml
output_dir: experiments/alignment_sanity
```
This is a config-content quality issue, not a code bug, but it will cause silent data loss in
normal multi-scenario workflows.

---

_Reviewed: 2026-06-01T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
