# Phase 15: Experiment Tracking & Run Management - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Create `eval/tracking/` at the repo root — an importable Python package that writes one JSON file + one CSV file per run to `evaluation/runs/`. The public API is a single `log_run()` function that accepts 6 caller-supplied fields, auto-captures `git_hash` / `zreg_version` / `timestamp` internally, and persists all 9 EVAL-04 required fields. Unit tests in `tests/test_tracking.py` are included in this phase.

Phase 16 runner scripts (`run_synthetic.py`, `run_real.py`) are out of scope — those are EVAL-05.

</domain>

<decisions>
## Implementation Decisions

### API Design
- **D-01:** Public API is a plain function `log_run()` — stateless, no class. Phase 16 runner scripts call `from eval.tracking import log_run`.
- **D-02:** `log_run()` returns the `run_id` str so callers can reference the logged entry (e.g., for sweep correlation).
- **D-03:** Caller signature: `log_run(run_id, dataset_path, frame_indices, seed, n_points_before, n_points_after, output_dir="evaluation/runs") -> str`. Three fields are auto-captured internally: `git_hash` via `subprocess.run(["git", "rev-parse", "HEAD"])`, `zreg_version` via `importlib.metadata.version("zreg")`, `timestamp` via `datetime.datetime.now(datetime.timezone.utc).isoformat()`.

### run_id Strategy
- **D-04:** `run_id` is a required `str` parameter — caller-provided, no auto-generation. Runner scripts control naming (useful for correlating with sweep IDs or Optuna trial numbers in Phase 17).

### Output Files & Storage
- **D-05:** Per-run output: `{output_dir}/{run_id}.json` + `{output_dir}/{run_id}.csv`. The CSV contains a header row + one data row (all 9 fields). JSON contains the same data as a dict.
- **D-06:** Default `output_dir` is `"evaluation/runs"` (relative to cwd) — matches the path specified in EVAL-05 for runner output.
- **D-07:** `log_run()` auto-creates `output_dir` via `Path(output_dir).mkdir(parents=True, exist_ok=True)` — zero setup friction for callers.

### Tests
- **D-08:** Unit tests in `tests/test_tracking.py` — included in Phase 15, following the Phase 13/14 pattern of same-phase tests.
- **D-09:** Auto-captured fields (`git_hash`, `zreg_version`, `timestamp`) are tested via `unittest.mock.patch` against `subprocess.run` and `importlib.metadata.version` — tests don't depend on repo state or installed package version. Validates all 9 fields are present in both output files.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §EVAL-04 — Full spec: required fields (`run_id`, `dataset_path`, `frame_indices`, `seed`, `n_points_before`, `n_points_after`, `git_hash`, `zreg_version`, `timestamp`), stdlib-only constraint (`csv.DictWriter`, `json.dump`), output location

### Existing eval/ Infrastructure
- `eval/generators/__init__.py` — Pattern for eval subpackage `__init__.py`: re-exports public symbols, defines `__all__`
- `eval/generators/generators.py` — Pattern for a module inside an eval subpackage: public function with NumPy docstring, `__all__`, module-level docstring

### Testing
- `tests/conftest.py` — CUDA skip pattern and fixture conventions; `sys.path.insert(0, repo_root)` is already present (added in Phase 14) so `from eval.tracking import log_run` works in tests without additional changes

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/generators/` package — `eval/tracking/` follows the same structure: `__init__.py` + one or more `.py` modules, no `eval/__init__.py` (namespace directory, established Phase 14)
- `tests/conftest.py` `sys.path.insert` — already covers `eval.*` imports; no change needed for Phase 15 tests

### Established Patterns
- `__all__` in every `__init__.py` — all zreg subpackages and eval subpackages define explicit `__all__`; `eval/tracking/__init__.py` must do the same
- NumPy-style docstrings on all public functions — `log_run()` needs `Parameters`, `Returns`, `Raises` sections
- `Path(output_dir).mkdir(parents=True, exist_ok=True)` — stdlib pathlib; no third-party dependency
- `csv.DictWriter` write pattern: open with `newline=""`, write header (`writeheader()`), then one row (`writerow(record)`) — stdlib only, no pandas
- `json.dump(record, f, indent=2, default=str)` — `default=str` handles non-serializable types (e.g., list of ints for `frame_indices`) gracefully

### Integration Points
- Phase 16 runner scripts will call `from eval.tracking import log_run` — the package must be importable after adding repo root to `sys.path` (already done in `tests/conftest.py`)
- `evaluation/runs/` is the shared output directory for Phase 15 (tracking) and Phase 16 (runners) — `log_run()` auto-creates it so runners don't need to set it up separately
- Optuna trials in Phase 17 (EVAL-06) may pass trial numbers as `run_id` — the caller-provided `run_id: str` design supports this directly

</code_context>

<specifics>
## Specific Ideas

- `frame_indices` in JSON/CSV: stored as a JSON array string (e.g., `"[0, 1, 2]"`) — consistent serialization via `json.dump(record, ...)` for JSON and `str(frame_indices)` for CSV
- Auto-captured `git_hash`: use `subprocess.run(..., capture_output=True, text=True)` — if git is unavailable, fall back to `"unknown"` rather than raising
- Auto-captured `zreg_version`: use `importlib.metadata.version("zreg")` — if package not installed, fall back to `"unknown"`
- Auto-captured `timestamp`: ISO 8601 UTC string from `datetime.datetime.now(datetime.timezone.utc).isoformat()`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 15-Experiment Tracking & Run Management*
*Context gathered: 2026-05-18*
