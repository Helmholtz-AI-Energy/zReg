# Phase 15: Experiment Tracking & Run Management - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 15-experiment-tracking-run-management
**Areas discussed:** API design, run_id strategy, CSV layout, Tests in this phase

---

## API Design

| Option | Description | Selected |
|--------|-------------|----------|
| Plain functions | `log_run(...)` — stateless, no class. Phase 16 runners just import and call. | ✓ |
| RunLogger class | `RunLogger(output_dir)` with `.log(...)` — instantiated once per script, reused across a sweep. | |

**User's choice:** Plain functions

---

| Option | Description | Selected |
|--------|-------------|----------|
| Return the run_id | Makes it easy for runner scripts to reference the ID that was logged. | ✓ |
| Return None | Fire-and-forget. Callers already know the run_id they passed in. | |

**User's choice:** Return the run_id

---

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-capture all three | `git_hash` via subprocess, `zreg_version` via importlib.metadata, `timestamp` via datetime — internal to tracking module. | ✓ |
| Caller supplies all fields | All 9 fields explicit in the signature. No hidden subprocess calls. | |

**User's choice:** Auto-capture git_hash, zreg_version, and timestamp internally

---

## run_id Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Caller-provided | Required `str` parameter — runner scripts control naming. | ✓ |
| Auto-generated if not given | `run_id: str \| None = None`, generate `uuid.uuid4().hex[:8]` when None. | |

**User's choice:** Caller-provided required string

---

## CSV Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Append-mode shared CSV | One `evaluation/runs/runs.csv` — each call appends a row. | |
| Per-run CSV alongside JSON | `{run_id}.json` + `{run_id}.csv` per run — single-row CSV per run. | ✓ |

**User's choice:** Per-run CSV alongside JSON

---

| Option | Description | Selected |
|--------|-------------|----------|
| evaluation/runs/ | Matches EVAL-05 path for runner output. | ✓ |
| Caller-configured path | `output_dir` parameter controls everything; `evaluation/runs/` is just a default. | |

**User's choice:** `evaluation/runs/` as default

---

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-create with mkdir -p | `Path(output_dir).mkdir(parents=True, exist_ok=True)` before writing. | ✓ |
| Require directory to exist | Raises `FileNotFoundError` if missing — caller is responsible. | |

**User's choice:** Auto-create

---

## Tests in This Phase

| Option | Description | Selected |
|--------|-------------|----------|
| Include tests (Phase 15) | `tests/test_tracking.py` — same-phase, following Phase 13/14 pattern. | ✓ |
| Defer to Phase 16 | Tests added alongside runner tests in Phase 16. | |

**User's choice:** Include in Phase 15

---

| Option | Description | Selected |
|--------|-------------|----------|
| Mock subprocess + importlib | `unittest.mock.patch` — tests don't depend on git state or installed version. | ✓ |
| Test with real values | Assert git_hash is 40-char hex, version matches installed package. | |

**User's choice:** Mock subprocess and importlib.metadata

---

## Claude's Discretion

None — all gray areas were explicitly decided by the user.

## Deferred Ideas

None — discussion stayed within phase scope.
