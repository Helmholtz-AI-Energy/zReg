# Phase 29: Viz Unification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 29-viz-unification
**Areas discussed:** Subsampling scope, pandas import style, Return value

---

## Subsampling scope

| Option | Description | Selected |
|--------|-------------|----------|
| 4K per cloud | Each scatter() call (source + aligned) independently capped at 4K. Up to 8K points visible per subplot. | ✓ |
| 2K each (4K total) | Treat 4K as total per-subplot budget, split 2K per cloud. | |
| You decide | Claude picks — 4K per cloud is the natural port of the script's logic. | |

**User's choice:** 4K per cloud (recommended)
**Notes:** Natural port of the script's `_scatter3` logic where 4K is a per-scatter clarity cap.

---

## pandas import style

| Option | Description | Selected |
|--------|-------------|----------|
| Lazy inside function | `import pandas as pd` inside `render_dataset_triptych` only. Lightweight module load. | ✓ |
| Module-level import | `import pandas as pd` at top of `eval/viz.py`. Simpler but adds load cost for all callers. | |
| try/except guard | Module-level guard following CONVENTIONS.md optional import pattern. | |

**User's choice:** Lazy inside function (recommended)
**Notes:** Keeps `eval/viz.py` lightweight for the common case (callers using `plot_trajectory`/`plot_metrics` never need CSV reading).

---

## Return value

| Option | Description | Selected |
|--------|-------------|----------|
| Path to written PNG | Returns `Path`. Consistent with `export_trajectory` and `plot_trajectory`. | ✓ |
| None | Matches old `render_dataset` behaviour. Function prints its own status line. | |

**User's choice:** Path to written PNG (recommended)
**Notes:** Consistent with the pattern established in Phase 24 (export_trajectory) and Phase 25 (plot_trajectory). Allows the refactored script to use the return value for its print statement.

---

## Claude's Discretion

- **Colour hardcoding:** `render_dataset_triptych` hardcodes `"#2a6496"` — no colour parameter (roadmap signature is `(csv_path, name, output_dir, dpi=150)`).
- **Helper placement:** `_frame_indices` and `_load_three_frames` move into `eval/viz.py` as private helpers (or inlined) — natural consequence of `render_dataset_triptych` owning its I/O.
- **Script cleanup depth:** `_frame_indices` and `_load_three_frames` are also removed from the script (not just `_scatter3` and `render_dataset`), since `render_dataset_triptych` now handles all CSV I/O.

## Deferred Ideas

None — discussion stayed within phase scope.
