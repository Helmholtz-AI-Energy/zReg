# Phase 29: Viz Unification - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Update `eval/viz.py` scatter style to match `scripts/visualize_datasets.py` (style params, subsampling, DPI, fonts, pane fills). Add a new public function `render_dataset_triptych(csv_path, name, output_dir, dpi=150)` to `eval/viz.py` that self-contains all CSV I/O and rendering for a 1×3 triptych PNG. Refactor `scripts/visualize_datasets.py` to import `render_dataset_triptych` from `eval.viz` and remove `_scatter3` and `render_dataset`. Existing public API (`plot_trajectory`, `plot_metrics`) and PDF+PNG output are preserved.

</domain>

<decisions>
## Implementation Decisions

### Subsampling in plot_trajectory

- **D-01:** The 4,000-point cap applies **per cloud independently**. Each separate `scatter()` call in a subplot (e.g., source cloud and aligned cloud) is individually capped at 4K points. Combined per-subplot maximum is ~8K — acceptable with `s=1.5`. Subsampling uses `numpy.random.default_rng(0).choice()` on the `.detach().cpu().numpy()` array (matching the script's pattern).

### Style parameters applied to plot_trajectory

- **D-02:** All of the following style parameters from `scripts/visualize_datasets.py` apply to every 3D subplot in `plot_trajectory`:
  - `s=1.5, alpha=0.45` on scatter calls (replacing current `s=5`)
  - `fontsize=6` on all tick labels (`ax.get_xticklabels()` + y + z)
  - `fontsize=7, labelpad=2` on axis labels (`set_xlabel`, `set_ylabel`, `set_zlabel`)
  - `ax.xaxis.pane.fill = False`, `ax.yaxis.pane.fill = False`, `ax.zaxis.pane.fill = False`
  - DPI=150 added to the PNG `fig.savefig` call (PDF unchanged — vector format, DPI irrelevant)
- **D-03:** The label-transfer subplot colours by label ID — not a single solid colour. The style params (s, alpha, fonts, pane.fill, DPI) apply; the colour per-point comes from the label colormap (unchanged from Phase 25).

### render_dataset_triptych implementation

- **D-04:** `render_dataset_triptych(csv_path, name, output_dir, dpi=150)` **self-contains all I/O** — it chunk-reads the CSV internally using the same pattern as `_frame_indices` + `_load_three_frames` from the script. These helpers become private to `eval/viz.py` (or inlined). The script no longer needs `_frame_indices` or `_load_three_frames` after refactoring.
- **D-05:** **pandas is imported lazily** — `import pandas as pd` placed inside `render_dataset_triptych` (not at module level). This keeps `eval/viz.py` lightweight for callers of `plot_trajectory` and `plot_metrics` who never need CSV reading.
- **D-06:** **Returns `Path`** to the written PNG. Consistent with `export_trajectory` (Phase 24) and `plot_trajectory` (Phase 25) which return path information. The refactored script uses the return value for its print statement, removing the ROOT-relative path construction from the script.
- **D-07:** The scatter colour for `render_dataset_triptych` is hardcoded to `"#2a6496"` (ported from the script). No colour parameter — the roadmap signature `(csv_path, name, output_dir, dpi=150)` is final.
- **D-08:** `output_dir.mkdir(parents=True, exist_ok=True)` is called inside `render_dataset_triptych` (the script currently does this per `out_path.parent`). The function is responsible for creating its output directory.
- **D-09:** Chunk size for CSV reading: `chunksize=60_000`, reading `usecols=["x", "y", "z", "t"]` — ported from `_load_three_frames`. Frame index column is `"t"`.

### Script refactoring

- **D-10:** After refactoring, `scripts/visualize_datasets.py` removes: `_scatter3`, `render_dataset`, `_frame_indices`, `_load_three_frames`. The `collect_datasets`, `main`, and the `matplotlib.use("Agg")` call (which can be removed since `eval/viz.py` uses `rc_context`) are reviewed. `main()` replaces `render_dataset(csv_path, name, out_dir / f"{name}.png")` with `render_dataset_triptych(csv_path, name, out_dir)`.
- **D-11:** `scripts/visualize_datasets.py` must import `render_dataset_triptych` from `eval.viz`. The existing libomp import-order constraint means `eval.viz` (which imports `zreg.*`) must be imported before any `torch` import in the script. The script currently doesn't import torch directly, but it imports matplotlib which is fine. The `from eval.viz import render_dataset_triptych` line goes near the top after stdlib imports.

### FRAME-08 rules (carry-forward from Phase 25)

- **D-12:** All figure code in `eval/viz.py` (including `render_dataset_triptych`) stays inside `matplotlib.rc_context({"backend": "Agg"})`. `plt.close(fig)` called after every `fig.savefig`. `bbox_inches="tight"` on every savefig. These rules are non-negotiable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Files being modified

- `eval/viz.py` — primary target. Functions to update: `plot_trajectory` (scatter style, subsampling, DPI, fonts, pane.fill). Function to add: `render_dataset_triptych`. Update `__all__` to include the new function.
- `scripts/visualize_datasets.py` — secondary target. Functions to remove: `_scatter3`, `render_dataset`, `_frame_indices`, `_load_three_frames`. Import `render_dataset_triptych` from `eval.viz`.

### Style source (copy these exact values)

- `scripts/visualize_datasets.py` — contains the canonical style parameters: `MAX_PTS = 4_000`, `DPI = 150`, `COLOR = "#2a6496"`, scatter call (`s=1.5, alpha=0.45, linewidths=0`), font settings (`fontsize=6` ticks, `fontsize=7` labels, `labelpad=2`), pane fill (`ax.xaxis.pane.fill = False`).

### Tests

- `tests/test_viz.py` — existing tests must pass after scatter style changes. A new smoke test for `render_dataset_triptych` must be added (writes a PNG from a real or synthetic CSV fixture; verifies file exists and is nonzero-size).

### Requirements

- `REQUIREMENTS.md` §VIZ-01 — referenced in ROADMAP.md as the requirement this phase satisfies. VIZ-01 will be formally added to REQUIREMENTS.md during phase verification.

### Prior phase context (viz decisions)

- `.planning/phases/25-visualisation-refactor/25-CONTEXT.md` — Phase 25 decisions for `plot_trajectory` (signature, return type, file structure). D-01 through D-10 all carry forward; this phase adds style parameters on top.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `eval/viz.py:plot_trajectory` (lines ~120–210) — the function receiving scatter style updates. Currently uses `s=5`, no subsampling, no DPI on PNG saves, no pane.fill settings. Frame selection logic (first/mid/last via `sorted_keys`) stays unchanged.
- `scripts/visualize_datasets.py:_scatter3` — the rendering helper to be ported. Its exact implementation (rng, subsampling, scatter params, font settings) is the style target for both `plot_trajectory` and `render_dataset_triptych`.
- `scripts/visualize_datasets.py:_load_three_frames` — chunk-reading logic to move into `render_dataset_triptych`. Uses `pd.read_csv(..., usecols=["x","y","z","t"], chunksize=60_000)`.
- `scripts/visualize_datasets.py:_frame_indices` — reads only `"t"` column via `usecols=["t"]`; returns `sorted(vals.unique())`.

### Established Patterns

- FRAME-08 rule: all figure code inside `matplotlib.rc_context({"backend": "Agg"})` with `plt.close(fig)` after every savefig. Both `render_dataset_triptych` and the updated `plot_trajectory` subplots must follow this.
- Lazy import for optional deps (from CONVENTIONS.md): `import pandas as pd` placed inside `render_dataset_triptych` body, not at module level.
- Return path information from viz functions (established in Phase 25): `plot_trajectory` returns `list[str]`; `render_dataset_triptych` returns `Path`.
- `numpy.random.default_rng(0)` for reproducible subsampling — ported from script's `_scatter3`.

### Integration Points

- `eval/viz.py.__all__` — must be extended: add `"render_dataset_triptych"`.
- `scripts/visualize_datasets.py:main()` — the call site that replaces `render_dataset(csv_path, name, out_dir / f"{name}.png")` with `render_dataset_triptych(csv_path, name, out_dir)`. Print the returned Path for user feedback.
- `tests/test_viz.py` — smoke test for `render_dataset_triptych` needs a CSV fixture (either a real synthetic CSV from `data/synthetic/` if available in CI, or a programmatically generated minimal CSV with columns `x, y, z, t`).

</code_context>

<specifics>
## Specific Ideas

- The script's `_scatter3` is the style canonical: copy its scatter call (`s=1.5, alpha=0.45, c=COLOR, linewidths=0`), its tick font loop, and its pane fill assignments verbatim into the helper used by `plot_trajectory`. For `plot_trajectory`, `c` changes per cloud (blue/orange for alignment; per-label colour for label transfer).
- `render_dataset_triptych` uses `fig.suptitle(name, fontsize=12, fontweight="bold", y=1.01)` and `figsize=(13, 4.2)` — ported from the script's `render_dataset`. These give it the same title and aspect ratio as the old standalone script.
- For the smoke test fixture: a minimal CSV with 3 frames (`t=1,2,3`) and ~50 points each is sufficient. Can be generated with numpy in a pytest fixture — no real data file required in CI.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 29-viz-unification*
*Context gathered: 2026-06-11*
