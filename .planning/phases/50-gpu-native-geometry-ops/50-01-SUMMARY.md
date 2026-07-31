---
phase: 50-gpu-native-geometry-ops
plan: 01
subsystem: infra
tags: [torch_cluster, cuda, horeka, gpu]

requires: []
provides:
  - "HoreKa torch_cluster install verification result: FAIL — no pre-built wheel for torch 2.13.0+cu130; source build fails (torch not in build env)"
affects: []

tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - ".planning/phases/50-gpu-native-geometry-ops/50-01-SUMMARY.md"
  modified: []

key-decisions:
  - "RESULT: FAIL — torch_cluster cannot be installed on HoreKa with torch 2.13.0+cu130. Phase 50 closed as a no-op; Open3D fallback unchanged."

patterns-established: []

requirements-completed: []

duration: ~5min
completed: 2026-07-31
---

# Phase 50-01: HoreKa torch_cluster Install Verification — RESULT: FAIL

**No pre-built wheel exists for torch 2.13.0+cu130 on data.pyg.org; source build fails because the pip build subprocess cannot find `torch` — phase 50 closed, Open3D fallback unchanged.**

## Performance

- **Duration:** ~5 min
- **Completed:** 2026-07-31
- **Tasks:** 3 (2 human, 1 auto)
- **Files modified:** 1 (this summary)

## Accomplishments

- Discovered HoreKa environment: torch 2.13.0+cu130, Python 3.12
- Attempted install: `pip install torch_cluster -f https://data.pyg.org/whl/torch-2.13.0+cu130.html`
- Confirmed FAIL: no pre-built `.whl` on the PyG index for this torch/CUDA combination; pip fell back to source build (`torch_cluster-1.6.3.tar.gz`) which failed because `torch` is not available in the isolated pip build subprocess

## Task Results

### Task 1: Identify HoreKa PyTorch and CUDA versions

HoreKa output:
```
torch: 2.13.0+cu130
cuda: 13.0
```

Resolved wheel URL: `https://data.pyg.org/whl/torch-2.13.0+cu130.html`

### Task 2: Install torch_cluster on HoreKa and verify import

Install command attempted:
```
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.13.0+cu130.html
```

Full pip output:
```
Looking in links: https://data.pyg.org/whl/torch-2.13.0+cu130.html
Collecting torch_cluster
  Downloading torch_cluster-1.6.3.tar.gz (54 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error

  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [20 lines of output]
      ...
      ModuleNotFoundError: No module named 'torch'
      [end of output]

ERROR: Failed to build 'torch_cluster' when getting requirements to build wheel
```

**Outcome: FAIL** — pip exited non-zero. The import verification (`python -c "from torch_cluster import fps, radius, radius_graph; print('OK')"`) was not reached.

**Root cause:** `torch_cluster` has no pre-built wheel for torch 2.13.0+cu130 on `data.pyg.org/whl/`. The PyG wheel index only covers released PyTorch versions; torch 2.13.0 appears to be a very recent or pre-release version without a matching `torch_cluster` wheel. The source build fails because pip's isolated build subprocess does not inherit the virtualenv's `torch` installation.

### Task 3: Write 50-01-SUMMARY.md

Written now. RESULT: FAIL documented.

## Files Created/Modified

- `.planning/phases/50-gpu-native-geometry-ops/50-01-SUMMARY.md` — this file; install verification result

## Decisions Made

- Phase 50 is **closed as a no-op**. No code changes are made to `src/zreg/models/_ops.py` or any other source file.
- The existing Open3D CPU fallback in `_ops.py` is unchanged and remains the active implementation.
- `src/zreg/models/_ops.py` was NOT modified (confirmed: this plan modifies no source files).

## Deviations from Plan

None — plan executed exactly as written. FAIL outcome was the anticipated gate result.

## Issues Encountered

- `torch_cluster` 1.6.3 (latest) has no pre-built wheel for torch 2.13.0+cu130. The PyG wheel index (`data.pyg.org/whl/torch-2.13.0+cu130.html`) exists as a URL but contains no matching `.whl` for `torch_cluster`.
- Source build fails at `get_requires_for_build_wheel` step: the pip build isolation creates a clean subprocess without the virtualenv's `torch`, causing `ModuleNotFoundError: No module named 'torch'`.

## Recommendation

Re-evaluate if the HoreKa environment changes:
- **If PyG releases a wheel for torch 2.13.0:** Re-run `pip install torch_cluster -f https://data.pyg.org/whl/torch-2.13.0+cu130.html` — should succeed once the wheel is published.
- **Alternative:** Try `pip install torch_cluster --no-build-isolation -f https://data.pyg.org/whl/torch-2.13.0+cu130.html` — `--no-build-isolation` allows the build subprocess to see the virtualenv's `torch`, enabling source build. This may succeed if a C++ compiler and CUDA toolkit are available on the login node.
- **Alternative:** Downgrade to a torch version with an existing wheel (e.g., 2.3.0+cu121) if the HoreKa environment can be changed.

## Next Phase Readiness

Phase 50 is closed. Plan 50-02 is **skipped** (gate condition not met). The Open3D implementation in `src/zreg/models/_ops.py` is the active path.

---
*Phase: 50-gpu-native-geometry-ops*
*Completed: 2026-07-31*
