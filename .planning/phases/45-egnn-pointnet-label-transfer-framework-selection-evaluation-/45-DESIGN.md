# Phase 45 Design: eGNN & PointNet++ Label Transfer — Locked Architecture/Library Decisions

**Status:** Locked. This document is the single authoritative synthesis of `45-CONTEXT.md`
(D-01..D-04) and `45-RESEARCH.md` for Phases 46-49. It contains **no** PointNet++/eGNN model
implementations, training-loop code, or `LabelTransferStage` dispatch edits — those belong to
Phases 47/48 respectively. Illustrative signatures/equations may appear as reference; working
implementations do not.

---

## Locked Library Decision (per model)

Per D-02, the `torch_geometric` (PyG) vs. hand-rolled question is resolved **per model**, not
uniformly:

- **PointNet++ = HAND-ROLLED set-abstraction.** Use Open3D's
  `PointCloud.farthest_point_down_sample` (FPS) and `KDTreeFlann.search_radius_vector_3d` (ball
  query), wrapped exactly the way `src/zreg/registration/icp.py` wraps Open3D's ICP (Phase 39
  precedent: extract positions → build an Open3D structure → call the native op → convert results
  back to `torch.Tensor`). Do **NOT** use `torch_cluster` for this model.
- **eGNN = HAND-ROLLED equivariant convolution** (Satorras et al. 2021, arXiv:2102.09844),
  implemented as a `torch_geometric.nn.conv.MessagePassing` subclass operating over a SPARSE
  radius/kNN graph. Do **NOT** adopt `egnn-pytorch` (dense O(N²) pairwise attention — unfit for
  this repo's ~15k-47.5k-point "realistic" synthetic sizes) and do **NOT** vendor the paper
  authors' `vgsatorras/egnn` GitHub repo (not pip-installable, and vendoring research code is worse
  than hand-rolling the small, well-specified update equations on top of `MessagePassing`).

**Rationale:** RESEARCH.md reproduced the `torch_cluster` build failure **live on this project's
own dev machine** — `pip install torch_cluster` fails at `setup.py egg_info` with the exact
`OMP: Error #15` libomp SIGABRT this codebase already documents workarounds for elsewhere
(`tests/conftest.py:20-24`). `torch_cluster` ships no macOS-ARM wheel (sdist-only, last released
2023-10-12, ~2.75 years stale) and hard-depends on that compiled extension for `fps`/`radius`/
`knn_interpolate` with no pure-PyTorch fallback. This single reproducible, on-machine failure is
decisive against `torch_cluster` for PointNet++. By contrast, the pure-Python `torch_geometric`
core wheel (`torch_geometric-2.8.0-py3-none-any.whl`, no compiled extension) installs cleanly and
its `MessagePassing` base class no longer requires `torch_scatter`/`torch_cluster`/`pyg-lib` for
basic aggregation (modern PyG uses native `torch.scatter_reduce`) — safe scaffolding for the
hand-rolled eGNN math, which is genuinely novel and has no off-the-shelf sparse/scalable
implementation anyway.

Both models are therefore **hand-rolled** at the model-architecture level; the only PyG usage is
as lightweight scaffolding (`MessagePassing`, `MLP`) for eGNN, never for PointNet++'s indexing ops.

## Dependency Decision

- Add `torch_geometric` (core package only) as a declared dependency in **Phase 46/47**, not this
  phase — `setup.cfg` is intentionally untouched by Task 1 of this plan.
- **NEVER** add `torch_cluster`, `torch_scatter`, `torch_sparse`, or `pyg-lib` — all four are
  compiled-extension siblings in the same family as the reproduced build failure; `torch_geometric`
  core needs none of them for the `MessagePassing`/`MLP` scaffolding this design uses.
- `open3d` 0.19.0 is already a hard `install_requires` dependency — no new geometry dependency is
  needed for PointNet++'s FPS/ball-query ops.

**Task 1 install-verification evidence (embedded per plan requirement):**

- **Verification date:** 2026-07-10 (research), re-confirmed 2026-07-11 (this plan's Task 1 execution).
- `pip install torch_geometric` completed with **no build/compile step** — resolved directly to the
  pure-Python wheel `torch_geometric-2.8.0-py3-none-any.whl` (plus its one pure-Python transitive
  dependency, `xxhash`). No `egg_info`/C++ build occurred, unlike `torch_cluster`.
- Observed version: **`torch_geometric.__version__ == "2.8.0"`**.
- `from torch_geometric.nn import MessagePassing, MLP` — **import OK**.
- `git diff --stat setup.cfg` shows **no changes** — `torch_geometric` was left as an
  environment-only install, not a declared dependency, per this task's scope.
- No `torch_cluster`/`torch_scatter`/`torch_sparse`/`pyg-lib` install was attempted.

**Import-order caveat discovered during Task 1 (new finding, not previously documented in
RESEARCH.md — carried forward here for Phase 46/47):** running
`python -c "import torch_geometric; from torch_geometric.nn import MessagePassing, MLP; ..."` in a
**fresh process with no prior import** reproduces the same `OMP: Error #15` libomp SIGABRT this
codebase already works around for `zreg`+`torch` (`tests/conftest.py:20-24`,
`eval/data_factory.py:18-35`, `eval/metrics.py:53-67`, `eval/types.py:48-53`). Importing
`zreg.dataset` (which pulls in `scipy`) **before** `torch_geometric` avoids the conflict, exactly
mirroring the existing "zreg/scipy before torch" convention. Phase 46/47 code that imports
`torch_geometric` MUST follow this same import-order rule: `import zreg...` (or any module that
transitively imports `scipy`) before `import torch_geometric`. This is a Rule 3 (blocking-issue)
finding from this plan's execution, not a new architectural decision — it extends the project's
existing, already-documented workaround to the new dependency.

## Target Module Structure

A new `src/zreg/models/` package, mirroring the existing `src/zreg/registration/` and
`src/zreg/cpd/` per-method-family submodule pattern:

```
src/zreg/models/                 # NEW — created in Phase 47, NOT this phase
├── __init__.py                  # public exports for pointnet2 / egnn entry points
├── _ops.py                      # shared Open3D-wrapped FPS + ball-query + radius-graph builder
├── pointnet2.py                 # hand-rolled set-abstraction + feature-propagation layers
└── egnn.py                      # hand-rolled EGNN conv (torch_geometric.nn.MessagePassing subclass)
```

`src/zreg/color_transfer.py` stays **UNCHANGED** — it holds pure, weightless classical methods
(`nearest_neighbor`, `cpd_weighted`, `knn_voting`, `gaussian_kernel`) and is architecturally
distinct from stateful, learned model code. eGNN/PointNet++ do not belong there.

**Phase attribution:** none of these four files are created in this phase (45). They are created
in **Phase 47**.

## Joint-Cloud Conditioning (required adaptation for BOTH models)

Neither PointNet++ nor eGNN natively accepts "a labeled reference cloud + an unlabeled query
cloud" as two separate inputs — both are natively single-cloud, closed-set segmentation/
classification networks. The required adaptation (RESEARCH Pattern 2), which **both** models must
apply:

1. **Concatenate** source and target point clouds into one joint point set before running the
   network.
2. Give every point an extra input feature channel: **source points carry their one-hot
   ground-truth label**; **target points carry a zero vector (or a learned "unknown" embedding)**.
3. Run set-abstraction (PointNet++) or message-passing (eGNN) over the **joint** cloud, so
   information propagates from labeled source points to unlabeled target points through local
   geometric neighborhoods.
4. Read out per-point class logits at the end, but **only supervise/use the target-point subset**
   of the output — source points are present only to carry signal into the network, not to be
   re-predicted.

A **closed-set output head is legitimate** here because `generate_labels` (`src/zreg/generators/
labels.py`) uses a fixed, small Voronoi-seed vocabulary (`n_classes`), not an open/growing label
space.

**Anti-pattern (flag loudly, do not do this):** training PointNet++/eGNN as single-cloud
segmentation networks that never see the source cloud at inference. This would silently become a
fixed-vocabulary semantic segmenter with no actual transfer mechanism — contrast with
`knn_voting`/`cpd_weighted`, which are both explicitly two-cloud. Always use the joint-cloud
conditioning above.

## label vs id Discipline

Per RESEARCH Pitfall 2: class targets come **EXCLUSIVELY** from `pc["label"]` (the small,
fixed-vocabulary categorical class from `generate_labels`'s Voronoi assignment), **NEVER** from
`pc["id"]` (the monotonically-growing per-cell identity used for ground-truth correspondence in
`DataFactory.get_ground_truth`/`get_synthetic_ground_truth`). This matches the existing
`LabelTransferStage` convention (`eval/stages/label_transfer.py:337`, `labels_tensor =
src_frame.get("label")`).

This binds:
- **Phase 46's** training-triple generation code (`DataFactory` extension) — triples must read
  `pc["label"]`, not `pc["id"]`.
- **Phase 47's** training-target code — loss must be computed against `label`, never `id`.

**Warning sign to watch for:** a training run where the effective number of classes grows over the
trajectory is a symptom of accidentally training against `id` (which increases monotonically as
new cells are born in the growth-model generator).

## Train / Infer Split

Resolves D-04:

- **Training is GPU, offline**, in a new training entry point that lives **OUTSIDE** the
  `eval/` runtime path (Phase 47 — likely `scripts/` or a new `eval/training/`, TBD at that
  phase). Training on CPU is impractically slow even for small synthetic datasets, a deliberate
  divergence from the rest of zReg's otherwise CPU-friendly eval pipeline.
- **Inference must stay CPU-friendly**, matching every other label-transfer method (`knn_voting`,
  `cpd_weighted`), **INSIDE** `eval/stages/label_transfer.py` (Phase 48).
- Phase 48 adds two new `VALID_METHODS` values to `LabelTransferStage`: **`"egnn"`** and
  **`"pointnet++"`** — exactly named as such, via the Phase 44 `OPTIONAL_PARAMS` +
  `EvalConfig.label_transfer_method` precedent (mirrors how `"cpd_weighted"` was added to the
  existing `VALID_METHODS: tuple = ("knn_voting", "cpd_weighted")` tuple in
  `eval/stages/label_transfer.py`). `method` remains an `OPTIONAL_PARAMS` key that defaults to
  `config.label_transfer_method` when absent from `params`, same as today.

## Training Data Source

Resolves D-03: training data comes **solely** from the existing synthetic bowl/ball growth-model
generator (`eval/data_factory.py` / `scripts/generate_datasets.py`) — not from the real annotated
Kobitski/Shah datasets used elsewhere in this project's evaluation framework.

- **Phase 46** extends `DataFactory` to emit `(source cloud+labels, target cloud, target labels)`
  training triples, reusing the generator's existing paired-frame exact-correspondence machinery.
- Kobitski/Shah **stay eval-only**, exactly as they are today for the classical methods — this
  training/eval split is a deliberate continuation of the project's existing convention, not a new
  one.
- **Deferred item (carried forward):** real annotated data may be reconsidered in a later phase
  once the synthetic-only approach is validated — e.g., if Phase 49 evaluation shows
  synthetic-trained models don't generalize well to real Kobitski/Shah data.

## Evaluation Strategy Pointers (Phase 49)

These are **Claude's-Discretion inputs for Phase 49, not final locks** — Phase 49 owns the final
eval-strategy design.

- Reuse `zreg.metrics.compute_f1` and `zreg.metrics.knn_consistency` directly — both are already
  label-tensor-based and method-agnostic (they take label tensors, not classical-method-specific
  objects), so no new bespoke metric functions are needed for the base metrics.
- **Add generalization dimensions** beyond raw synthetic-holdout F1:
  - Held-out synthetic geometry variants (train on `small`, evaluate on `realistic_kobitski`/
    `realistic_shah` point counts and growth trajectories).
  - Qualitative check on real Kobitski/Shah data (no strict F1 ground truth available there, but
    visual/consistency inspection is still valuable).
- **Add CPU inference-latency and point-count robustness** benchmarking, since a GPU-trained model
  can be prohibitively slow at CPU inference time inside `LabelTransferStage`'s per-frame-pair loop
  (RESEARCH Pitfall 4).
- **Explicitly guard against RESEARCH Pitfall 3 (train/eval leakage):** D-03's synthetic-only
  training data is deterministic (fixed seed 42 throughout `DataFactory`/`generate_datasets.py`).
  If Phase 49's evaluation ALSO exclusively uses synthetic bowl/ball data, reported F1/
  knn_consistency scores would reflect memorization of the synthetic geometry family rather than
  genuine label-transfer generalization. The evaluation strategy must include cross-domain and/or
  held-out-variant checks, not just synthetic-training-distribution holdout.

## Downstream Phase Ownership Map

| Decision / Deliverable | Owning Phase | Notes |
|---|---|---|
| Training-triple data generation (`(source+labels, target, target-labels)`) | **46** | Extends `DataFactory`; must read `pc["label"]`, never `pc["id"]` |
| `src/zreg/models/` package (`_ops.py`, `pointnet2.py`, `egnn.py`) + training loop + Wave-0 tests | **47** | GPU training entry point outside `eval/` runtime; benchmarks Open3D-wrapper op performance (Assumption A2) |
| `LabelTransferStage` dispatch (`"egnn"`/`"pointnet++"` in `VALID_METHODS`) + `EvalConfig` checkpoint field + `torch.load(..., weights_only=True)` | **48** | CPU-friendly inference; mirrors Phase 44's `OPTIONAL_PARAMS` precedent |
| Benchmarking / evaluation strategy execution | **49** | Reuses `compute_f1`/`knn_consistency`; adds generalization + CPU-latency dimensions |

## Open Questions Carried Forward

1. **PointNet++ cross-cloud receptive field:** should PointNet++'s ball-query neighborhoods
   (computed over the joint concatenated cloud) also explicitly guarantee inclusion of nearby
   source points, or is the joint-cloud trick alone sufficient given this project's existing
   pre-alignment via `AlignmentStage`? Not a decision to lock now — Phase 47 should treat this as
   an empirical hyperparameter/architecture choice to validate.
2. **Phase 46 point-count regime:** which point-count range(s) will Phase 46 actually target for
   training triples (100-point "small" variant vs. the ~15k-47.5k-point "realistic" variants)? This
   directly affects both the (already-resolved-toward-sparse) eGNN decision and the Open3D-wrapper
   performance question. Phase 46 should explicitly decide and document a target range; Phase 47
   should benchmark the Open3D-wrapper ops at that scale before locking neighborhood sizes.
3. **Phase 47 CUDA-vs-MPS compute target:** D-04 assumes GPU training but does not specify which
   backend. This dev machine has no CUDA GPU, only Apple MPS
   (`torch.backends.mps.is_available() == True`, confirmed both in RESEARCH.md and re-confirmed by
   this plan's Task 1). MPS has historically had incomplete op coverage vs. CUDA. Phase 47 should
   confirm its actual training compute target early and budget time for an MPS forward-pass smoke
   test if MPS is the intended backend.

**Checkpoint-deserialization security note (carried forward for Phase 48):** `torch.load` uses
`pickle` by default and can execute arbitrary code from a malicious `.pt`/`.pth` file. Phase 48's
checkpoint-loading code MUST use `torch.load(..., weights_only=True)` explicitly (rather than
relying on the current PyTorch default), and must only ever load checkpoints produced by this
project's own Phase 47 training step — never externally-sourced `.pt` files, consistent with D-03's
synthetic-only/no-external-data posture. No checkpoint code is written in this phase.

## Downstream Wave-0 Test Obligations

Reproducing RESEARCH.md's Test Map rows, for the phases that will implement them:

| Test file | Behavior tested | Owning Phase |
|---|---|---|
| `tests/test_zreg_models_ops.py` | Open3D FPS/ball-query/radius-graph wrapper returns correct indices/shapes | 47 |
| `tests/test_egnn_equivariance.py` | Rotating input rotates output coordinates without changing output labels (E(3)/SE(3) equivariance) | 47 |
| `tests/test_zreg_models_joint_cloud.py` | Model output shape == target point count, not source+target (joint-cloud conditioning contract) | 47 |
| `tests/test_data_factory_training_triples.py` | Training triples derive class targets from `pc["label"]`, not `pc["id"]` | 46 |

## Source Decisions

This document synthesizes locked decisions D-01, D-02, D-03, D-04 from
`45-egnn-pointnet-label-transfer-framework-selection-evaluation-/45-CONTEXT.md`, and the full
findings of `45-egnn-pointnet-label-transfer-framework-selection-evaluation-/45-RESEARCH.md`
(researched 2026-07-10), for use by Phases 46-49.
