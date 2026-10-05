# Roadmap: zReg

## Milestones

- 🚧 **v1.8 Code Review Remediation** — Phases 59–64 (in progress)
- ✅ **v1.7 Minor Adjustments** — Phases 57–58 (shipped 2026-08-04)
- ✅ **v1.6 HoreKa Cluster Execution** — Phases 51–54 (shipped 2026-07-23)
- ✅ **v1.5 Learned Label Transfer Methods** — Phases 44–50 (shipped 2026-07-31)
- ✅ **v1.4 Trajectory Alignment & Optimization Enhancements** — Phases 39–43 (shipped 2026-07-08) — [archive](.planning/milestones/v1.4-ROADMAP.md)
- ✅ **v1.2 Evaluation Framework & Debt Resolution** — Phases 12–38 (shipped 2026-06-26) — [archive](.planning/milestones/v1.2-ROADMAP.md)
- ✅ **v1.1 Code Quality & Refactoring** — Phases 6–11.1 (shipped 2026-05-13) — [archive](.planning/milestones/v1.1-ROADMAP.md)
- ✅ **v1.0 Consolidation** — Phases 1–5 (shipped 2026-04-09) — [archive](.planning/milestones/v1.0-ROADMAP.md)

## Phases

**Phase Numbering:**

- Integer phases (59–64): Active milestone v1.8; 44–58 are complete
- Decimal phases (e.g. 44.1): Urgent insertions (marked with INSERTED)
- Note: Phase 55 was already consumed by an ad-hoc out-of-band phase (spherical-cap/Gaussian label generators for `zreg.data_generation.labels`, unrelated to any numbered milestone), completed 2026-07-30. Phase 56 was also already consumed — by a separate ad-hoc phase developed concurrently on `feature/evaluation_framework` (Configurable multi-label region-based labeling, completed 2026-07-31) before this milestone's branch merged back in. v1.7 was originally planned as Phases 56–57, but was renumbered to 57–58 on merge to resolve that collision — see Phase Details below for the corresponding `.planning/phases/` directory renames.

### 🚧 v1.8 Code Review Remediation (In Progress)

**Milestone Goal:** Every finding of the 8-unit external code review (`.planning/reviews/`, target `feature/evaluation_framework` @ `6c1c37f`, 2026-09-17) is fixed and pinned by a contract-level regression test that fails on `6c1c37f`, passes after the fix, and does not mock the unit under test.

- [x] **Phase 59: P0 Runnability & Silent Number Corruption** - HoreKa configs and scripts run again; ICP/SWD round-trips, HPO scoring and paired transfer direction are correct; degenerate outcomes fail loudly instead of scoring 1.0
- [x] **Phase 60: CPD/DTW Numerics** - CPD convergence, `q` and `sigma2` follow Myronenko & Song; identity defaults, warm start, DTW save/load and stale paths fixed
- [x] **Phase 61: Distances, Aligners & MPI** - Pairwise matrix never deadlocks; SW variants are dtype/device-correct; SWD/ICP return proper rotations or reject degenerate clouds; one `cdist` per frame
- [x] **Phase 62: Data, Label Transfer & GPU Path** - `device: cuda` runs end-to-end; label configs validated at config time; label-transfer weighting NaN-safe and correctly indexed
- [x] **Phase 63: HPC Orchestration, Viz/Export & Docs** - Warm start, resume and shared output dir work on HoreKa; figures/metadata complete and consistent; restructure leftovers cleaned
- [x] **Phase 64: Tech debt cleanup: doctests, stale refs, validation bookkeeping** - Whole-package doctests pass (pytest and the `tox -e doctests` Sphinx builder), no stale module/file references remain, and the v1.8 planning record matches reality

### ✅ v1.7 Minor Adjustments (Complete)

**Milestone Goal:** General-purpose catch-all milestone for small, unrelated fixes and additions that don't warrant their own themed milestone — fixing the ground-truth field mismatch in the evaluation framework and adding a synthetic labeled subsample-pair generation mechanism. (v1.6 HoreKa work was believed paused at Phase 54 when this milestone started, but turned out to already be complete — see v1.6 section below.)

- [x] **Phase 57: Ground-Truth Field Consistency** - Label-transfer F1 ground truth is drawn from the same field `LabelTransferStage` actually transfers, and stays correctly paired even when `transform_spec` changes point counts (completed 2026-08-04)
- [x] **Phase 58: Synthetic Labeled Subsample-Pair Generation** - Label-transfer HPO can run against a config-driven synthetic subsample-pair ground truth mechanism, alongside the existing `transform_spec` mechanism (completed 2026-08-04)

### ✅ v1.6 HoreKa Cluster Execution (Complete)

**Milestone Goal:** Run the `baseline_experiments` evaluation suite on the HoreKa HPC cluster with GPU support, replacing the current laptop-constrained (CPU-only, subsampled) execution, within a 3-hour GPU time budget.

- [x] **Phase 51: Environment & Access** - Operator can activate the HoreKa environment, transfer real datasets, and submit a working end-to-end job script *(1 plan — ready to execute)* (completed 2026-07-15)
- [x] **Phase 52: Multi-Rank Parallelism & Validation** - HPO trials run concurrently across MPI ranks via propulate, orchestration stays single-writer, validated on a short test job (completed)
- [x] **Phase 53: GPU Acceleration** - Real per-operation GPU acceleration threaded through EvalConfig, DataFactory, and AlignmentStage (completed)
- [x] **Phase 54: Budget Calibration & Full-Suite Gate** - Full 7-run suite is calibrated and verified to fit the 3-hour GPU cap before the full allocation is submitted (completed 2026-07-23, UAT 9/9 passed — see `.planning/phases/54-budget-calibration-full-suite-gate/54-UAT.md`; this milestone's completion had drifted out of STATE.md/ROADMAP.md tracking after the branch that completed it diverged from `feature/evaluation_framework` before merging back — reconciled 2026-08-05)

## Phase Details

### Phase 59: P0 Runnability & Silent Number Corruption

**Goal**: The HoreKa suite and example scripts start again, and no aligner, HPO path or label-transfer direction produces a silently wrong number. Degenerate results show up as failures, not as perfect scores.
**Depends on**: Nothing (first phase of v1.8; baseline is `6c1c37f`)
**Requirements**: RUN-01, RUN-02, NUM-01, NUM-02, NUM-03, NUM-04, NUM-05
**Success Criteria** (what must be TRUE):

  1. Every YAML under `configs/`, `configs_horeka/` and `baseline_experiments/configs*/` loads through `EvalConfig.from_yaml` (enforced by one test that globs them all). `generate_real_previews.py`, `label_transfer_example.py`, `example_plots.py` and `dtw_testing.py` import without error, and `from zreg.dtw import X` works.
  2. An identity ICP or SWD registration returns its input coordinates unchanged (for example, `[3,5,7]` stays `[3,5,7]` and does not become `[-1,1,3]`). Denormalised results use the bounds of the frame they live in. A paired-mode run transfers Shah (target, real germ-layer labels) labels onto the aligned Kobitski source, not Kobitski's arbitrary colour indices onto Shah (restores `c60a943` semantics).
  3. In multi-seed subsample-pair HPO, a deliberately misaligned trial scores worse chamfer/hausdorff than an aligned one, because the aligned per-frame dicts are scored. A list-valued `seed` runs sanity and dev tiers on `seed[0]`, and those tiers no longer swallow `ValueError`/`TypeError`.
  4. Degenerate outcomes cannot normalise to `1.0`. An empty frame-key intersection returns a non-finite value or raises, partial overlap is flagged, and a raising trial is recorded as failed. `sanity_check` reports each case.
  5. Each fix has a regression test that fails on `6c1c37f`, passes after the fix, and does not mock the unit under test. In particular, the HPO scoring tests call the real `compute_stage_metrics` and `_score_subsample_pair_multiseed`.

**Plans:** 8 plans (revised 2026-10-02 for cross-AI review 59-REVIEWS.md)

Plans:
**Wave 1**

- [x] 59-01-PLAN.md — Restore `EvalConfig.label_source`, reject duplicate YAML keys, glob-load all 84 configs; shared label-direction helper; all 27 Kobitski->Shah configs take labels from Shah, no exemption (RUN-01, NUM-04) [wave 1]
- [x] 59-02-PLAN.md — `zreg.dtw` sys.modules shim; repair 4 scripts to current module paths, import-safe, main() smoke tests + runtime fixes, no xfail escape hatch (library blockers fixed in-phase) (RUN-02) [wave 1] (revised for cycle 2 review)
- [x] 59-03-PLAN.md — Shared normalisation helper with exact D_inv; ICP/SWD shared bounds; ICP CUDA-safe; seeded SWD tests (NUM-01) [wave 1]
- [x] 59-04-PLAN.md — Frame-averaged chamfer/hausdorff: empty -> inf, partial/degenerate flagged via StageMetrics.coverage_flags + sanity_check; direct FrameAverage tests (NUM-05 metrics) [wave 1]

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 59-05-PLAN.md — Eval runner follows label provider/receiver (runner, export, viz); F1-unavailable flag; disabled alignment reported as inf + "stage unavailable"; disabled-stage fixtures keep one stage enabled (NUM-04 runner, NUM-05 runner) [wave 2] (revised for cycle 2 review)
- [x] 59-06-PLAN.md — Multiseed HPO scores aligned dicts (averaged objective asserted); list seed runs every tier on seed[0]; `_objective` honours label_source (NUM-02, NUM-03, NUM-04 HPO) [wave 2]

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 59-08-PLAN.md — cpd_weighted supports label_source="target" via the row-normalised CPD posterior; runner tests cover rigid and the real.yaml non-rigid settings (NUM-04) [wave 3] (revised for cycle 2 review)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 59-07-PLAN.md — Failed trials recorded, -inf, per-run counter reset; failures AND successful histories of all MPI ranks merged on rank 0 (gather + bcast, finite-score filter, explicit Propulate contract), failed_trials.json, raise only on empty global history; full-suite gate (NUM-05 optimizer) [wave 4] (revised for cycle 2 review)

**Note**: NUM-05 changes failure semantics (`0.0` becomes non-finite or raises). Phases 60–63 build on this, so later regression tests can assert loud failure.

### Phase 60: CPD/DTW Numerics

**Goal**: CPD registration follows Myronenko & Song (convergence, `q`, `sigma2`), starts from honest defaults, and every CPD/DTW variant either works end-to-end or rejects unsupported input clearly.
**Depends on**: Phase 59
**Requirements**: CPD-01, CPD-02, CPD-03, CPD-04, CPD-05, CPD-06, CPD-07, CPD-08, CPD-09
**Success Criteria** (what must be TRUE):

  1. Scaling the input by 0.1% leaves CPD `n_iters` essentially unchanged; the bug at `6c1c37f` flips it from 12 to 1. Convergence is only tested after four real `q` values exist.
  2. Rigid CPD `q` matches the affine form, so it is non-negative-consistent as a DTW cost. Fixed-scale `sigma2` matches Eq. 23 (`tr_xp1x − 2·tr_atr + tr_yp1y`) on a hand-computed reference, and converged `sigma2` is near the true residual (not 1.33 against about 1e-4).
  3. A fresh `RigidCPD` and a default `AffineTransformation()` start from the identity; the dataset-specific pose applies only through `tf_init_params`. `init_cpd_from_existing` warm-starts `AffineCPD` from the given transform. `ConstrainedNonRigidCPD` exposes `self.transformation` like `NonRigidCPD`.
  4. DTW `save()`/`load()` round-trips a config whose `distance_metric` is a callable. `use_color=True` either works end-to-end or raises a clear error. Non-rigid CPD accepts sources wider than 3 columns. `rbf_kernel_matrix` matches CPD's kernel (or is gone). The TPS docstring example runs, and stale module paths in `cpd/`, `dtw/`, `core/transforms/` and `eval/stages/alignment.py` are updated.
  5. Each fix has a regression test that fails on `6c1c37f`, passes after the fix, and runs the real CPD/DTW code without mocks.

**Plans**: 5 plans (wave 1: 60-01, 60-02, 60-03 in parallel; wave 2: 60-04; wave 3: 60-05)

Plans:
**Wave 1**

- [x] 60-01-PLAN.md — EM numerics: rigid sigma2/q (CPD-02/03), 4-value convergence window (CPD-01), identity RigidCPD + opt-in SHAH_KOBITSKI_EMPIRICAL_INIT (CPD-08), target_colors guard (CPD-05)
- [x] 60-02-PLAN.md — Variants: identity AffineTransformation + AffineCPD warm start (CPD-06), ConstrainedNonRigidCPD.transformation (CPD-04), use_color rejection (CPD-05), wide sources / rbf_kernel_matrix / transform docstrings (CPD-09)
- [x] 60-03-PLAN.md — DTW: sigma2 cpd cost via _cpd_dtw_cost (CPD-02/D-02), callable-metric save/load (CPD-07), stale paths + CR-01 comment (CPD-09)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 60-04-PLAN.md — Wide sources: xyz-only E-step, shape/rank checks for all CPD variants (CPD-09)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 60-05-PLAN.md — Integration gate: rigid cpd DTW end-to-end, alignment fallback identity, strict full suite, D-02/D-04 decision records

**Note**: CPD-02 changes the `q` that `distance_metric="cpd"` uses as DTW cost. Phase 61's DIST-03 (cpd-branch handling in the pairwise sweep) should run after or alongside this phase so its tests assert the corrected `q`.

### Phase 61: Distances, Aligners & MPI

**Goal**: Pairwise distance computation never deadlocks and works for any dtype, device and frame-key layout. The SWD/ICP aligners return valid rotations or reject degenerate input. HPO metrics cost one `cdist` per frame.
**Depends on**: Phase 59 (NUM-01's `D_inv` fix and NUM-05's failure semantics; DIST-04 edits the same `icp.py`/`swd_aligner.py`)
**Requirements**: DIST-01, DIST-02, DIST-03, DIST-04, DIST-05
**Success Criteria** (what must be TRUE):

  1. The pairwise distance matrix completes, under a timeout, on both code paths when some rank owns no pair in a row. This is checked by a multi-rank run or a rank-simulating test.
  2. MaxSWD, GSWD and PSWD work on float64 and on CUDA inputs, keeping projections on the input dtype/device. Changing `max_sw_num_iters`/`max_sw_lr` measurably changes the optimisation.
  3. A pairwise sweep over non-zero-based frame keys produces the right matrix. The cpd branch with `fn is None` and a leading-`None` `distance_kwargs` behave consistently.
  4. The SWD aligner registers clouds with unequal point counts and returns a rotation with `det=+1` that is orthogonal at return. ICP and SWD raise an informative error on empty or degenerate clouds.
  5. Frame-averaged chamfer/hausdorff gives the same values as before but uses one `cdist` per frame. Each fix has a regression test that fails on `6c1c37f` and does not mock the unit under test.

**Plans**: 4 plans

Plans:
- [x] 61-01-PLAN.md — SW variants follow input dtype/device, MaxSWD honours its kwargs without gradient leak, unequal-N quantile coupling (legacy max(N, M) scaling), empty-set guard (DIST-02, DIST-04)
- [x] 61-02-PLAN.md — ICP/SWD reject empty/coincident/non-finite/mis-shaped clouds via registration_bounds; SWD projects to SO(3) every step and at return (DIST-04)
- [x] 61-03-PLAN.md — MPI-safe per-row allgather, positional key sweep, kwargs normalisation, cpd guard; runs only after Phase 60-03 has landed (DIST-01, DIST-03)
- [x] 61-04-PLAN.md — chamfer and hausdorff from one cdist per frame with a single batched transfer (DIST-05)

### Phase 62: Data, Label Transfer & GPU Path

**Goal**: A `device: cuda` run works end-to-end. Data augmentation and label configs keep correspondences and labels valid. Every label-transfer method (kNN, Gaussian, CPD `pmat`, model-based) and label metric gives finite, correctly indexed results.
**Depends on**: Phase 59
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, LT-01, LT-02, LT-03, LT-04
**Success Criteria** (what must be TRUE):

  1. A run with `device: cuda` gets through `get_synthetic_ground_truth` gather and label-id indexing without a device-mismatch error, because buffers are created on the input device.
  2. After `augment()` with outliers and dropout, correspondence indices still point at matching points. Empty `label_specs`, `n_labels: 0` and an unknown `mode` fail when the config is built. The `subsample_pair` synthesize path uses `config.label_generation`. The RGB-to-index remap considers all non-empty frames.
  3. Model-based transfer takes `n_classes` from the model and accepts 1-D label tensors. Gaussian-kernel and CPD-`pmat` weighting stay NaN-free on zero row sums. The transposed-`pmat` guard is correct when `n_source == n_target`. A missing `source["label"]` raises clearly, and the caller's input is not mutated.
  4. `sample_bowl` rejects an invalid `d_ratio`/`radius` instead of hanging. `knn_consistency` excludes the query point by index. `temporal_stability` raises the documented exception type. WR-01 truncation passes consistent `labels_for_knn`/positions in `_objective` and the multi-seed path.
  5. Each fix has a regression test that fails on `6c1c37f` and does not mock the unit under test; the model-based path runs a real (tiny) model. CUDA-only tests are marked to skip without a GPU and must be run once on a CUDA host (for example HoreKa).

**Plans**: 6 plans

Plans:
- [x] 62-01-PLAN.md — DataFactory: device-following GT gather, outlier correspondence tracking, LabelGenerationConfig constraints, synthesize honours label_generation, drop source_corr (DATA-01..04, wave 1)
- [x] 62-02-PLAN.md — zreg data generation & loader: label ids on pos device, generate_labels mode/empty-spec guards, sample_bowl validation + d_ratio bound + total candidate budget, canonical colour normalisation + RGB remap over non-empty frames with long labels in every frame (DATA-01, DATA-03, DATA-04, LT-03, wave 1)
- [x] 62-03-PLAN.md — Library label transfer: pmat_layout contract, one shared pmat zero-row policy (repair_pmat_rows: input validation, zero-receiver contract, NN fallback, -1 for non-finite positions, 50% bound; IN-09a/c), softmax Gaussian, label None-check, model.n_classes + 1-D labels with real tiny models, stage CUDA hardening (LT-01, LT-02, wave 1)
- [x] 62-04-PLAN.md — Label metrics: knn_consistency self-exclusion by index, temporal_stability TypeError (LT-04, wave 1)
- [x] 62-05-PLAN.md — Optimizer: wave-1 gate, consistent kNN inputs (U5-3 WR-01), cpd_weighted align_result wiring in HPO, Trial.flags incl. non-empty multiseed/MPI tests (IN-09b) (LT-02, LT-04, wave 2)
- [x] 62-06-PLAN.md — Optimizer Propulate path keeps real Trial payloads (metrics + flags, IN-09b), flagged placeholders for checkpoint-restored individuals; fully green full-suite gate + HoreKa CUDA/real-Propulate test list (all phase reqs, wave 3)

### Phase 63: HPC Orchestration, Viz/Export & Docs

**Goal**: HoreKa orchestration does what its scripts claim (warm start, resume, one output directory), exported figures and metadata are complete and consistent, and the restructure's leftover stale references are gone.
**Depends on**: Phase 59 (RUN-01 makes the HoreKa configs loadable)
**Requirements**: HPC-01, HPC-02, HPC-03, HPC-04, VIZ-01, VIZ-02, VIZ-03, DOC-01
**Success Criteria** (what must be TRUE):

  1. In `baseline_with_combined`, the first HPO trial evaluates the injected combined parameters; where the active search strategy can't, it fails loudly. `merge_combined_params` returns only calibrated values, with no defaults dilution.
  2. Resubmitting the HoreKa baseline job resumes HPO from existing checkpoints. All MPI ranks write into one timestamped output directory chosen by rank 0. Eval jobs run a single rank. The SLURM log directory exists before submission, and `test_propulate_interactive.sh` guards against an empty directory.
  3. Saved superposed-trajectory and label-trajectory PDFs/PNGs contain their legends, and dataset-preview titles are not clipped. With `step: 2`, the source panel shows the same frames as the aligned panel. `label_metadata.json` frame count and indices match `label_trajectory.csv`.
  4. `tox -e doctests` passes. No docstring references `zreg.metrics`, `zreg.distances` or `zreg.dataset`. The `eval_runner.py` frame-convention docstrings agree, and the `pyproject.toml` coverage `omit` points at an existing path.
  5. Each fix has a regression test that fails on `6c1c37f` and does not mock the unit under test. Shell and SLURM fixes get a test that runs the script logic (for example in dry-run or with a stubbed `sbatch` binary), not just a grep.

**Plans**: 9 plans

Plans:
- [x] 63-01-PLAN.md — `merge_combined_params` averages only calibrated values per regime, with no defaults dilution (HPC-02)
- [x] 63-02-PLAN.md — Exported figures keep their legends and unclipped titles; source panel mirrors the aligned panel's `step`; label metadata matches the CSV (VIZ-01, VIZ-02)
- [x] 63-03-PLAN.md — HoreKa launchers resume HPO by default (`ZREG_CLEAR_CHECKPOINTS=1` opt-in clear) (HPC-03, HPC-04)
- [x] 63-04-PLAN.md — Eval jobs run a single rank, SLURM log directory created before submission, interactive Propulate test guards an empty directory (HPC-04)
- [x] 63-05-PLAN.md — All MPI ranks share rank 0's timestamped output directory and only rank 0 runs `EvaluationRunner` (HPC-04)
- [x] 63-06-PLAN.md — Config consistency: eval-side `dtw_dist_fn` validator (no non-DTW `cosine` searches), F1 weight 0.0 where F1 is unavailable by construction (HPC-01)
- [x] 63-07-PLAN.md — `HyperparamOptimizer` consumes `config.default_params` and the combined warm start; Propulate warm-start seeds evaluated as recorded trials (D-09, HPC-01)
- [x] 63-08-PLAN.md — Strict JSON for non-finite metrics (null + marker) instead of literal `Infinity` (VIZ-03)
- [x] 63-09-PLAN.md — Restructure leftovers: stale `zreg.metrics/distances/dataset/transforms` refs, frame-convention docstrings, coverage `omit` path, doctest gate (DOC-01)

### Phase 64: Tech debt cleanup: doctests, stale refs, validation bookkeeping

**Goal:** DOC-01's "doctests pass" holds for the whole `zreg` package (pytest `--doctest-modules src/zreg` and the `tox -e doctests` Sphinx builder), no stale module/file references remain, and the v1.8 planning record (ROADMAP, SUMMARY frontmatter, VALIDATION/UAT, milestone audit) matches reality. No registration/HPO numerics change.
**Requirements**: DOC-01 (residual), v1.8 audit tech debt (63 items 1-2, planning-bookkeeping items 1-3)
**Depends on:** Phase 63
**Success Criteria** (what must be TRUE):

  1. `pytest --doctest-modules src/zreg` reports 0 failed on a CPU-only host; the config doctest no longer leaks a CUDA default device or matmul precision; the Open3D example is guarded inside its docstring (`# doctest: +SKIP`, honoured by both pytest and Sphinx) and executed by a test wherever Open3D imports.
  2. Every docstring example runs with an empty namespace (Sphinx-equivalent); `sphinx-build -b doctest` (the tox doctests command) reports 0 failures with and without Open3D, and a real `tox -e doctests` run is attempted (if tox cannot install, the record says "passed via equivalent sphinx-build command" and never claims a tox run); `tests/test_doctests.py` pins the pytest and empty-namespace gates.
  3. The `id=None` downsampling crash exposed by the DTW example is fixed with a regression test that fails on the pre-fix code.
  4. No stale slash-path or dotted references (`src/zreg/metrics|transforms|registration|generators/...`, `zreg.generators`) remain; `tests/test_doc_hygiene.py` checks them.
  5. ROADMAP, all 59-63 SUMMARY `requirements-completed` fields, 63-HUMAN-UAT, 63-VERIFICATION and every enumerated stale location of the v1.8 audit are updated; `/gsd:validate-phase 62` and `63` are run by the orchestrator at the 64-04 checkpoint, and the audit's Nyquist block/status flip to compliant/passed (64-05) only if both VALIDATION.md files then show `nyquist_compliant: true` with no pending rows; otherwise the gap stays recorded as open debt.

**Plans:** 5 plans

Plans:

**Wave 1**
- [x] 64-01-PLAN.md — `id=None` downsampling fix (RED-first, both random branches), self-contained imports for import-only docstrings, stale refs in models/_ops.py and eval/data_factory.py

**Wave 2** *(depends on 64-01)*
- [x] 64-02-PLAN.md — config.py leak (restores saved device/precision), 9 genuine doctest fixes, utils/generators imports, data_generation stale refs, Open3D example guarded inline with `+SKIP`

**Wave 3** *(depends on 64-01, 64-02)*
- [x] 64-03-PLAN.md — tests/test_doctests.py gate (subprocess literal + guarded empty-namespace runner + Open3D example runner), hygiene extension, Sphinx evidence with/without Open3D, time-boxed real `tox -e doctests`, full suite

**Wave 4** *(depends on 64-01..03; non-autonomous)*
- [x] 64-04-PLAN.md — Bookkeeping: SUMMARY requirements-completed, ROADMAP 59-63, 63 UAT/VERIFICATION, enumerated audit refresh; ORCHESTRATOR POST-STEP: `/gsd:validate-phase 62` and `63`

**Wave 5** *(depends on 64-04)*
- [x] 64-05-PLAN.md — Gated audit Nyquist/status finalization, Phase 64 ROADMAP/STATE completion, 64-VALIDATION, consistency check

---

### Phase 57: Ground-Truth Field Consistency

**Goal**: Label-transfer F1 scoring is computed against the correct ground-truth field (`pc["label"]`, the field `LabelTransferStage` actually transfers) with correct per-point correspondence preserved even when `transform_spec`'s `dropout_fraction`/`n_new_points` change point counts between source and target, and existing ground-truth configs are audited/updated to the corrected convention.
**Depends on**: Nothing (first phase of v1.7)
**Requirements**: GT-01, GT-02, GT-03
**Success Criteria** (what must be TRUE):

  1. `DataFactory.get_ground_truth()` and `get_synthetic_ground_truth()` read `pc["label"]` (not `pc["id"]`) for both `pipeline_mode: paired` and `pipeline_mode: synthetic`, matching the field `LabelTransferStage` actually transfers
  2. When `transform_spec`'s `dropout_fraction`/`n_new_points` cause source and target point counts to diverge, `eval_runner._run_single`'s `y_true`/`y_pred` pairing preserves correct per-point correspondence instead of truncating both arrays to `min(len(y_true), len(y_pred))` by position
  3. Existing ground-truth configs (`baseline_experiments/configs/ground_truth/shah_sample1.yaml`, `kobitski_ew06.yaml`, and `configs/experiments/stage2_label_transfer/*/synthetic.yaml`) are audited and updated so label-transfer F1 remains meaningful under the corrected GT-field convention
  4. A label-transfer evaluation run against a `transform_spec` config with nonzero `dropout_fraction`/`n_new_points` produces a non-degenerate F1 score (not silently near-zero or spuriously perfect from misaligned arrays)

**Plans**: 2 plans

Plans:

- [x] 57-01-PLAN.md — EvalConfig.ground_truth_field + DataFactory correspondence-tracking (drop_points/sample_new_points) + GT extraction rewrite + eval_runner WR-01 re-scoping (GT-01, GT-02)
- [x] 57-02-PLAN.md — Ground-truth config audit: Shah/Kobitski comment fixes + degenerate-label documentation on stage2_label_transfer synthetic.yaml configs (GT-03)

### Phase 58: Synthetic Labeled Subsample-Pair Generation

**Goal**: Users can configure a synthetic evaluation dataset pair generated by subsampling a larger labeled point cloud into two views (source/target) with known per-point correspondence, and run label-transfer HPO against it — a sibling mechanism to the existing known-transform (`transform_spec: rigid`/`noise`) path.
**Depends on**: Phase 57
**Requirements**: GT-04, GT-05, GT-06
**Success Criteria** (what must be TRUE):

  1. User can specify, via YAML config, a synthetic dataset pair generated by subsampling a larger labeled point cloud into two views (source/target) with known per-point correspondence
  2. `DataFactory` exposes a method that generates/retrieves such labeled subsample pairs, reusing `generate_training_triple()`'s geometry -> `generate_labels()` -> `generate_target()` composition pattern
  3. `EvaluationRunner` and `HyperparamOptimizer` can run a label-transfer HPO sweep end-to-end against subsample-pair-generated ground truth (not just the training pipeline that originally consumed `generate_training_triple()`)
  4. The subsample-pair mechanism coexists with the existing `transform_spec` known-transform mechanism — both remain independently selectable via config without one breaking the other

**Plans**: 4 plans

Plans:

**Wave 1**

- [x] 58-01-PLAN.md — DataFactory.generate_subsample_pair() + _subsample_source_view attribute + config docstring (GT-04, GT-05)

**Wave 2** *(both depend on 57-01, no file overlap between them)*

- [x] 58-02-PLAN.md — EvaluationRunner.run() subsample_pair dispatch branch (GT-04, GT-06)
- [x] 58-03-PLAN.md — HyperparamOptimizer single-seed subsample_pair wiring across run()/_tier_dataset()/_objective() (GT-06)

**Wave 3** *(depends on 57-03, same file)*

- [x] 58-04-PLAN.md — D-07 opt-in multi-seed averaging, gated to the full tier (GT-04, GT-06)

### Phase 51: Environment & Access

**Goal**: Operator can stand up a working HoreKa environment — Python + MPI-built dependencies, real data transferred, and a SLURM job script that launches the suite end-to-end with outputs landing in the correct directory convention.
**Depends on**: Nothing (first phase of v1.5)
**Requirements**: ENV-01, ENV-02, ENV-03, OUT-01
**Success Criteria** (what must be TRUE):

  1. Operator can activate a Python environment on HoreKa with zReg plus the `propulate`/`mpi4py` extras installed, and `mpi4py` is built against HoreKa's system MPI (verified via a successful import on a compute node)
  2. Operator can transfer the real datasets (Kobitski tracklets, Shah CSV) to HoreKa, and every existing eval config's `data_path`/`target_data_path` resolves without a file-not-found error
  3. Operator can submit a SLURM job script that requests GPU nodes/ranks and launches the experiment suite end-to-end (job runs to completion or a defined stopping point, not a script/config error)
  4. Suite outputs land in the same `baseline_experiments/experiments/<phase>/<name>/` directory convention as local runs, hosted on HoreKa's workspace filesystem

**Plans**: 1 plan
Plans:

- [x] 51-01-PLAN.md — Setup script, smoke config, and SLURM job scripts (all 4 ENV-01/02/03 + OUT-01 artifacts)

### Phase 52: Multi-Rank Parallelism & Validation

**Goal**: HPO trials execute concurrently across MPI ranks via the existing propulate backend, run_all.py orchestration is single-writer regardless of world size, cluster configs exist independently of laptop configs, and correct multi-rank behavior is proven on a short test job before committing to a full allocation.
**Depends on**: Phase 51
**Requirements**: PARA-01, PARA-02, PARA-03, BUDG-01, BUDG-04
**Success Criteria** (what must be TRUE):

  1. HPO trials for an optimize run execute concurrently across MPI ranks (not sequentially) when a config specifies `search_strategy: propulate`
  2. `run_all.py`'s orchestration (file writes, `EvaluationRunner` calls, per-run bookkeeping) executes exactly once per run regardless of MPI world size, while `HyperparamOptimizer.run()` remains collective across ranks
  3. Cluster-targeted configs (`search_strategy: propulate`/`auto`) exist as separate files from the local laptop configs (`search_strategy: sobol`), and both remain independently runnable without interfering
  4. Cluster configs default to the same `max_points_per_frame`/`step` subsampling already calibrated on the laptop, not full point density
  5. A short test job on HoreKa validates correct multi-rank behavior (no duplicated or racing output writes) and produces real per-trial timing data, completed before any full 3-hour allocation is submitted

**Plans**: 3 plans
Plans:

- [x] 52-01-PLAN.md — run_all.py rank-awareness + --configs-dir arg (PARA-01, PARA-02)
- [x] 52-02-PLAN.md — Cluster configs (configs_horeka/ — 7 mirrored YAMLs) (PARA-03, BUDG-01)
- [x] 52-03-PLAN.md — Multi-rank test job (smoke config + launch_horeka_multirank_test.sbatch) (BUDG-04)

### Phase 53: GPU Acceleration

**Goal**: Real per-operation GPU acceleration is threaded through the pipeline — configurable device, data loaded onto that device, and registration provably executing on GPU tensors with no silent CPU fallback.
**Depends on**: Phase 52
**Requirements**: GPU-01, GPU-02, GPU-03
**Success Criteria** (what must be TRUE):

  1. `EvalConfig` exposes a `device` field that controls where tensors are loaded and computed
  2. `DataFactory` loads real/target/ground-truth data onto the configured device instead of the current hardcoded CPU
  3. Operator can verify end-to-end that `AlignmentStage`/CPD registration actually runs on GPU tensors, with no silent CPU fallback anywhere in the path

**Plans**: 2 plans
Plans:

- [x] 53-01-PLAN.md — EvalConfig device field + DataFactory device threading + tests (GPU-01, GPU-02)
- [x] 53-02-PLAN.md — 8 cluster YAML configs + sbatch GPU-03 annotation (GPU-03)

### Phase 54: Budget Calibration & Full-Suite Gate

**Goal**: Calibration constants are environment-aware, and the full 7-run suite's projected runtime is verified to fit the 3-hour GPU cap before the full allocation is submitted.
**Depends on**: Phase 53
**Requirements**: BUDG-02, BUDG-03
**Success Criteria** (what must be TRUE):

  1. `aggregate_cost.py`'s calibration constants are parameterized per environment (laptop vs. HoreKa), so cluster timing data doesn't silently mix with or overwrite laptop calibration
  2. Operator can run an `aggregate_cost.py`-style estimation using HoreKa GPU timing data (from Phase 52's short test job plus Phase 53's GPU path) to project the full 7-run suite's total runtime
  3. The projected total runtime for the full suite is confirmed to fit within the 3-hour GPU time cap before the full allocation is submitted (or the plan is revised if it doesn't fit)

**Plans**: 2 plans

Plans:

**Wave 1**

- [x] 54-01-PLAN.md — aggregate_cost.py argparse (--calibration, --configs-dir, --budget-hours) + calibration JSON files (BUDG-02, BUDG-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 54-02-PLAN.md — extract_calibration.py new script + tests for aggregate_cost.py and extract_calibration.py (BUDG-02, BUDG-03)

### ✅ v1.5 Learned Label Transfer Methods (Complete)

**Milestone Goal:** Extend LabelTransferStage with learned point-cloud methods (CPD-weighted, eGNN, PointNet++) — full stack from architecture selection through training infrastructure, inference integration, and evaluation/benchmarking.

- [x] **Phase 44: CPD-Weighted Label Transfer** — completed 2026-07-15
- [x] **Phase 45: eGNN/PointNet++ Framework Selection & Evaluation Strategy** — completed 2026-07-15
- [x] **Phase 46: Training Data Pipeline** — completed 2026-07-15
- [x] **Phase 47: eGNN/PointNet++ Model Implementation & Training Infrastructure** — completed 2026-07-15
- [x] **Phase 48: LabelTransferStage Integration for Learned Methods** — completed 2026-07-15
- [x] **Phase 49: Evaluation & Benchmarking of Learned Label-Transfer Methods** — completed 2026-07-15
- [x] **Phase 50: GPU-Native Geometry Ops** — completed 2026-07-31 (VERIFICATION.md: passed)

### Phase 55: Spherical-cap and Gaussian label generators for zreg.data_generation.labels ✅ 2026-07-30

**Goal:** zreg.data_generation.labels exposes assign_cap_labels (hard spherical-cap boundary) and assign_gaussian_labels (angle-dependent Bernoulli labels), both following the immutable deep-copy contract and exported from the package.
**Requirements**: none mapped
**Depends on:** Phase 54
**Plans:** 4/4 plans complete

Plans:

- [x] 55-01-PLAN.md — Implement assign_cap_labels + assign_gaussian_labels, export them, and add tests

### Phase 56: Configurable multi-label region-based labeling: rework generate_labels() to support arbitrary n_labels, voronoi/gaussian-blob/gaussian-cone region shapes, deterministic and probabilistic assignment modes, and config-driven specification via EvalConfig ✅ 2026-07-31

**Goal:** zreg.data_generation.labels.generate_labels() becomes the single, config-driven entry point for labeling a point cloud trajectory — any number of labels, each defined by one or more region components (voronoi/gaussian blob/gaussian cone), assigned deterministically or probabilistically, with region centers fixed once per trajectory (not redrawn per frame) so label change is spatially traceable. assign_cap_labels/assign_gaussian_labels (Phase 55) are deleted entirely, absorbed as the cone shape's special case. EvalConfig.label_generation lets scenario YAML declare a label spec instead of hardcoded Python literals.
**Requirements**: none mapped (see 56-CONTEXT.md D-01 through D-13 — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 55
**Plans:** 5/5 plans complete

Plans:

**Wave 1**

- [x] 56-01-PLAN.md — LabelComponentSpec/LabelSpec pydantic models + per-shape scoring (_component_score) + mixture aggregation (_label_scores) (D-08, D-09, D-10, D-11)

**Wave 2** *(depends on Wave 1)*

- [x] 56-02-PLAN.md — Assignment modes (_assign_deterministic/_assign_probabilistic) + reworked generate_labels() orchestrator with D-07 center-once fix + deletion of assign_cap_labels/assign_gaussian_labels and their 13 tests (D-01, D-04, D-05, D-06, D-07, D-12)

**Wave 3** *(depends on Wave 2)*

- [x] 56-03-PLAN.md — EvalConfig.label_generation field + n_classes->n_labels rename cascade through DataFactory/optimizer.py/benchmark_runner.py/train_label_transfer.py + example scenario YAML (D-04, D-13)

**Wave 4** *(depends on Waves 2+3, parallel)*

- [x] 56-04-PLAN.md — Comprehensive new tests: per-shape correctness, assignment-mode behavior, D-07 regression, LabelGenerationConfig coverage (D-07, D-08, D-09, D-10, D-11, D-12, D-13)
- [x] 56-05-PLAN.md — Fix rename-cascade breakage across the remaining existing test suite (9 files) (D-04)

### Phase 50: GPU-Native Geometry Ops for Cluster Deployment

**Goal:** Investigate replacing Phase 47's Open3D-CPU-based FPS/ball-query ops
(`src/zreg/models/_ops.py`) with `torch_cluster`'s GPU-native equivalents for PointNet++, if the
user's target cluster (Linux/CUDA) can actually install `torch_cluster` — a build failure that
was only ever reproduced on this dev machine (macOS ARM). `zRegPointCloud` is unaffected either
way (the ops already take/return plain `torch.Tensor`, not Open3D objects). eGNN's hand-rolled
equivariant conv layer is out of scope — that choice was architecture-fit-driven (no suitable
packaged implementation exists at any point-cloud scale), not platform-driven, and stays
unchanged regardless of this phase's outcome.
**Requirements**: TBD (see 50-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 49
**Plans:** 2/2 plans complete

Plans:

- [x] 50-01-PLAN.md — HoreKa torch_cluster install verification gate (autonomous: false, human-executed) (D-01, D-02)
- [x] 50-02-PLAN.md — torch_cluster dual-path in _ops.py for all three ops + setup.cfg cluster extra + smoke tests (D-03–D-08)

### Phase 44: CPD-Weighted Label Transfer Method

**Goal:** Wire zreg.color_transfer's existing CPD_WEIGHTED method into LabelTransferStage as a
selectable alternative to the hardcoded KNN_VOTING, mirroring the Phase 39 alignment_method
optional-param precedent — fixing the pmat-orientation and categorical-averaging bugs found during
investigation, without touching zreg.color_transfer itself.
**Requirements**: D-01 through D-09 (see 44-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 43
**Plans:** 4/4 plans complete

Plans:

- [x] 44-01-PLAN.md — AlignResult.estep_results field + EvalConfig.label_transfer_method field/validator (D-06, D-09)
- [x] 44-02-PLAN.md — AlignmentStage CPD posterior capture (_build_aligned_cloud + run()) (D-01, D-02, D-03)
- [x] 44-03-PLAN.md — LabelTransferStage cpd_weighted wiring (transpose + one-hot/argmax + align_result kwarg) (D-04, D-05, D-06, D-07, D-08)
- [x] 44-04-PLAN.md — EvaluationRunner align_result threading (D-08)

**Status:** ✅ Complete — all 4 plans executed, Phase 44 fully wired end-to-end.

### Phase 45: eGNN & PointNet++ Label Transfer — Framework Selection & Evaluation Strategy

**Goal:** Decide the model architecture/library approach for eGNN and PointNet++ as new
LabelTransferStage methods (off-the-shelf library vs. hand-rolled, against this project's current
lean torch/numpy/open3d dependency footprint), research each architecture's implementation
requirements, and design an evaluation strategy for learned label-transfer methods. Produces a
design document (45-DESIGN.md) synthesizing the locked decisions for Phases 46–49 — no production
code in this phase. (Note: /gsd:ai-integration-phase was evaluated and rejected as a category error —
eGNN/PointNet++ are classical supervised point-cloud networks, not LLM/agent frameworks; see
45-CONTEXT.md.)
**Requirements**: D-01 through D-04 (see 45-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 44
**Plans:** 1/1 plans complete

Plans:

- [x] 45-01-PLAN.md — Verify torch_geometric core install + write 45-DESIGN.md locking per-model library, module structure, joint-cloud adaptation, train/infer split, and eval-strategy pointers for Phases 46–49 (D-01, D-02, D-03, D-04)

**Status:** ✅ Complete — torch_geometric core verified installable, 45-DESIGN.md locks all Phase 46-49 architecture/library decisions.

### Phase 46: Training Data Pipeline for Learned Label Transfer

**Goal:** Extend DataFactory to emit seed-driven (source cloud + labels, target cloud, target
labels) training triples for Phase 47's eGNN/PointNet++ training loop — porting a single-frame
ball/bowl geometry sampler into `zreg.generators`, wiring the existing Voronoi `generate_labels`,
reusing Phase 30's transform-based exact-correspondence `generate_target`, and adding a seed-level
train/val split. Small point-count regime only (100-300 pts/frame), on-the-fly seeded generation,
no persisted dataset.
**Requirements**: D-01 through D-04 (see 46-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 45
**Plans:** 3/3 plans complete

Plans:

- [x] 46-01-PLAN.md — Port single-frame sample_ball()/sample_bowl() geometry samplers into zreg/generators (D-04)
- [x] 46-02-PLAN.md — Add frozen TrainingTriple result model to eval/types.py (D-02)
- [x] 46-03-PLAN.md — split_seeds() + DataFactory.generate_training_triple()/generate_training_set() + augment() per-seed RNG threading + Wave-0 tests (D-01, D-02, D-03, D-04)

**Status:** ✅ Complete — 1248 tests pass (was 1229 at 46-02 close), 100% coverage on zreg/eval.

### Phase 47: eGNN and PointNet++ Model Implementation & Training Infrastructure

**Goal:** Implement the eGNN and PointNet++ model architectures selected in Phase 45, plus a
training loop and checkpoint management (none of which exists in this repo today).
**Requirements**: MODEL-01 through MODEL-07 (derived in 47-RESEARCH.md — phase added ad hoc, no formal REQUIREMENTS.md IDs); binds D-01 through D-03 (see 47-CONTEXT.md)
**Depends on:** Phase 46
**Plans:** 5/5 plans complete

Plans:

- [x] 47-01-PLAN.md — Foundation: declare torch_geometric in setup.cfg + src/zreg/models/_ops.py (Open3D FPS/ball-query/radius-graph, isolated-point guard) + ops tests incl. D-03 benchmark (MODEL-01, MODEL-04)
- [x] 47-02-PLAN.md — PointNet++ joint-cloud model: SetAbstraction/FeaturePropagation/PointNet2LabelTransfer + tests (MODEL-02, MODEL-04)
- [x] 47-03-PLAN.md — eGNN: verified single-propagate EGNNConv (MessagePassing subclass) + EGNNLabelTransfer + E(3) equivariance test (MODEL-03, MODEL-04)
- [x] 47-04-PLAN.md — Training entry point (train_label_transfer.py, MPS-aware) + models __init__ exports + smoke test (forward/backward/loss-decrease/checkpoint round-trip + MPS run) (MODEL-04, MODEL-05, MODEL-06, MODEL-07)
- [x] 47-05-PLAN.md — Joint-cloud output-shape==target-point-count contract test for both models (MODEL-02, MODEL-03)

**Status:** ✅ Complete — all 5 plans executed. 1285 tests pass (was 1248 at Phase 46 close), zero regressions.

### Phase 48: LabelTransferStage Integration for Learned Methods

**Goal:** Wire trained eGNN/PointNet++ models into LabelTransferStage as new selectable method
values (checkpoint loading, inference-time dispatch), following the OPTIONAL_PARAMS +
EvalConfig.label_transfer_method precedent established in Phase 44 for cpd_weighted.
**Requirements**: D-01 through D-03 (see 48-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 47
**Plans:** 2/2 plans complete

Plans:

- [x] 48-01-PLAN.md — EvalConfig egnn_checkpoint_path/pointnet2_checkpoint_path fields + 4-value label_transfer_method validator (D-02)
- [x] 48-02-PLAN.md — LabelTransferStage learned-method wiring: VALID_METHODS + _load_learned_model + train_step-parity joint-cloud branch + smoke-test verification (D-01, D-02, D-03)

**Status:** ✅ Complete — both plans executed. 1300 tests pass (was 1285 at Phase 47 close), zero regressions.

### Phase 49: Evaluation & Benchmarking of Learned Label-Transfer Methods

**Goal:** Apply the existing F1/knn_consistency metrics plus any additional evaluation dimensions
from Phase 45's strategy to benchmark eGNN/PointNet++ against the knn_voting/cpd_weighted
baselines across this project's dataset sources.
**Requirements**: D-01 through D-03, D-02.1 through D-02.4 (see 49-CONTEXT.md — phase added ad hoc, no formal REQUIREMENTS.md IDs)
**Depends on:** Phase 48
**Plans:** 3/3 plans complete

Plans:

- [x] 49-01-PLAN.md — MethodBenchmarkResult/BenchmarkReport types + Wave-0 verification (D-01, D-02.2 fixtures)
- [x] 49-02-PLAN.md — LabelTransferBenchmark orchestration class (D-01, D-02.1–D-02.4, D-03)
- [x] 49-03-PLAN.md — benchmark_label_transfer.py CLI entry point (D-01)

**Status:** ✅ Complete — all 3 plans executed. 1312 tests pass (was 1300 at Phase 48 close), zero regressions. eGNN/PointNet++ track (Phases 44–49) fully complete.

<details>
<summary>✅ v1.4 Trajectory Alignment & Optimization Enhancements (Phases 39–43) — SHIPPED 2026-07-08</summary>

- [x] Phase 39: ICP Registration as CPD Alternative (2/2 plans) — completed 2026-06-29
- [x] Phase 40: Sliced Wasserstein Variants as Alignment Method (4/4 plans) — completed 2026-06-29
- [x] Phase 41: Alignment Preprocessing — Principal Axes + Velocity Landmarks (2/2 plans) — completed 2026-06-29
- [x] Phase 42: Sobol Quasi-Random Search as Default (2/2 plans) — completed 2026-06-30
- [x] Phase 43: Per-Trajectory Data Standardization (1/1 plans) — completed 2026-06-30

Full details: [.planning/milestones/v1.4-ROADMAP.md](.planning/milestones/v1.4-ROADMAP.md)

</details>

<details>
<summary>✅ v1.2 Evaluation Framework & Debt Resolution (Phases 12–38) — SHIPPED 2026-06-26</summary>

- [x] Phase 12: Carry-Forward Debt Closure (3/3 plans) — completed 2026-05-14
- [x] Phase 13: Core Metrics Library (3/3 plans) — completed 2026-05-15
- [x] Phase 14: Synthetic Data Generators (3/3 plans) — completed 2026-05-18
- [x] Phase 15: Experiment Tracking & Run Management (2/2 plans) — completed 2026-05-18
- [x] Phase 16: Runner Scripts (2/2 plans) — completed 2026-05-19
- [x] Phase 17: Framework Config & DataFactory (2/2 plans) — completed 2026-05-27
- [x] Phase 18: MetricsEngine & Result Types (2/2 plans) — completed 2026-05-28
- [x] Phase 19: AlignmentStage (2/2 plans) — completed 2026-05-28
- [x] Phase 20: LabelTransferStage (2/2 plans) — completed 2026-05-29
- [x] Phase 21: EvaluationRunner & Visualisation (2/2 plans) — completed 2026-05-29
- [x] Phase 22: HyperparamOptimizer & Search Strategies (2/2 plans) — completed 2026-05-29
- [x] Phase 23: CLI Entrypoint & Scenario Configs (2/2 plans) — completed 2026-06-02
- [x] Phase 24: Trajectory Export (2/2 plans) — completed 2026-06-04
- [x] Phase 25: Visualisation Refactor (2/2 plans) — completed 2026-06-04
- [x] Phase 26: Propulate Optimizer (2/2 plans) — completed 2026-06-06
- [x] Phase 27: DataFactory Geometric Augmentation Methods (2/2 plans) — completed 2026-06-11
- [x] Phase 28: Script Integration — generate_datasets uses DataFactory (1/1 plans) — completed 2026-06-11
- [x] Phase 29: Viz Unification (2/2 plans) — completed 2026-06-12
- [x] Phase 30: Two-Dataset Paired Alignment Architecture (3/3 plans) — completed 2026-06-12
- [x] Phase 31: Synthetic Pipeline Mode — Transform-Spec Target & GT-Aware HPO (3/3 plans) — completed 2026-06-14
- [x] Phase 32: Heterogeneous Paired Evaluation — target_data_format (2/2 plans) — completed 2026-06-14
- [x] Phase 33: CPD-Aligned Trajectory Output from AlignmentStage (2/2 plans) — completed 2026-06-16
- [x] Phase 34: Alignment Quality Guard in LabelTransferStage (1/1 plans) — completed 2026-06-19
- [x] Phase 35: Reuse Step-1 CPD Transforms in Aligned-Cloud Construction (2/2 plans) — completed 2026-06-22
- [x] Phase 36: plot_trajectory Alignment Figure Refactor (1/1 plans) — completed 2026-06-22
- [x] Phase 37: plot_trajectory Label Figure Refactor (1/1 plans) — completed 2026-06-23
- [x] Phase 38: zRegPointCloud color→label Field Rename (2/2 plans) — completed 2026-06-24

Full details: [.planning/milestones/v1.2-ROADMAP.md](.planning/milestones/v1.2-ROADMAP.md)

</details>

<details>
<summary>✅ v1.1 Code Quality & Refactoring (Phases 6–11.1) — SHIPPED 2026-05-13</summary>

- [x] Phase 6: Python 3.12 Migration (2/2 plans) — completed 2026-04-13
- [x] Phase 7: CPD Deep Restructure (3/3 plans) — completed 2026-04-20
- [x] Phase 8: DTW Deep Restructure (2/2 plans) — completed 2026-04-23
- [x] Phase 9: Distance & Transform Restructure (2/2 plans) — completed 2026-04-27
- [x] Phase 10: Code Quality & Verification (3/3 plans) — completed 2026-04-29
- [x] Phase 11: Validate Refactoring and Fix Coverage (1/1 plan) — completed 2026-05-12
- [x] Phase 11.1: Close DTW-02 — Consistent Metric Variant Interface (1/1 plan, INSERTED) — completed 2026-05-13

Full details: [.planning/milestones/v1.1-ROADMAP.md](.planning/milestones/v1.1-ROADMAP.md)

</details>

<details>
<summary>✅ v1.0 Consolidation (Phases 1–5) — SHIPPED 2026-04-09</summary>

- [x] Phase 1: Validation Foundation & Quick Wins (2/2 plans) — completed 2026-04-09
- [x] Phase 2: Distance Metric & CPD Bug Fixes (3/3 plans) — completed 2026-04-09
- [x] Phase 3: DTW, Transform & CPD Enhancements (3/3 plans) — completed 2026-04-09
- [x] Phase 4: Infrastructure & Color Transfer Quality (2/2 plans) — completed 2026-04-09
- [x] Phase 5: Test Coverage (3/3 plans) — completed 2026-04-09

Full details: [.planning/milestones/v1.0-ROADMAP.md](.planning/milestones/v1.0-ROADMAP.md)

</details>

## Progress

**Execution Order (v1.8):**
Phase 59 first (P0; NUM-05 changes failure semantics everything else builds on). Phases 60, 61 and 62 each depend only on 59 and may run in parallel, with two coordination points: 61 (DIST-04) edits the same `icp.py`/`swd_aligner.py` as 59 (NUM-01), and 61's DIST-03 cpd-branch tests should assert the `q` corrected in 60 (CPD-02). Phase 63 depends on 59. Default order: 59 → 60 → 61 → 62 → 63.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 59. P0 Runnability & Silent Number Corruption | v1.8 | 8/8 | Complete   | 2026-10-02 |
| 60. CPD/DTW Numerics | v1.8 | 5/5 | Complete    | 2026-10-03 |
| 61. Distances, Aligners & MPI | v1.8 | 4/4 | Complete   | 2026-10-03 |
| 62. Data, Label Transfer & GPU Path | v1.8 | 6/6 | Complete   | 2026-10-03 |
| 63. HPC Orchestration, Viz/Export & Docs | v1.8 | 9/9 | Complete   | 2026-10-03 |
| 64. Tech Debt Cleanup (doctests, stale refs, bookkeeping) | v1.8 | 5/5 | Complete | 2026-10-05 |
| 44. CPD-Weighted Label Transfer | v1.5 | 4/4 | ✅ Complete | 2026-07-15 |
| 45. eGNN/PointNet++ Framework Selection | v1.5 | 1/1 | ✅ Complete | 2026-07-15 |
| 46. Training Data Pipeline | v1.5 | 3/3 | ✅ Complete | 2026-07-15 |
| 47. eGNN/PointNet++ Models & Training | v1.5 | 5/5 | ✅ Complete | 2026-07-15 |
| 48. LabelTransferStage Integration | v1.5 | 2/2 | ✅ Complete | 2026-07-15 |
| 49. Benchmarking of Learned Methods | v1.5 | 3/3 | ✅ Complete | 2026-07-15 |
| 50. GPU-Native Geometry Ops | v1.5 | 2/2 | Complete    | 2026-07-31 |
| 51. Environment & Access | v1.6 | 1/1 | Complete    | 2026-07-15 |
| 52. Multi-Rank Parallelism & Validation | v1.6 | 3/3 | Complete    | 2026-07-23 |
| 53. GPU Acceleration | v1.6 | 2/2 | Complete    | 2026-07-23 |
| 54. Budget Calibration & Full-Suite Gate | v1.6 | 2/2 | Complete    | 2026-07-23 |
| 57. Ground-Truth Field Consistency | v1.7 | 2/2 | Complete    | 2026-08-04 |
| 58. Synthetic Labeled Subsample-Pair Generation | v1.7 | 4/4 | Complete    | 2026-08-04 |

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 Consolidation | 1–5 (5) | 13 | ✅ Complete | 2026-04-09 |
| v1.1 Code Quality & Refactoring | 6–11.1 (7) | 14 | ✅ Complete | 2026-05-13 |
| v1.2 Evaluation Framework & Debt Resolution | 12–38 (27) | 55 | ✅ Complete | 2026-06-26 |
| v1.4 Trajectory Alignment & Optimization Enhancements | 39–43 (5) | 11 | ✅ Complete | 2026-07-08 |
| v1.5 Learned Label Transfer Methods | 44–50 (7) | 18 | ✅ Complete | 2026-07-31 |
| v1.6 HoreKa Cluster Execution | 51–54 (4) | 9 | ✅ Complete | 2026-07-23 |
| v1.7 Minor Adjustments | 57–58 (2) | 6 | ✅ Complete | 2026-08-04 |
| v1.8 Code Review Remediation | 59–64 (6) | 37 | 🚧 In progress | - |

_v1.8 Code Review Remediation (Phases 59–64) is active. v1.5 (44–50), v1.6 (51–54) and v1.7 (57–58) are complete but not yet formally archived via `/gsd:complete-milestone`. Phases 55 and 56 were ad-hoc out-of-band phases (see the Phase Numbering note above)._
