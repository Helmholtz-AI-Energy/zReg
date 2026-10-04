# Requirements: zReg v1.8 Code Review Remediation

**Defined:** 2026-10-02
**Core Value:** Every existing capability works correctly, fails informatively, and is covered by tests.
**Source:** External code review of `feature/evaluation_framework` @ `6c1c37f` (2026-09-17) — `.planning/reviews/00-INDEX.md` (triage) and `unit-01..unit-08`. Finding IDs `U<unit>-<n>` refer to those files; each file gives location, mechanism and (often) a reproducer.

**Cross-cutting rule (applies to every requirement):** each fix ships with a regression test that exercises the real contract — it must fail on `6c1c37f` and pass after the fix, and must not mock the unit under test (review pattern 3: "100% coverage is line coverage over mocks").

## v1 Requirements

### P0 — Runnability (HoreKa suite and scripts start at all)

- [ ] **RUN-01**: All 5 `configs_horeka/*/ew06_vs_shah.yaml` load through `EvalConfig.from_yaml` (`label_source` restored as a validated `EvalConfig` field — see NUM-04 — so the existing keys are valid again; mid-comment key insertion + duplicate key in `hpo_paired` fixed) and a test loads every YAML under `configs/` and `baseline_experiments/configs*/` (U4-1, U7-2, U4-6)
- [ ] **RUN-02**: `scripts/generate_real_previews.py`, `scripts/label_transfer_example.py`, `example_plots.py`, `dtw_testing.py` import and resolve every `zreg.*` name they use; `from zreg.dtw import X` works via a real `sys.modules` shim (U7-5, U7-6, U8-5)

### P0 — Silent number corruption

- [ ] **NUM-01**: ICP and SWD aligners' `D_inv` is the exact inverse of `D` (identity registration round-trips input coordinates), and denormalisation uses the bounds of the frame the result lives in (U3-1, U3-2, U3-5)
- [ ] **NUM-02**: Multi-seed subsample-pair HPO scores the *aligned* per-frame dicts (not raw tensors / unaligned source view), so chamfer/hausdorff reflect actual alignment quality (U1-1, U5-1)
- [ ] **NUM-03**: List-valued `seed` in `subsample_pair` mode runs sanity and dev tiers on `seed[0]` (pre-populate happens; no swallowed `ValueError`/`TypeError`) (U5-2, U4-4)
- [ ] **NUM-04**: Paired-mode label transfer again takes labels from Shah (target) and transfers them onto the aligned Kobitski source, as `c60a943` intended — the `c68c63c` cleanup reverted this and left an inverted comment (U4-2)
- [ ] **NUM-05**: Degenerate metric/trial outcomes can no longer masquerade as perfect scores: empty frame-key intersection returns non-finite or raises, partial overlap is flagged, a failed trial is recorded as failed (not `0.0` → normalised `1.0`), and `sanity_check` catches it (U1-2, U1-3, U1-5, U1-6; index cross-cutting pattern 1)

### CPD / DTW numerics

- [x] **CPD-01**: CPD convergence test only runs once four real `q` values exist (no `torch.arange(4)` seed); a 0.1% input-scale change no longer flips `n_iters` 12→1 (U2-1)
- [x] **CPD-02**: Rigid CPD `q` matches the affine form (log-likelihood term added, not in the denominator); `q` used as DTW cost is non-negative-consistent (U2-2)
- [x] **CPD-03**: Rigid CPD fixed-scale `sigma2` uses `tr_xp1x − 2·tr_atr + tr_yp1y` (Myronenko & Song Eq. 23) (U2-3)
- [x] **CPD-04**: `ConstrainedNonRigidCPD` sets `self.transformation` like `NonRigidCPD` (U2-4)
- [x] **CPD-05**: `use_color=True` either works end-to-end or is removed/rejected with a clear error (U2-5)
- [x] **CPD-06**: `init_cpd_from_existing` warm-starts `AffineCPD` from the given transform; `AffineTransformation()` default is the identity (U2-6, U2-7)
- [x] **CPD-07**: DTW `save()`/`load()` round-trips configs whose `distance_metric` is a callable (U2-8)
- [x] **CPD-08**: `RigidCPD`'s default initial rotation is the identity; the dataset-specific non-orthogonal pose is opt-in via `tf_init_params` (U2-9)
- [x] **CPD-09**: Non-rigid CPD handles sources wider than 3 columns; `rbf_kernel_matrix` matches the kernel CPD uses (or is removed); TPS docstring example runs; stale module paths/comments in `cpd/`, `dtw/`, `core/transforms/`, `eval/stages/alignment.py` updated (U2-10, U2-11, U2-12, U2-13)

### Distances, aligners & MPI

- [ ] **DIST-01**: Pairwise distance matrix never deadlocks when a rank owns no pair in a row (both code paths) — verified with a multi-rank test or a rank-simulating test (U3-3)
- [ ] **DIST-02**: MaxSWD/GSWD/PSWD projections follow input dtype and device (float64 and CUDA work); `max_sw_num_iters`/`max_sw_lr` are honoured (U3-4, U3-7)
- [ ] **DIST-03**: Pairwise sweep works with non-zero-based frame keys; cpd-branch `fn is None` and leading-`None` `distance_kwargs` handled consistently (U3-6, U3-11, U3-12)
- [ ] **DIST-04**: SWD aligner handles unequal point counts, returns a proper rotation (`det=+1`, orthogonalised at return), and ICP/SWD reject degenerate/empty clouds with an informative error (U3-8, U3-9, U3-10)
- [ ] **DIST-05**: Frame-averaged chamfer/hausdorff computes one `cdist` per frame and derives both metrics from it (U1-4)

### Data, label transfer & GPU path

- [x] **DATA-01**: `device: cuda` runs work through `get_synthetic_ground_truth` gather and label-id indexing (buffers created on the input device) (U4-3, U6-2)
- [x] **DATA-02**: `augment()` orders outlier insertion/dropout so correspondence indices stay valid (U4-5)
- [x] **DATA-03**: `LabelGenerationConfig` / `generate_labels` reject empty `label_specs` / `n_labels: 0` and unknown `mode` at config time; `subsample_pair` synthesize path honours `config.label_generation` (U4-7, U4-8, U6-6, U6-8)
- [x] **DATA-04**: RGB→index label remap is decided over all (non-empty) frames, not frame 0 only; unused `source_corr` removed (U4-9, U4-10)
- [x] **LT-01**: Model-based label transfer takes `n_classes` from the model and accepts 1-D label tensors (U6-1)
- [x] **LT-02**: Gaussian-kernel and CPD `pmat` weighting never produce NaN on zero row sums; transposed-`pmat` guard is correct when `n_source == n_target`; `source["label"]` is None-checked and does not clobber caller input (U6-3, U6-5, U6-7, U6-9)
- [x] **LT-03**: `sample_bowl` validates `d_ratio`/`radius` instead of looping forever (U6-4)
- [x] **LT-04**: `knn_consistency` excludes the query point by index (not by column 0); `temporal_stability` raises the documented exception type; WR-01 truncation passes consistent `labels_for_knn`/positions in `_objective` and multi-seed path (U1-8, U1-9, U5-3)

### HPC orchestration

- [ ] **HPC-01**: `baseline_with_combined` warm-start actually seeds the HPO (optimizer reads injected defaults / `warm_start` is passed and supported by the active search strategy, or the limitation fails loudly) (U7-1)
- [ ] **HPC-02**: `merge_combined_params` merges only calibrated values (no defaults dilution) (U7-3)
- [ ] **HPC-03**: Resubmitting the HoreKa baseline job resumes HPO from checkpoints (no unconditional `--clear-checkpoints`) (U7-4)
- [ ] **HPC-04**: All MPI ranks share one timestamped output directory (rank-0 broadcast); eval jobs run a single rank; SLURM log directory exists before submission; `test_propulate_interactive.sh` empty-dir guard; partition comment fixed (U7-7, U7-8, U7-9, U7-10, U7-11)

### Viz, export & docs

- [ ] **VIZ-01**: Superposed-trajectory and label-trajectory figures contain their legends, and dataset-preview titles are not clipped (U8-1, U8-2, U8-3)
- [ ] **VIZ-02**: Source panel frame selection mirrors `_build_aligned_cloud` (`source_sorted[::step]`) (U8-4)
- [ ] **VIZ-03**: `label_metadata.json` frame count/indices describe the same frames as `label_trajectory.csv` (U8-6)
- [ ] **DOC-01**: Restructure leftovers cleaned: doctest in `evaluation/label_transfer.py` imports a live module (and `tox -e doctests` passes), stale `zreg.metrics`/`zreg.distances`/`zreg.dataset` references in docstrings, contradictory frame-convention docstrings in `eval_runner.py`, `path_smoothness` module header, `pyproject.toml` coverage `omit` path (U1-7, U1-10, U1-11, U1-12, U5-4, U3 non-findings)

## Future Requirements

- Audit of `tests/` (23k LOC): which tests mock the unit under test vs. assert a contract (index "Outstanding")
- Consistency sweep of `configs/` + `baseline_experiments/configs*/` beyond the RUN-01 load test
- Merge-gate review of the full `main...feature/evaluation_framework` diff after this milestone
- Re-running baseline/HPO results that were produced with the defective CPD/ICP/SWD code

## Out of Scope

| Feature | Reason |
|---------|--------|
| `data_generation/transforms.py:118` sign-clamp | Reviewer explicitly "not reported" — unreachable; comment fix only, optional |
| Retroactive correction of published numbers | Fixes the mechanism; re-runs tracked under Future Requirements |
| Pushing to `origin/feature/evaluation_framework` | Local commits only; push is an explicit operator decision |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RUN-01 | Phase 59 | Pending |
| RUN-02 | Phase 59 | Pending |
| NUM-01 | Phase 59 | Pending |
| NUM-02 | Phase 59 | Pending |
| NUM-03 | Phase 59 | Pending |
| NUM-04 | Phase 59 | Pending |
| NUM-05 | Phase 59 | Pending |
| CPD-01 | Phase 60 | Complete |
| CPD-02 | Phase 60 | Complete |
| CPD-03 | Phase 60 | Complete |
| CPD-04 | Phase 60 | Complete |
| CPD-05 | Phase 60 | Complete |
| CPD-06 | Phase 60 | Complete |
| CPD-07 | Phase 60 | Complete |
| CPD-08 | Phase 60 | Complete |
| CPD-09 | Phase 60 | Complete |
| DIST-01 | Phase 61 | Pending |
| DIST-02 | Phase 61 | Pending |
| DIST-03 | Phase 61 | Pending |
| DIST-04 | Phase 61 | Pending |
| DIST-05 | Phase 61 | Pending |
| DATA-01 | Phase 62 | Complete |
| DATA-02 | Phase 62 | Complete |
| DATA-03 | Phase 62 | Complete |
| DATA-04 | Phase 62 | Complete |
| LT-01 | Phase 62 | Complete |
| LT-02 | Phase 62 | Complete |
| LT-03 | Phase 62 | Complete |
| LT-04 | Phase 62 | Complete |
| HPC-01 | Phase 63 | Pending |
| HPC-02 | Phase 63 | Pending |
| HPC-03 | Phase 63 | Pending |
| HPC-04 | Phase 63 | Pending |
| VIZ-01 | Phase 63 | Pending |
| VIZ-02 | Phase 63 | Pending |
| VIZ-03 | Phase 63 | Pending |
| DOC-01 | Phase 63 | Pending |

**Coverage:** 37/37 v1 requirements mapped, no orphans, no duplicates.

---
*Requirements defined: 2026-10-02 · Traceability filled by roadmap: 2026-10-02*
