---
phase: 53
slug: gpu-acceleration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-16
---

# Phase 53 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `setup.cfg` |
| **Quick run command** | `python -m pytest tests/test_eval_config.py tests/test_data_factory.py -x -q` |
| **Full suite command** | `python -m pytest -x -q` |
| **Estimated runtime** | ~30 seconds (full suite ~60s) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_eval_config.py tests/test_data_factory.py -x -q`
- **After every plan wave:** Run `python -m pytest -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 53-01-01 | 01 | 1 | GPU-01 | device field defaults to "cpu"; invalid values raise EvalConfigError | unit | `pytest tests/test_eval_config.py -k "TestEvalConfigDevice" -x -q` | ❌ W0 | ⬜ pending |
| 53-01-02 | 01 | 1 | GPU-01 | device="cuda" + alignment_method="icp" raises EvalConfigError at parse time | unit | `pytest tests/test_eval_config.py -k "TestEvalConfigDeviceICPGuard" -x -q` | ❌ W0 | ⬜ pending |
| 53-01-03 | 01 | 1 | GPU-02 | DataFactory.__init__ raises RuntimeError when device="cuda" and CUDA unavailable (mocked) | unit | `pytest tests/test_data_factory.py -k "TestDataFactoryDeviceGuard" -x -q` | ❌ W0 | ⬜ pending |
| 53-01-04 | 01 | 1 | GPU-02 | load_real(), load_target(), get_ground_truth() pass self.config.device to loaders | unit | `pytest tests/test_data_factory.py -k "test_dispatches" -x -q` | ✅ (update) | ⬜ pending |
| 53-02-01 | 02 | 2 | GPU-03 | All 8 YAML configs parse successfully with device: "cuda" field | unit | `pytest tests/test_eval_config.py -k "test_device_yaml" -x -q` | ❌ W0 | ⬜ pending |
| 53-02-02 | 02 | 2 | GPU-03 | GPU tensors confirmed in SLURM output | manual | N/A — grep SLURM log on HoreKa | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_eval_config.py` — ADD `TestEvalConfigDevice` class (device default, valid values, invalid raises)
- [ ] `tests/test_eval_config.py` — ADD `TestEvalConfigDeviceICPGuard` class (cross-field model_validator)
- [ ] `tests/test_data_factory.py` — ADD `TestDataFactoryDeviceGuard` class (CUDA unavailable → RuntimeError, mock torch.cuda.is_available)
- [ ] `tests/test_data_factory.py` — UPDATE existing loader dispatch tests to assert `device=self.config.device` instead of `device="cpu"`

*Existing test infrastructure (pytest, conftest.py with @pytest.mark.cuda skip) covers all phase requirements — no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU tensors in SLURM output | GPU-03 | Requires HoreKa compute node with GPU | Submit smoke sbatch with `device: "cuda"` config; grep SLURM log for `"Loaded source dataset on cuda"` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
