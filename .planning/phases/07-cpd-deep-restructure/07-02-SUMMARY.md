---
phase: 07-cpd-deep-restructure
plan: 02
subsystem: cpd
tags: [cpd, inheritance, refactoring, rigid, affine, nonrigid]
dependency_graph:
  requires: [07-01]
  provides: [RigidCPD, AffineCPD, NonRigidCPD, ConstrainedNonRigidCPD]
  affects: [src/zreg/cpd/]
tech_stack:
  added: []
  patterns: [inheritance, abstract-base-class, template-method]
key_files:
  created:
    - src/zreg/cpd/rigid.py
    - src/zreg/cpd/affine.py
    - src/zreg/cpd/nonrigid.py
  modified: []
decisions:
  - "All CPD variants inherit from CoherentPointDrift abstract base class"
  - "Each variant implements only _initialize and _maximization_step (no duplicated E-step)"
  - "NonRigidCPD and ConstrainedNonRigidCPD override maximization_step for custom parameter passing"
metrics:
  duration: 3m 12s
  completed: 2026-04-20
  tasks: 3
  files: 3
  lines_added: 713
---

# Phase 07 Plan 02: CPD Variant Classes Summary

CPD registration variants (Rigid, Affine, NonRigid, ConstrainedNonRigid) migrated to inherit from the abstract CoherentPointDrift base class, eliminating E-step duplication.

## One-liner

Four CPD variant classes now inherit from CoherentPointDrift ABC, each implementing only _initialize and _maximization_step while sharing the common E-step and registration loop from the base class.

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create RigidCPD variant | f9666c2 | src/zreg/cpd/rigid.py |
| 2 | Create AffineCPD variant | a531d7a | src/zreg/cpd/affine.py |
| 3 | Create NonRigidCPD and ConstrainedNonRigidCPD | eac4c98 | src/zreg/cpd/nonrigid.py |

## Changes Made

### Task 1: RigidCPD (src/zreg/cpd/rigid.py)

Created RigidCPD class inheriting from CoherentPointDrift:
- Implements `_initialize` for rigid registration parameter setup
- Implements `_maximization_step` for rotation, translation, and optional scale computation
- Overrides `maximization_step` to pass `update_scale` parameter
- Maintains `reset_transform` method for transformation reset
- Uses SVD for optimal rotation computation (Myronenko & Song 2010)

### Task 2: AffineCPD (src/zreg/cpd/affine.py)

Created AffineCPD class inheriting from CoherentPointDrift:
- Implements `_initialize` for affine registration parameter setup
- Implements `_maximization_step` for affine matrix and translation computation
- Uses linear solve for affine parameter estimation
- Properly clamps sigma2 to prevent numerical instability

### Task 3: NonRigidCPD and ConstrainedNonRigidCPD (src/zreg/cpd/nonrigid.py)

Created two non-rigid CPD classes:

**NonRigidCPD:**
- Implements smooth deformations via Gaussian RBF kernel
- Overrides `set_source` to initialize transformation object with kernel
- Overrides `maximization_step` to pass transformation object and regularization parameter
- Uses linear solve for deformation weight computation

**ConstrainedNonRigidCPD:**
- Extends non-rigid registration with point correspondence constraints
- Implements constraint matrices (p_tilde, p1_tilde, px_tilde) in `_initialize`
- Adds alpha parameter for prior reliability control
- References ECPD paper (Golyanik et al. 2016)

## Verification Results

All acceptance criteria verified:
- All four variants inherit from `CoherentPointDrift`
- Each variant imports base class via `from .base import CoherentPointDrift`
- Each variant implements `_initialize` and `_maximization_step`
- RigidCPD maintains `reset_transform` and `update_scale` functionality
- All `__all__` exports correctly defined

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

Created files verified:
- FOUND: src/zreg/cpd/rigid.py
- FOUND: src/zreg/cpd/affine.py
- FOUND: src/zreg/cpd/nonrigid.py

Commits verified:
- FOUND: f9666c2
- FOUND: a531d7a
- FOUND: eac4c98
