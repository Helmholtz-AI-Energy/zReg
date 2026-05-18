# Phase 14: Synthetic Data Generators - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 14-Synthetic Data Generators
**Areas discussed:** Generator input model, Module structure, Label representation, Test strategy

---

## Generator Input Model

| Option | Description | Selected |
|--------|-------------|----------|
| Wrap real data | Accept existing trajectory dict, apply transforms, return transformed copy | |
| From scratch | Generate point clouds from random/geometric primitives given n_points, n_frames | |
| Both | From-scratch factory + immutable corruption wrappers | ✓ |

**User's choice:** Both

---

**Follow-up: from-scratch geometry**

| Option | Description | Selected |
|--------|-------------|----------|
| Uniform random in cube | torch.rand(n_points, 3) scaled to bounding box | |
| Gaussian blob | Points from N(0, I) — bell-curve density | ✓ |
| You decide | Leave to planner | |

**User's choice:** Gaussian blob

---

**Follow-up: corruption wrapper mutability**

| Option | Description | Selected |
|--------|-------------|----------|
| Return new dict | Immutable — returns fresh dict, source unchanged | ✓ |
| Mutate in-place | Modifies input dict directly | |

**User's choice:** Return new dict (Recommended)

---

**Follow-up: factory output granularity**

| Option | Description | Selected |
|--------|-------------|----------|
| Multi-frame trajectory | Returns dict[int, zRegPointCloud] directly with N frames | ✓ |
| Single point cloud | Returns one zRegPointCloud; caller assembles dict | |

**User's choice:** Multi-frame trajectory (Recommended)

---

## Module Structure

| Option | Description | Selected |
|--------|-------------|----------|
| By category | generators.py, transforms.py, corruption.py, labels.py + __init__.py | ✓ |
| Single file | All functions in generators.py | |
| Flat __init__.py | Everything in __init__.py | |

**User's choice:** By category (Recommended)

---

**Follow-up: importable package**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, importable | __init__.py re-exports all public functions | ✓ |
| Plain directory | No __init__.py; runners import from individual files | |

**User's choice:** Yes, importable package (Recommended)

---

## Label Representation

| Option | Description | Selected |
|--------|-------------|----------|
| color field | zRegPointCloud['color'] — consistent with real data pipeline | ✓ |
| id field | Use id for integer class labels | |
| New field | Add 'labels' or 'celltype' key | |

**User's choice:** color field (Recommended)

---

**Follow-up: label format**

| Option | Description | Selected |
|--------|-------------|----------|
| Integer tensor (N,) | torch.long, one class ID per point | ✓ |
| Float tensor (N, n_classes) | One-hot / soft assignments | |
| Integer tensor (N, 1) | Same as (N,) with extra dim | |

**User's choice:** Integer tensor, shape (N,) (Recommended)

---

**Follow-up: label assignment strategy**

| Option | Description | Selected |
|--------|-------------|----------|
| Voronoi-based | k random seeds, assign each point to nearest seed | |
| Random uniform | Each point independently draws class ID | |
| You decide | Leave to planner | ✓ |

**User's choice:** You decide

---

**Follow-up: remove_labels() behavior**

| Option | Description | Selected |
|--------|-------------|----------|
| Set color to None | Matches default __init__ behavior | ✓ |
| Set color to empty tensor | Avoids None checks but requires size checks | |

**User's choice:** Set color to None (Recommended)

---

## Test Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| tests/test_generators.py | Project convention; needs sys.path insert | ✓ |
| eval/generators/tests/ | Co-located, breaks single tests/ convention | |
| No unit tests | Skip unit tests; rely on runner integration | |

**User's choice:** tests/test_generators.py (Recommended)

---

**Follow-up: import mechanism**

| Option | Description | Selected |
|--------|-------------|----------|
| conftest.py sys.path insert | One-time repo-root insert; all tests benefit | ✓ |
| Per-file sys.path | test_generators.py does its own insert | |
| Editable install of eval/ | pip install -e eval/ — clean but over-engineered | |

**User's choice:** conftest.py sys.path insert (Recommended)

---

## Claude's Discretion

- **Label assignment strategy** — Voronoi-based or random uniform; planner picks whichever is simpler to implement correctly with the seed contract.

## Deferred Ideas

None — discussion stayed within phase scope.
