# Phase 56 - Deferred Items

Out-of-scope discoveries logged during plan execution (not fixed, per scope boundary rules).

## From Plan 56-02

- **`tests/test_icp_registration.py::TestICPRegistration::test_icp_translation_recovery` and `test_icp_rotation_recovery` fail only when run as part of the full suite (`python -m pytest` with no filter), but pass in isolation and when run alongside `tests/test_generators.py`.** This is pre-existing test-order-dependent flakiness (likely global RNG/Open3D state leakage from an unrelated test file running earlier in full-suite order) and is unrelated to this plan's `generate_labels()` rework — ICP registration code was not touched by Plan 56-02. Confirmed via: `pytest tests/test_icp_registration.py` alone (13/13 pass) and `pytest tests/test_icp_registration.py::...test_icp_rotation_recovery tests/test_generators.py` together (53/53 pass). Not fixed — out of scope for this plan.
