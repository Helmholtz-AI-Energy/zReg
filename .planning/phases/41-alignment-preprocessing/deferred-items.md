# Deferred Items — Phase 41

Out-of-scope discoveries logged during execution. NOT fixed by this plan.

## Pre-existing test failures (unrelated to plan 41-01)

**`tests/test_eval_config.py::TestEvalConfigAlignmentMethodValidation`** — 3 failing tests:
- `test_alignment_method_invalid_value_raises`
- `test_alignment_method_case_sensitive`
- `test_alignment_method_empty_string_raises`

**Cause:** Stale test expectations. These tests assert the validator error message
`"alignment_method must be 'cpd' or 'icp'"`, but the `validate_alignment_method`
validator in `eval/config.py` (line 246) was updated in Phase 40 (SWD support) to emit
`"alignment_method must be 'cpd', 'icp', or 'swd'"`. The tests were not updated at that time.

**Confirmed pre-existing:** At base commit `5853e6e` (before plan 41-01) the config already
emits the 3-way message while the tests still expect the 2-way message. Plan 41-01 did not
touch the `validate_alignment_method` validator (verified: `git diff 5853e6e..HEAD -- eval/config.py`
contains no change to the message).

**Suggested fix (future plan):** Update the three test assertions' `match=` regex in
`tests/test_eval_config.py` to `"alignment_method must be 'cpd', 'icp', or 'swd'"`.
