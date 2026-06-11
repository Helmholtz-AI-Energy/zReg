---
status: passed
phase: 28-script-integration-generate-datasets-uses-datafactory
source: [28-VERIFICATION.md]
started: 2026-06-11T15:00:00.000Z
updated: 2026-06-11T15:00:00.000Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Noise + Scaling bit-identity
expected: Re-run the refactored script on `shah_sample1`; MD5 hashes of the 6 noise/scaling CSVs match output from the pre-refactor script. All 6 hashes identical.
result: [passed] — Verified during execution (MD5 match reported in SUMMARY.md); human approved 2026-06-11.

### 2. Dropout statistical parity + reproducibility
expected: Row counts within ±1 of pre-refactor numpy version; two consecutive runs produce byte-identical output.
result: [passed] — Verified during execution (exact row-count match + reproducibility confirmed); human approved 2026-06-11.

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
