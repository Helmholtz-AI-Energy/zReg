"""Extract calibration data from search_history.json files produced by crashed HoreKa jobs.

Reads one or more `search_history.json` files, groups CPD trial durations by
`window_size`, computes per-window means, and linearly interpolates/extrapolates
missing window sizes (5, 10, 20). Prints a JSON snippet to stdout ready to paste
into `horeka.json`.

Usage
-----
    python baseline_experiments/scripts/extract_calibration.py path/to/search_history.json
    python baseline_experiments/scripts/extract_calibration.py job1/search_history.json job2/search_history.json

Output schema (cpd_trial_seconds values are always dicts):
    {
      "cpd_trial_seconds": {
        "5":  {"seconds": 130.5, "extrapolated": false},
        "10": {"seconds": 176.5, "extrapolated": false},
        "20": {"seconds": 278.6, "extrapolated": true}
      },
      "no_cpd_trial_seconds": 4.0,
      "label_transfer_overhead_seconds": null,
      "_note": "label_transfer_overhead_seconds requires manual measurement",
      "_source": "[VERIFY ON HOREKA] Extracted from crashed job ..."
    }

The output format is accepted by aggregate_cost.py --calibration via the dict-form
loader: `v["seconds"] if isinstance(v, dict) else v`.

Threat T-54-06: malformed trial records (missing both duration fields) are skipped
with a stderr warning rather than raising an uncaught exception.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The three window sizes that appear in the baseline_experiments search spaces.
TARGET_WINDOW_SIZES = [5, 10, 20]


def _duration_seconds(trial: dict) -> float | None:
    """Extract duration from a trial record; prefer duration_seconds, fall back to elapsed_seconds."""
    if "duration_seconds" in trial:
        return trial["duration_seconds"]
    if "elapsed_seconds" in trial:
        return trial["elapsed_seconds"]
    return None


def _load_trials(files: list[str]) -> list[dict]:
    """Load and merge trial records from one or more search_history.json files."""
    all_trials: list[dict] = []
    for path_str in files:
        path = Path(path_str)
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(f"ERROR: could not parse JSON in {path}: {exc}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(data, list):
            print(f"WARNING: {path} did not contain a JSON array — skipping", file=sys.stderr)
            continue
        all_trials.extend(data)
    return all_trials


def _group_trials(trials: list[dict]) -> tuple[dict[int, list[float]], list[float]]:
    """Separate trials into CPD (grouped by window_size) and no-CPD groups.

    Returns:
        cpd_groups: {window_size: [duration_seconds, ...]}
        no_cpd_durations: [duration_seconds, ...]
    """
    cpd_groups: dict[int, list[float]] = {}
    no_cpd_durations: list[float] = []
    skipped = 0

    for i, trial in enumerate(trials):
        params = trial.get("params", {})
        duration = _duration_seconds(trial)
        if duration is None:
            # T-54-06: skip trials missing both duration fields with a warning.
            print(
                f"WARNING: trial {i} missing both 'duration_seconds' and 'elapsed_seconds' — skipping",
                file=sys.stderr,
            )
            skipped += 1
            continue

        cpd_penalty = params.get("cpd_penalty")
        if cpd_penalty is None:
            # No-CPD trial: contributes to no_cpd_trial_seconds, not CPD timing.
            no_cpd_durations.append(float(duration))
        else:
            window_size = params.get("window_size")
            if window_size is None:
                print(
                    f"WARNING: CPD trial {i} missing 'window_size' — skipping",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            window_size = int(window_size)
            cpd_groups.setdefault(window_size, []).append(float(duration))

    if skipped:
        print(f"WARNING: skipped {skipped} trial(s) due to missing fields", file=sys.stderr)

    return cpd_groups, no_cpd_durations


def _linear_extrapolate(
    observed: dict[int, float],
    targets: list[int],
) -> dict[int, tuple[float, bool]]:
    """Compute values for all target window sizes, interpolating/extrapolating missing ones.

    Args:
        observed: {window_size: mean_seconds} for observed window sizes
        targets: list of target window sizes (e.g. [5, 10, 20])

    Returns:
        {window_size: (seconds, extrapolated)} for all targets
    """
    result: dict[int, tuple[float, bool]] = {}

    if len(observed) == 0:
        # No data at all — cannot compute anything.
        raise ValueError("No observed window sizes to interpolate from.")

    if len(observed) == 1:
        # Only one point: use it directly for observed, copy value for all others
        # (we can't determine a slope, so flat extrapolation is the safest fallback).
        (only_ws, only_val) = next(iter(observed.items()))
        for ws in targets:
            extrapolated = ws not in observed
            result[ws] = (only_val, extrapolated)
        return result

    # Fit a line through all observed points using least-squares (or exact if 2 points).
    xs = sorted(observed.keys())
    ys = [observed[x] for x in xs]

    # Linear regression: y = a*x + b
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xx = sum(x * x for x in xs)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        # Degenerate: all x are equal (shouldn't happen after dedup, but be safe).
        a = 0.0
        b = sum_y / n
    else:
        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n

    for ws in targets:
        if ws in observed:
            result[ws] = (observed[ws], False)
        else:
            predicted = a * ws + b
            result[ws] = (predicted, True)

    return result


def extract(files: list[str]) -> dict:
    """Core extraction logic — callable from tests without subprocess.

    Args:
        files: list of paths to search_history.json files

    Returns:
        Output dict suitable for json.dumps() and pasting into horeka.json.

    Raises:
        SystemExit(1) if no readable trial data is found.
    """
    trials = _load_trials(files)
    cpd_groups, no_cpd_durations = _group_trials(trials)

    # Check we have something to work with.
    if not cpd_groups and not no_cpd_durations:
        print(
            "ERROR: no readable trial records found in the provided files. "
            "Each trial must have 'params.window_size' (for CPD trials) or "
            "'params.cpd_penalty': null (for no-CPD trials) and at least one "
            "of 'duration_seconds' / 'elapsed_seconds'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compute per-window means for CPD trials.
    observed_means: dict[int, float] = {
        ws: sum(durations) / len(durations)
        for ws, durations in cpd_groups.items()
    }

    # Interpolate/extrapolate to all three target window sizes.
    cpd_values: dict[int, tuple[float, bool]] = {}
    if observed_means:
        cpd_values = _linear_extrapolate(observed_means, TARGET_WINDOW_SIZES)
    else:
        # No CPD trials at all — mark all as null (will be emitted as null below).
        pass

    # Build the cpd_trial_seconds object.
    cpd_trial_seconds: dict[str, dict] = {}
    for ws in TARGET_WINDOW_SIZES:
        if ws in cpd_values:
            seconds, extrapolated = cpd_values[ws]
            cpd_trial_seconds[str(ws)] = {"seconds": round(seconds, 4), "extrapolated": extrapolated}
        else:
            # No CPD data at all for this key.
            cpd_trial_seconds[str(ws)] = {"seconds": None, "extrapolated": True}

    # no_cpd_trial_seconds: mean of no-CPD trial durations, or null.
    no_cpd_mean: float | None = (
        sum(no_cpd_durations) / len(no_cpd_durations) if no_cpd_durations else None
    )

    output = {
        "cpd_trial_seconds": cpd_trial_seconds,
        "no_cpd_trial_seconds": round(no_cpd_mean, 4) if no_cpd_mean is not None else None,
        "label_transfer_overhead_seconds": None,
        "_note": "label_transfer_overhead_seconds requires manual measurement",
        "_source": (
            "[VERIFY ON HOREKA] Extracted from crashed job search_history.json"
            " — fill _source with job ID and date before committing"
        ),
    }
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CPD calibration timing from search_history.json files produced "
            "by HoreKa HPO jobs. Prints a JSON snippet to stdout for pasting into "
            "baseline_experiments/calibrations/horeka.json."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more paths to search_history.json files.",
    )
    args = parser.parse_args(argv)

    output = extract(args.files)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
