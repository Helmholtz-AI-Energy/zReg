"""Merge selfcal-calibrated hyperparameters from two independent HPO runs.

The "baseline_with_selfcal" pair runs need a single params dict, but selfcal
produced three separate best_params.json files:

- selfcal/kobitski_ew06_alignment  -> alignment params only (window_size,
  step, cpd_penalty, dtw_dist_fn, n_breakpoints)
- selfcal/shah_alignment           -> alignment params only
- selfcal/shah_label_transfer      -> alignment + label-transfer params
  (k_neighbours, dist_metric, smoothing, threshold)

Every baseline_with_selfcal pair uses a Kobitski embryo as source and Shah
sample-1 as target, so its alignment params draw from BOTH Kobitski and Shah
selfcal calibration, and its label-transfer params draw from the Shah
label-transfer run alone.

Merge rule (documented, deterministic — no hidden tie-breaking):

- Key present in both of the two dicts being merged:
  - both values numeric (int/float, not bool) -> arithmetic mean, rounded
    back to int if both inputs were int.
  - values are non-numeric (str/None) and equal -> keep that value.
  - values are non-numeric and differ -> conflict; fall back to the
    `defaults` dict entry for that key (the config's default_params),
    logged as a warning. This avoids arbitrarily favouring one calibration
    run over the other for categorical choices.
- Key present in only one of the two dicts -> use that dict's value
  directly (nothing to merge against).

`defaults` is consulted ONLY for conflict resolution and, at the very end of
:func:`merge_selfcal_params`, to fill any key still missing after combining
all three real sources — it is never unioned into intermediate merge steps
(`merge_two`'s key set is `set(a) | set(b)` only). Earlier versions unioned
`defaults` in at every stage, which meant Shah's calibrated label-transfer
keys (absent from the two alignment-only dicts) got a defaults-sourced
placeholder in the first merge step, then read back as a "conflict" against
Shah's real calibrated value in the second step — silently discarding it.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_NUMERIC_TYPES = (int, float)


def _is_numeric(v: object) -> bool:
    return isinstance(v, _NUMERIC_TYPES) and not isinstance(v, bool)


def merge_two(a: dict, b: dict, defaults: dict | None = None) -> dict:
    """Merge two flat param dicts using the numeric-average / fallback rule.

    Parameters
    ----------
    a, b : dict
        Param dicts to merge (order does not matter for numeric keys; for
        conflicting categorical keys neither is preferred over the other —
        both fall back to `defaults`).
    defaults : dict or None
        Fallback values used when a key conflicts non-numerically, or is
        present in neither ``a`` nor ``b``.

    Returns
    -------
    dict
        Merged param dict covering the union of all keys.
    """
    defaults = defaults or {}
    keys = set(a) | set(b)  # NOT unioned with defaults — see module docstring
    merged: dict = {}

    for key in keys:
        in_a, in_b = key in a, key in b

        if in_a and in_b:
            va, vb = a[key], b[key]
            if _is_numeric(va) and _is_numeric(vb):
                avg = (va + vb) / 2
                if isinstance(va, int) and isinstance(vb, int):
                    avg = round(avg)
                merged[key] = avg
            elif va == vb:
                merged[key] = va
            else:
                if key not in defaults:
                    raise ValueError(
                        f"merge_params: conflicting non-numeric values for "
                        f"{key!r} ({va!r} vs {vb!r}) and no default provided"
                    )
                log.warning(
                    "merge_params: conflict on %r (%r vs %r) — using default %r",
                    key, va, vb, defaults[key],
                )
                merged[key] = defaults[key]
        elif in_a:
            merged[key] = a[key]
        else:
            merged[key] = b[key]

    return merged


def merge_groundtruth_params(
    kobitski_alignment: dict,
    shah_both: dict,
    defaults: dict | None = None,
) -> dict:
    """Merge ground_truth best_params into one dict for a Kobitski-vs-Shah pair.

    Ground_truth has two HPO runs:
    - ``ground_truth/kobitski_ew06`` — alignment params only.
    - ``ground_truth/shah_sample1`` — alignment + label-transfer params (both stages).

    Since shah_sample1 is the sole source of label-transfer params, it is passed
    as both the alignment and label-transfer contributor, which is equivalent to
    calling :func:`merge_selfcal_params` with ``shah_both`` repeated twice.

    Parameters
    ----------
    kobitski_alignment : dict
        best_params.json from ground_truth/kobitski_ew06.
    shah_both : dict
        best_params.json from ground_truth/shah_sample1 (alignment + LT).
    defaults : dict or None
        Fallback values (typically the pair config's default_params).
    """
    return merge_selfcal_params(kobitski_alignment, shah_both, shah_both, defaults)


def merge_selfcal_params(
    kobitski_alignment: dict,
    shah_alignment: dict,
    shah_label_transfer: dict,
    defaults: dict | None = None,
) -> dict:
    """Merge three selfcal best_params dicts into one for a Kobitski-vs-Shah pair.

    Alignment params: mean of ``kobitski_alignment`` and ``shah_alignment``
    (numeric-average / fallback rule from :func:`merge_two`).

    Label-transfer params: taken directly from ``shah_label_transfer`` — it
    is the only run that calibrated them, so there is nothing to average
    against; any alignment keys present in ``shah_label_transfer`` are folded
    into the same averaging pass as a third numeric contributor is not
    supported, so it participates via ``merge_two`` as the "b" side after
    the alignment merge, letting its (re-)calibrated alignment values also
    contribute to the average.

    Parameters
    ----------
    kobitski_alignment : dict
        best_params.json contents from selfcal/kobitski_ew06_alignment.
    shah_alignment : dict
        best_params.json contents from selfcal/shah_alignment.
    shah_label_transfer : dict
        best_params.json contents from selfcal/shah_label_transfer.
    defaults : dict or None
        Fallback values (typically the pair config's default_params).

    Returns
    -------
    dict
        Single merged param dict ready to pass to
        ``EvaluationRunner(config, params)``. Covers the union of keys from
        all three sources plus any ``defaults`` keys touched by none of
        them.
    """
    defaults = defaults or {}
    alignment_merged = merge_two(kobitski_alignment, shah_alignment, defaults)
    combined = merge_two(alignment_merged, shah_label_transfer, defaults)
    return {**defaults, **combined}
