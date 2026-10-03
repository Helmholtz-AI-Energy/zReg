"""Strict RFC-8259 JSON writing for result artifacts (Phase 63 IN-01, D-04).

Since Phase 59 non-finite metric values are part of the result contract:
failed HPO trials score ``-inf`` and unavailable stages or metrics are
reported as ``+inf`` (normalised ``0.0``).  ``json.dump`` writes those as the
literals ``Infinity`` / ``-Infinity`` / ``NaN``, which are not JSON and are
rejected by strict parsers (jq, JavaScript ``JSON.parse``, many JSON
libraries).

The writers in ``eval.runners`` therefore sanitise before dumping:
non-finite floats become ``null`` and the dotted paths of the replaced values
are returned so that the caller can record them (``non_finite_fields``).
Dumping disables ``allow_nan``, so a non-finite value that escapes the
sanitiser raises ``ValueError`` instead of producing invalid JSON.

In-memory objects (``StageMetrics``, ``EvalReport``, ``Trial`` ...) keep their
non-finite values; only the on-disk representation changes.
"""

from __future__ import annotations

import json
import math
from typing import IO, Any

__all__ = ["sanitize_non_finite", "dump_strict"]


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def sanitize_non_finite(obj: Any, path: str = "") -> tuple[Any, list[str]]:
    """Return a copy of ``obj`` with non-finite floats replaced by ``None``.

    Parameters
    ----------
    obj : Any
        JSON-like value: dicts, lists, tuples and scalars.  Dicts and
        lists/tuples are walked recursively; tuples become lists (as
        ``json.dump`` would write them).  The input is never mutated.
    path : str, optional
        Path prefix of ``obj``.  Dict keys are joined with ``"."`` and list
        indices appended as ``"[i]"``.  Default ``""`` (top level).

    Returns
    -------
    clean : Any
        New containers with every non-finite ``float`` (``inf``, ``-inf``,
        ``nan``) replaced by ``None``.  Other values are passed through.
    paths : list[str]
        Paths of the replaced values in traversal order, e.g.
        ``["metrics.chamfer_distance", "history[2].score"]``.  A non-finite
        top-level scalar is recorded with the given ``path`` (``""`` by
        default).
    """
    paths: list[str] = []

    def _walk(value: Any, here: str) -> Any:
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            paths.append(here)
            return None
        if isinstance(value, dict):
            return {k: _walk(v, _join(here, k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_walk(v, f"{here}[{i}]") for i, v in enumerate(value)]
        return value

    clean = _walk(obj, path)
    return clean, paths


def dump_strict(obj: Any, fp: IO[str], **kwargs: Any) -> list[str]:
    """Sanitise ``obj`` and write it to ``fp`` as strict JSON.

    Parameters
    ----------
    obj : Any
        JSON-like value (see ``sanitize_non_finite``).
    fp : file-like
        Text file opened for writing.
    **kwargs
        Forwarded to ``json.dump`` (e.g. ``indent=2``).  ``allow_nan`` is
        always ``False``.

    Returns
    -------
    list[str]
        Paths of the values written as ``null`` because they were not finite.
        No marker is added to the output; callers that need one add it
        themselves before dumping.
    """
    clean, paths = sanitize_non_finite(obj)
    kwargs.pop("allow_nan", None)
    json.dump(clean, fp, allow_nan=False, **kwargs)
    return paths
