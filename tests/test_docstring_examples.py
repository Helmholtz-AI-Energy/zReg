"""Regression tests for runnable transform docstrings (CPD-09, U2-12 / U2-13).

On the 6c1c37f baseline:

- U2-12: the ``TPSTransformation`` docstring example built ``v`` with shape
  ``(n_control, d)``, but ``v`` lives in the null-space basis of the control
  points and must have shape ``(n_control - d - 1, d)``; the example raised a
  shape error.
- U2-13: docstrings in ``zreg.core.transforms``,
  ``zreg.core.transforms.homogeneous`` and ``zreg.data_generation.transforms``
  imported from the non-existent ``zreg.transforms`` / ``zreg.dataset``.

Examples are parsed with :mod:`doctest`, not with regular expressions.
"""

from __future__ import annotations

import doctest
import importlib

import pytest

# zreg must be imported before torch (libomp ordering on macOS ARM).
import zreg  # noqa: F401
import torch

DOCSTRING_MODULES = [
    "zreg.core.transforms",
    "zreg.core.transforms.homogeneous",
    "zreg.data_generation.transforms",
]


def _import_example_lines() -> list[tuple[str, str]]:
    """Collect every ``from zreg... import`` / ``import zreg...`` docstring example."""
    finder = doctest.DocTestFinder(exclude_empty=True)
    lines: list[tuple[str, str]] = []
    for mod_name in DOCSTRING_MODULES:
        module = importlib.import_module(mod_name)
        for test in finder.find(module):
            for example in test.examples:
                for line in example.source.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("from zreg") or stripped.startswith("import zreg"):
                        lines.append((test.name, stripped))
    return lines


_IMPORT_LINES = _import_example_lines()


def test_import_lines_were_collected():
    """The collector finds the docstring import lines (guards against a vacuous test)."""
    assert len(_IMPORT_LINES) >= 5


@pytest.mark.parametrize(
    "where,line", _IMPORT_LINES, ids=[f"{w}:{ln}" for w, ln in _IMPORT_LINES]
)
def test_docstring_import_lines_resolve(where, line):
    """Each docstring import line executes (baseline: ModuleNotFoundError)."""
    exec(compile(line, f"<docstring {where}>", "exec"), {})


def test_tps_docstring_example_runs():
    """The TPSTransformation class docstring example runs end to end (baseline: RuntimeError)."""
    from zreg.core.transforms import TPSTransformation

    examples = doctest.DocTestParser().get_examples(TPSTransformation.__doc__)
    assert examples, "TPSTransformation docstring has no examples"
    namespace = {"torch": torch, "TPSTransformation": TPSTransformation}
    for example in examples:
        exec(compile(example.source, "<TPSTransformation docstring>", "exec"), namespace)
    assert namespace["transformed"].shape == (50, 3)
