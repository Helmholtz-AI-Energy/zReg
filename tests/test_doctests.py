"""Doctest gate for every ``zreg`` docstring (Phase 64, DOC-01).

Two independent checks keep the docstring examples runnable:

- ``test_doctest_modules_src_zreg_pass`` runs the literal command
  ``pytest --doctest-modules src/zreg`` in a child process (``sys.executable``,
  list arguments, no shell) and requires exit code 0, no ``N failed`` /
  ``N error(s)`` summary and at least ``MIN_DOCTEST_ITEMS`` passed + skipped
  items.
- ``test_docstring_example_self_contained`` runs every docstring found in
  every ``zreg`` module on its own with an EMPTY namespace, the way the Sphinx
  ``doctest`` builder (``tox -e doctests``) sees it: no conftest fixtures, no
  module globals.  Torch global state (default device, float32 matmul
  precision, RNG) is snapshotted and restored around each example.

The Open3D example in ``zreg.core.dataset.open3d_to_zreg`` is marked
``# doctest: +SKIP`` inline so that pytest and Sphinx both skip it when
Open3D is missing.  ``test_open3d_docstring_example_runs_when_available``
strips the SKIP flags and executes it whenever ``open3d`` imports, so the
guarded example cannot rot.

Discovery runs at import time but never raises: the package walk, each
module import and each ``DocTestFinder.find`` are wrapped individually, and
failures are collected in ``IMPORT_FAILURES`` and reported by
``test_all_zreg_modules_import_for_doctests``.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# 31 docstrings with examples existed when the gate was added (Phase 64:
# 30 passed + 1 skipped under ``pytest --doctest-modules src/zreg``).
MIN_DOCTEST_ITEMS = 31

DOCTESTS: list[doctest.DocTest] = []
IMPORT_FAILURES: list[tuple[str, str]] = []


def _discover() -> None:
    """Fill ``DOCTESTS`` / ``IMPORT_FAILURES`` without ever raising."""
    try:
        import zreg
    except Exception as exc:
        IMPORT_FAILURES.append(("zreg", repr(exc)))
        return

    def _onerror(name: str) -> None:
        IMPORT_FAILURES.append((name, f"walk_packages could not import: {sys.exc_info()[1]!r}"))

    try:
        names = ["zreg"] + [
            info.name for info in pkgutil.walk_packages(zreg.__path__, "zreg.", onerror=_onerror)
        ]
    except Exception as exc:
        IMPORT_FAILURES.append(("zreg (walk_packages)", repr(exc)))
        names = ["zreg"]

    finder = doctest.DocTestFinder(exclude_empty=True)
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            IMPORT_FAILURES.append((name, repr(exc)))
            continue
        try:
            found = finder.find(module, name, globs={})
        except Exception as exc:
            IMPORT_FAILURES.append((name, f"DocTestFinder failed: {exc!r}"))
            continue
        DOCTESTS.extend(test for test in found if test.examples)


_discover()


@pytest.fixture
def restore_torch_state():
    """Snapshot and restore torch global state touched by docstring examples."""
    import torch

    # In torch 2.9 an earlier ``set_default_device(None)`` (tests/test_config.py
    # resets that way) leaves ``get_default_device()`` raising AttributeError.
    # Examples then run with an explicit cpu default, which is what torch uses
    # without a device context, and the prior None state is put back afterwards.
    try:
        device = torch.get_default_device()
    except AttributeError:
        device = None
        torch.set_default_device("cpu")
    precision = torch.get_float32_matmul_precision()
    with torch.random.fork_rng(devices=[]):
        yield
    if device is None:
        torch.set_default_device(None)
    else:
        try:
            changed = torch.get_default_device() != device
        except AttributeError:
            changed = True
        if changed:
            torch.set_default_device(device)
    torch.set_float32_matmul_precision(precision)


def _run(test: doctest.DocTest) -> tuple[doctest.TestResults, str]:
    test.globs = {}
    out: list[str] = []
    runner = doctest.DocTestRunner(verbose=False)
    result = runner.run(test, out=out.append, clear_globs=True)
    return result, "".join(out)


def test_doctest_modules_src_zreg_pass() -> None:
    """The literal ``pytest --doctest-modules src/zreg`` run is green."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--doctest-modules",
            "src/zreg",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--rootdir",
            str(REPO),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"pytest --doctest-modules src/zreg failed:\n{output}"
    assert not re.search(r"\b\d+ failed\b", output), output
    assert not re.search(r"\b\d+ errors?\b", output), output
    passed = sum(int(n) for n in re.findall(r"\b(\d+) passed\b", output))
    skipped = sum(int(n) for n in re.findall(r"\b(\d+) skipped\b", output))
    assert passed + skipped >= MIN_DOCTEST_ITEMS, (
        f"only {passed} passed + {skipped} skipped doctest items (expected >= {MIN_DOCTEST_ITEMS}):\n{output}"
    )


def test_all_zreg_modules_import_for_doctests() -> None:
    """Every zreg module imports, so no docstring silently drops out of the gate."""
    assert IMPORT_FAILURES == [], "zreg modules failed during doctest discovery:\n" + "\n".join(
        f"{name}: {err}" for name, err in IMPORT_FAILURES
    )


def test_doctest_discovery_not_vacuous() -> None:
    """Discovery finds the known docstrings with examples."""
    assert len(DOCTESTS) >= MIN_DOCTEST_ITEMS, f"only {len(DOCTESTS)} docstrings with examples found"
    assert sum(len(test.examples) for test in DOCTESTS) > 0


@pytest.mark.parametrize("test", DOCTESTS, ids=[test.name for test in DOCTESTS])
def test_docstring_example_self_contained(test: doctest.DocTest, restore_torch_state) -> None:
    """Each docstring example runs on its own with an empty namespace (Sphinx-equivalent)."""
    result, output = _run(test)
    assert result.failed == 0, f"{test.name}: {result.failed}/{result.attempted} examples failed\n{output}"


def test_open3d_docstring_example_runs_when_available(restore_torch_state) -> None:
    """The +SKIP-guarded Open3D example really runs where Open3D is installed."""
    pytest.importorskip("open3d")
    dataset = importlib.import_module("zreg.core.dataset")
    finder = doctest.DocTestFinder(exclude_empty=True)
    tests = [
        test
        for test in finder.find(dataset, "zreg.core.dataset", globs={})
        if test.name == "zreg.core.dataset.open3d_to_zreg"
    ]
    assert len(tests) == 1, [test.name for test in tests]
    test = tests[0]
    assert test.examples
    for example in test.examples:
        assert example.options.get(doctest.SKIP) is True, (
            f"open3d_to_zreg example line {example.lineno} lost its inline +SKIP guard: {example.source!r}"
        )
        example.options[doctest.SKIP] = False
    result, output = _run(test)
    assert result.failed == 0, output
    assert result.attempted >= 5, f"only {result.attempted} Open3D examples attempted"
