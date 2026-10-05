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

# The child ``pytest --doctest-modules`` run takes ~10 s; a hanging example
# (CUDA/MPI init on a cluster node, a blocking Open3D call) must fail the gate
# instead of hanging the whole session.
DOCTEST_SUBPROCESS_TIMEOUT_S = 600

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


_MISSING = object()

# Private torch internals the device-state helpers below rely on (checked
# against torch 2.9).  torch is not version-pinned, so they are probed once
# and a rename fails with one clear message naming the torch version instead
# of an opaque AttributeError/ImportError in every fixture setup.
_TORCH_DEVICE_INTERNALS = (
    ("torch", "_GLOBAL_DEVICE_CONTEXT"),
    ("torch._C", "_len_torch_function_stack"),
    ("torch.overrides", "_get_current_function_mode_stack"),
    ("torch.overrides", "_pop_mode"),
    ("torch.overrides", "_push_mode"),
    ("torch.utils._device", "CURRENT_DEVICE"),
    ("torch.utils._device", "DeviceContext"),
)


def _missing_torch_device_internals() -> list[str]:
    """Return the dotted names from ``_TORCH_DEVICE_INTERNALS`` this torch lacks."""
    missing = []
    for module_name, attr in _TORCH_DEVICE_INTERNALS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(f"{module_name} (module)")
            continue
        if not hasattr(module, attr):
            missing.append(f"{module_name}.{attr}")
    return missing


def _require_torch_device_internals() -> None:
    missing = _missing_torch_device_internals()
    if missing:
        import torch

        pytest.fail(
            f"tests/test_doctests.py device-state helpers rely on private torch internals that "
            f"torch {torch.__version__} no longer provides: {', '.join(missing)}. "
            f"Update _device_state / _restore_device_state for this torch version.",
            pytrace=False,
        )


def _device_state():
    """Return the raw default-device state: the global slot and the mode stack.

    ``torch.get_default_device()`` cannot tell "no device context" from "a cpu
    ``DeviceContext`` on the torch-function mode stack" (both report ``cpu``),
    yet the latter forces every factory call without ``device=`` onto cpu.
    So the fixture compares the actual ``DeviceContext`` object and the mode
    stack, not the reported device.
    """
    _require_torch_device_internals()
    import torch
    from torch.overrides import _get_current_function_mode_stack

    slot = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", _MISSING)
    return slot, list(_get_current_function_mode_stack())


def _set_device_slot(slot) -> None:
    import torch

    if slot is _MISSING:
        torch._GLOBAL_DEVICE_CONTEXT.__dict__.pop("device_context", None)
    else:
        torch._GLOBAL_DEVICE_CONTEXT.device_context = slot


def _restore_device_state(slot_before, stack_before) -> bool:
    """Put back the snapshotted slot and mode stack; return True if anything had leaked.

    The torch-function mode stack is rebuilt from the snapshot: every mode is
    popped and the snapshotted modes are pushed back in order.  That removes a
    leaked ``set_default_device`` context, an un-exited ``torch.device(...)``
    block and any other leaked ``TorchFunctionMode`` alike.  A leaked
    ``DeviceContext`` sits at the BOTTOM of the stack (``DeviceContext.__enter__``
    inserts it there), so popping only the extra top entries would remove the
    wrong mode.  ``torch.utils._device.CURRENT_DEVICE`` is reset to match the
    ``DeviceContext`` left on the rebuilt stack (``None`` if there is none).
    """
    import torch.utils._device as torch_device
    from torch._C import _len_torch_function_stack
    from torch.overrides import _pop_mode, _push_mode

    current_slot, current_stack = _device_state()
    # A missing slot and a None slot both mean "no default-device context"
    # (the fixture swaps a None slot for a missing one while examples run).
    current_ctx = None if current_slot is _MISSING else current_slot
    before_ctx = None if slot_before is _MISSING else slot_before
    leaked = current_ctx is not before_ctx or [id(mode) for mode in current_stack] != [
        id(mode) for mode in stack_before
    ]
    if leaked:
        while _len_torch_function_stack() > 0:
            _pop_mode()
        for mode in stack_before:
            _push_mode(mode)
        torch_device.CURRENT_DEVICE = next(
            (mode.device for mode in stack_before if isinstance(mode, torch_device.DeviceContext)),
            None,
        )
    _set_device_slot(slot_before)
    return leaked


@pytest.fixture
def restore_torch_state():
    """Snapshot and restore torch global state touched by docstring examples.

    If an example leaves a ``DeviceContext`` (in the default-device slot or
    from an un-exited ``torch.device(...)`` block) or any other torch-function
    mode behind, the slot and the whole mode stack are first restored to the
    snapshot, so later tests do not inherit the mode, and then the test fails.
    """
    import torch

    slot_before, stack_before = _device_state()
    # In torch 2.9 an earlier ``set_default_device(None)`` (tests/test_config.py
    # resets that way) leaves the slot set to None, which makes
    # ``get_default_device()`` raise AttributeError.  Examples run in the
    # pristine state (no slot, no context) and the None slot is put back after.
    if slot_before is None:
        _set_device_slot(_MISSING)
    precision = torch.get_float32_matmul_precision()
    with torch.random.fork_rng(devices=[]):
        yield
    leaked = _restore_device_state(slot_before, stack_before)
    torch.set_float32_matmul_precision(precision)
    if leaked:
        pytest.fail("docstring example leaked a torch default-device context / function mode")


def _run(test: doctest.DocTest) -> tuple[doctest.TestResults, str]:
    test.globs = {}
    out: list[str] = []
    runner = doctest.DocTestRunner(verbose=False)
    result = runner.run(test, out=out.append, clear_globs=True)
    return result, "".join(out)


def test_doctest_modules_src_zreg_pass() -> None:
    """The literal ``pytest --doctest-modules src/zreg`` run is green."""
    try:
        proc = _run_doctest_modules()
    except subprocess.TimeoutExpired as exc:
        partial = "".join(
            out.decode(errors="replace") if isinstance(out, bytes) else out
            for out in (exc.stdout, exc.stderr)
            if out
        )
        pytest.fail(
            f"pytest --doctest-modules src/zreg did not finish within {DOCTEST_SUBPROCESS_TIMEOUT_S} s"
            f" (a docstring example hangs?):\n{partial}"
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


def _run_doctest_modules() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
        timeout=DOCTEST_SUBPROCESS_TIMEOUT_S,
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


def test_config_docstrings_leave_no_device_context() -> None:
    """The ``zreg.config`` examples leave the device slot and mode stack untouched."""
    import torch

    config_tests = [test for test in DOCTESTS if test.name.startswith("zreg.config")]
    assert config_tests, "no zreg.config docstrings discovered"
    slot_before, stack_before = _device_state()
    # Not using restore_torch_state (it would mask the leak this test checks),
    # so restore matmul precision and RNG here in case an example fails early.
    precision = torch.get_float32_matmul_precision()
    if slot_before is None:
        _set_device_slot(_MISSING)
    pristine_slot, pristine_stack = _device_state()
    try:
        with torch.random.fork_rng(devices=[]):
            for test in config_tests:
                result, output = _run(test)
                assert result.failed == 0, output
                slot_after, stack_after = _device_state()
                assert slot_after is pristine_slot, f"{test.name} changed the default-device slot"
                assert [id(m) for m in stack_after] == [id(m) for m in pristine_stack], (
                    f"{test.name} left a torch-function mode on the stack: {stack_after!r}"
                )
    finally:
        _restore_device_state(slot_before, stack_before)
        torch.set_float32_matmul_precision(precision)


def test_torch_device_internals_available() -> None:
    """The private torch names the device-state helpers use still exist in this torch."""
    import torch

    missing = _missing_torch_device_internals()
    assert missing == [], (
        f"torch {torch.__version__} lacks private internals used by restore_torch_state: {missing}"
    )


def test_restore_detects_and_removes_leaked_device_context() -> None:
    """``set_default_device(previous)`` reports cpu again but leaks a DeviceContext; the fixture catches it."""
    import torch

    slot_before, stack_before = _device_state()
    try:
        torch.set_default_device("cpu")
        assert _restore_device_state(slot_before, stack_before) is True
        slot_after, stack_after = _device_state()
        assert slot_after is slot_before
        assert [id(m) for m in stack_after] == [id(m) for m in stack_before]
        assert _restore_device_state(slot_before, stack_before) is False
    finally:
        _restore_device_state(slot_before, stack_before)


def test_restore_removes_leaked_scoped_device_context() -> None:
    """An un-exited ``torch.device(...)`` block (not in the slot) is detected AND removed."""
    import torch
    import torch.utils._device as torch_device

    slot_before, stack_before = _device_state()
    current_device_before = torch_device.CURRENT_DEVICE
    try:
        torch.device("meta").__enter__()
        assert torch.zeros(1).device.type == "meta"
        assert _restore_device_state(slot_before, stack_before) is True
        slot_after, stack_after = _device_state()
        assert slot_after is slot_before
        assert [id(m) for m in stack_after] == [id(m) for m in stack_before]
        assert torch_device.CURRENT_DEVICE == current_device_before
        assert torch.zeros(1).device.type != "meta"
        assert _restore_device_state(slot_before, stack_before) is False
    finally:
        _restore_device_state(slot_before, stack_before)


def test_restore_removes_leaked_function_mode() -> None:
    """A leaked non-device ``TorchFunctionMode`` is detected AND removed."""
    import torch
    from torch.overrides import TorchFunctionMode

    slot_before, stack_before = _device_state()
    try:
        TorchFunctionMode().__enter__()
        assert _restore_device_state(slot_before, stack_before) is True
        _, stack_after = _device_state()
        assert [id(m) for m in stack_after] == [id(m) for m in stack_before]
        assert torch.zeros(1).device.type == "cpu"
    finally:
        _restore_device_state(slot_before, stack_before)
