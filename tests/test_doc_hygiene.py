"""Documentation hygiene checks (Phase 63, DOC-01).

These tests keep docstrings, comments and packaging metadata in line with the
code after the package restructure.  Per D-06 grep-style checks over source
text are allowed; in addition the ``zreg.evaluation`` doctests are executed
for real (the D-09 equivalent of ``tox -e doctests``, which needs sphinx/tox
that are not installed in every environment).

Checks:

- ``test_zreg_evaluation_doctests_pass``: every doctest in every
  ``zreg.evaluation`` submodule runs and passes.  Submodule import failures are
  collected together with doctest failures so a broken import surfaces as an
  assertion naming the module, not as a collection error.
- ``test_pyproject_coverage_omit_paths_exist``: every
  ``[tool.coverage.run].omit`` entry names an existing file.
- ``test_eval_runner_frame_convention_docstrings``: the ``eval_runner``
  module and ``EvaluationRunner._run_single`` docstrings describe the frame
  convention the code uses (F1 ground truth from the source's LAST frame), with
  no stale "first frame" claim.
- ``test_path_smoothness_header_matches_implementation``: ``path_smoothness``
  is described as implemented (cross-product based), not by slope changes.

Out of scope by definition (not per-file exclusions):

- ``tests/``: ``tests/test_script_imports.py`` and this module contain the
  removed module names as literal patterns.
- non-``.py`` files (``*.md`` etc.): DOC-01 is about docstrings and comments
  in Python sources.
- ``.planning/``: historical planning records describe past states on purpose.

Repo modules are imported inside the test functions so that a missing or
broken module produces an assertion-level failure, never a collection error.
"""

from __future__ import annotations

import doctest
import importlib
import inspect
import pkgutil
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_zreg_evaluation_doctests_pass() -> None:
    """All doctests in ``zreg.evaluation`` import live modules and pass."""
    problems: list[str] = []
    attempted = 0

    try:
        package = importlib.import_module("zreg.evaluation")
    except Exception as exc:  # pragma: no cover - reported via the assertion
        problems.append(f"zreg.evaluation: import failed: {exc!r}")
        package = None

    if package is not None:
        module_names = ["zreg.evaluation"]
        module_names += [
            info.name
            for info in pkgutil.walk_packages(
                package.__path__,
                "zreg.evaluation.",
                onerror=lambda name: problems.append(f"{name}: walk_packages import failed"),
            )
        ]

        finder = doctest.DocTestFinder()
        for name in module_names:
            try:
                module = importlib.import_module(name)
            except Exception as exc:
                problems.append(f"{name}: import failed: {exc!r}")
                continue
            for test in finder.find(module, name):
                if not test.examples:
                    continue
                failures: list[str] = []

                def _out(text: str, _sink: list[str] = failures) -> None:
                    _sink.append(text)

                runner = doctest.DocTestRunner(verbose=False)
                result = runner.run(test, out=_out, clear_globs=True)
                attempted += result.attempted
                if result.failed:
                    problems.append(f"{test.name}: {result.failed} failed\n" + "".join(failures))

    assert not problems, "zreg.evaluation doctest problems:\n" + "\n".join(problems)
    assert attempted > 0, "no doctest examples were attempted in zreg.evaluation"


def test_pyproject_coverage_omit_paths_exist() -> None:
    """Every ``[tool.coverage.run].omit`` path exists relative to the repo root."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    omit = config.get("tool", {}).get("coverage", {}).get("run", {}).get("omit", [])
    missing = [entry for entry in omit if not (REPO_ROOT / entry).exists()]
    assert not missing, f"pyproject.toml coverage omit paths do not exist: {missing}"


def test_eval_runner_frame_convention_docstrings() -> None:
    """eval_runner docstrings describe the last-frame F1 convention the code uses."""
    runner_module = importlib.import_module("eval.runners.eval_runner")
    module_doc = inspect.getdoc(runner_module) or ""
    run_single_doc = inspect.getdoc(runner_module.EvaluationRunner._run_single) or ""

    stale = [
        (label, phrase)
        for label, doc in (("module", module_doc), ("_run_single", run_single_doc))
        for phrase in ("first frame", "first-frame")
        if phrase in doc.lower()
    ]
    assert not stale, f"eval_runner docstrings still claim a first-frame convention: {stale}"

    combined = (module_doc + "\n" + run_single_doc).lower()
    assert "last frame" in combined, (
        "eval_runner docstrings must state that F1 ground truth comes from the "
        "source's last frame"
    )


def test_path_smoothness_header_matches_implementation() -> None:
    """path_smoothness is described as cross-product based, not slope based."""
    alignment = importlib.import_module("zreg.evaluation.alignment")
    eval_types = importlib.import_module("eval.types")

    header = inspect.getdoc(alignment) or ""
    stage_doc = inspect.getdoc(eval_types.StageMetrics) or ""

    offenders = [
        label
        for label, doc in (("zreg.evaluation.alignment", header), ("eval.types.StageMetrics", stage_doc))
        if "slope" in doc.lower()
    ]
    assert not offenders, f"path_smoothness still described via slope changes in: {offenders}"
    assert "cross-product" in header.lower() or "cross product" in header.lower(), (
        "zreg.evaluation.alignment header must describe path_smoothness as cross-product based"
    )
