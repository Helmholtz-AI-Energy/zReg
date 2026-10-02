"""Regression tests for RUN-02: the example / HPC scripts import and resolve.

The package restructure moved ``zreg.dataset``, ``zreg.cpd``,
``zreg.downsampling``, ``zreg.transforms`` and
``zreg.pairwise_distance_matrix`` under ``zreg.core`` / ``zreg.algorithms`` /
``zreg.preprocessing``.  These tests guard against scripts silently referring
to module paths that no longer exist, and against import-time side effects
(data loads, directory creation, SLURM environment reads).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# zreg must be imported before torch (libomp ordering on macOS ARM).
import zreg

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

SCRIPTS = [
    "generate_real_previews.py",
    "label_transfer_example.py",
    "example_plots.py",
    "dtw_testing.py",
]


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>`` as a fresh module object (not registered in sys.modules)."""
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(f"_zreg_script_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# zreg.dtw backward-compatibility shim
# ---------------------------------------------------------------------------


def test_zreg_dtw_from_import():
    from zreg.dtw import DynamicTimeWarping  # noqa: F401

    assert DynamicTimeWarping is zreg.algorithms.dtw.DynamicTimeWarping


def test_zreg_dtw_is_algorithms_dtw():
    assert importlib.import_module("zreg.dtw") is zreg.algorithms.dtw
    assert zreg.dtw is zreg.algorithms.dtw


# ---------------------------------------------------------------------------
# Script importability (no side effects)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPTS)
def test_script_imports_cleanly(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    previews_dir = REPO_ROOT / "reports" / "dataset_previews" / "real"
    existed_before = previews_dir.exists()

    module = _load_script(script)

    assert callable(getattr(module, "main", None)), f"{script} must define main()"
    assert list(tmp_path.iterdir()) == []
    assert previews_dir.exists() == existed_before


# ---------------------------------------------------------------------------
# AST check: every zreg.* name a script references resolves
# ---------------------------------------------------------------------------


def _attribute_chain(node: ast.Attribute) -> str | None:
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == "zreg":
        parts.append("zreg")
        return ".".join(reversed(parts))
    return None


def _zreg_references(source: str) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain is not None:
                refs.add(chain)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "zreg" or alias.name.startswith("zreg."):
                    refs.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and (mod == "zreg" or mod.startswith("zreg.")):
                refs.add(mod)
                for alias in node.names:
                    refs.add(f"{mod}.{alias.name}")
    # Keep only maximal chains: ``zreg.cpd`` is implied by ``zreg.cpd.RigidCPD``.
    return {r for r in refs if not any(o.startswith(r + ".") for o in refs)}


def _resolves(dotted: str) -> bool:
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


def test_resolver_rejects_stale_names():
    """Sanity check of the resolver itself: old pre-restructure paths fail."""
    assert _resolves("zreg.core.dataset.zRegPointCloud")
    assert not _resolves("zreg.dataset.zRegPointCloud")
    assert not _resolves("zreg.cpd.registration_cpd")


@pytest.mark.parametrize("script", SCRIPTS, ids=SCRIPTS)
def test_zreg_names_resolve(script):
    source = (SCRIPTS_DIR / script).read_text()
    refs = _zreg_references(source)
    unresolved = sorted(r for r in refs if not _resolves(r))
    assert unresolved == []
