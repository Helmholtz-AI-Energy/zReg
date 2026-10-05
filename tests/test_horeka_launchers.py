"""Phase 63 HPC-03/HPC-04: HoreKa launcher and interactive Propulate test scripts.

The scripts are executed for real with stub srun/module/python on PATH; no
grep-only assertions. Each stub appends ``"<name> <argv>"`` to ``$STUB_LOG``
so the tests can assert on the exact command lines the scripts would run.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "baseline_experiments" / "scripts"

LAUNCHERS = [
    "launch_horeka_baseline.sbatch",
    "launch_horeka.sbatch",
    "launch_horeka_w20.sbatch",
]

_LOGGING_STUB = """#!/bin/bash
echo "{name} $*" >> "$STUB_LOG"
exit 0
"""

# srun stub for test_propulate_interactive.sh: logs argv; on --output-dir <dir>
# it fakes a Propulate run that wrote a checkpoint (and optionally best_params.json);
# when it runs the Test 2 helper script it prints the stale-skip warning.
_PROPULATE_SRUN_STUB = """#!/bin/bash
echo "srun $*" >> "$STUB_LOG"
out=""
helper=0
prev=""
for a in "$@"; do
    if [[ "$prev" == "--output-dir" ]]; then out="$a"; fi
    if [[ "$a" == *run_direct.py ]]; then helper=1; fi
    prev="$a"
done
if [[ -n "$out" ]]; then
    mkdir -p "$out/2026-01-01_00-00-00"
    touch "$out/2026-01-01_00-00-00/island_0_ckpt.pickle"
    if [[ "${STUB_WRITE_BEST:-0}" == "1" ]]; then
        echo '{"cpd_penalty": "rigid"}' > "$out/2026-01-01_00-00-00/best_params.json"
    fi
fi
if [[ "$helper" == "1" ]]; then
    echo "WARNING Skipping stale checkpoint individual"
fi
echo "fake srun"
exit 0
"""


def _write_exec(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stub_bin(tmp_path: Path, names: list[str]) -> Path:
    bin_dir = tmp_path / "bin"
    for name in names:
        _write_exec(bin_dir / name, _LOGGING_STUB.format(name=name))
    return bin_dir


def _clean_env() -> dict[str, str]:
    """os.environ without what makes bash bypass the PATH stubs.

    Inside a Slurm job on JSC systems, BASH_ENV sources Lmod and ``module``/``ml`` are
    exported as functions (BASH_FUNC_*); both win over the stub executables, so the
    launchers would run real (slow) ``module load`` calls.
    """
    return {k: v for k, v in os.environ.items() if k != "BASH_ENV" and not k.startswith("BASH_FUNC_")}


def _read_log(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


def _run_launcher(tmp_path: Path, script: str, clear: str | None) -> tuple[subprocess.CompletedProcess, list[str]]:
    bin_dir = _stub_bin(tmp_path, ["srun", "module", "python"])
    repo = tmp_path / "repo"
    repo.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("")
    _write_exec(venv / "bin" / "python", _LOGGING_STUB.format(name="python"))
    script_copy = tmp_path / script
    shutil.copy(SCRIPTS_DIR / script, script_copy)

    log = tmp_path / "stub.log"
    env = _clean_env()
    env.pop("ZREG_CLEAR_CHECKPOINTS", None)
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        STUB_LOG=str(log),
        ZREG_REPO_DIR=str(repo),
        ZREG_VENV=str(venv),
        HOME=str(tmp_path),
    )
    if clear is not None:
        env["ZREG_CLEAR_CHECKPOINTS"] = clear
    proc = subprocess.run(["bash", str(script_copy)], env=env, capture_output=True, timeout=60, cwd=tmp_path)
    return proc, _read_log(log)


def _diag(proc: subprocess.CompletedProcess, lines: list[str]) -> str:
    return (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}\nstub log:\n" + "\n".join(lines)
    )


def _srun_run_all_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("srun ") and "run_all.py" in ln]


@pytest.mark.parametrize("script", LAUNCHERS)
def test_launcher_default_does_not_clear_checkpoints(tmp_path: Path, script: str) -> None:
    proc, lines = _run_launcher(tmp_path, script, clear=None)
    srun_lines = _srun_run_all_lines(lines)
    assert srun_lines, "precondition: no srun run_all.py call logged\n" + _diag(proc, lines)
    assert proc.returncode == 0, _diag(proc, lines)
    offending = [ln for ln in srun_lines if "--clear-checkpoints" in ln.split()]
    assert not offending, "--clear-checkpoints passed by default (resubmission would restart HPO)\n" + _diag(proc, lines)


@pytest.mark.parametrize("script", LAUNCHERS)
def test_launcher_opt_in_clears_checkpoints(tmp_path: Path, script: str) -> None:
    proc, lines = _run_launcher(tmp_path, script, clear="1")
    srun_lines = _srun_run_all_lines(lines)
    assert srun_lines, "precondition: no srun run_all.py call logged\n" + _diag(proc, lines)
    assert proc.returncode == 0, _diag(proc, lines)
    missing = [ln for ln in srun_lines if "--clear-checkpoints" not in ln.split()]
    assert not missing, "ZREG_CLEAR_CHECKPOINTS=1 did not pass --clear-checkpoints\n" + _diag(proc, lines)


@pytest.mark.parametrize("script", LAUNCHERS)
@pytest.mark.parametrize("force", [None, "1"])
def test_launcher_force_is_opt_in(tmp_path: Path, script: str, force: str | None, monkeypatch) -> None:
    """63-REVIEW WR-04: ZREG_FORCE=1 passes --force to every run_all.py call; default does not."""
    if force is None:
        monkeypatch.delenv("ZREG_FORCE", raising=False)
    else:
        monkeypatch.setenv("ZREG_FORCE", force)
    proc, lines = _run_launcher(tmp_path, script, clear=None)
    assert proc.returncode == 0, _diag(proc, lines)
    run_all_lines = [ln for ln in lines if "run_all.py" in ln]
    assert run_all_lines, "precondition: no run_all.py call logged\n" + _diag(proc, lines)
    with_force = [ln for ln in run_all_lines if "--force" in ln.split()]
    expected = run_all_lines if force == "1" else []
    assert with_force == expected, _diag(proc, lines)


def test_baseline_launcher_still_runs_no_hpo_phase(tmp_path: Path) -> None:
    proc, lines = _run_launcher(tmp_path, "launch_horeka_baseline.sbatch", clear=None)
    assert proc.returncode == 0, _diag(proc, lines)
    py_lines = [ln for ln in lines if ln.startswith("python ") and "baseline_no_hpo" in ln]
    assert py_lines, "baseline_no_hpo python call not executed\n" + _diag(proc, lines)
    assert all("--clear-checkpoints" not in ln.split() for ln in py_lines)


def _run_propulate_interactive(tmp_path: Path, write_best: bool) -> tuple[subprocess.CompletedProcess, list[str]]:
    scripts = tmp_path / "baseline_experiments" / "scripts"
    scripts.mkdir(parents=True)
    script_copy = scripts / "test_propulate_interactive.sh"
    shutil.copy(SCRIPTS_DIR / "test_propulate_interactive.sh", script_copy)
    cfg = tmp_path / "baseline_experiments" / "configs_horeka" / "smoke" / "selfcal" / "kobitski_ew06_alignment.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("search_space:\n  cpd_penalty: [null, rigid]\ndefault_params:\n  cpd_penalty: null\n")

    bin_dir = _stub_bin(tmp_path, ["module", "python"])
    _write_exec(bin_dir / "srun", _PROPULATE_SRUN_STUB)
    _write_exec(tmp_path / "regvenv_horeka" / "bin" / "python", _LOGGING_STUB.format(name="python"))

    log = tmp_path / "stub.log"
    env = _clean_env()
    env.pop("ZREG_VENV", None)
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        STUB_LOG=str(log),
        HOME=str(tmp_path),
        STUB_WRITE_BEST="1" if write_best else "0",
    )
    proc = subprocess.run(["bash", str(script_copy), "2"], env=env, capture_output=True, timeout=60, cwd=tmp_path)
    return proc, _read_log(log)


def test_propulate_interactive_reaches_summary_without_best_params(tmp_path: Path) -> None:
    proc, lines = _run_propulate_interactive(tmp_path, write_best=False)
    stdout = proc.stdout.decode(errors="replace")
    stderr = proc.stderr.decode(errors="replace")
    assert "=== Summary" in stdout, "script died before summary\n" + _diag(proc, lines)
    assert proc.returncode != 0, "missing best_params.json must count as failure\n" + _diag(proc, lines)
    assert "No such file" not in stderr, _diag(proc, lines)
    assert "island_0_ckpt.pickle" in stdout, "checkpoint listing from fallback dir missing\n" + _diag(proc, lines)


def test_propulate_interactive_full_pass_with_stale_skip(tmp_path: Path) -> None:
    proc, lines = _run_propulate_interactive(tmp_path, write_best=True)
    stdout = proc.stdout.decode(errors="replace")
    assert "=== Summary" in stdout, "script died before summary\n" + _diag(proc, lines)
    assert any("run_direct.py" in ln for ln in lines if ln.startswith("srun ")), _diag(proc, lines)
    assert "0 failed" in stdout and proc.returncode == 0, _diag(proc, lines)
