"""Execution tests for the SLURM orchestrators and sbatch templates (HPC-04).

The orchestrators ``scripts/run_stage{1,2}_{horeka,juwels}.sh`` are run for
real against a stub ``sbatch`` (and a stub ``mkdir``) placed first on PATH.
The stubs record their argv in ``$STUB_LOG`` so the tests can check:

* eval submissions request a single rank / single GPU (U7-8),
* HPO submissions keep the template's multi-rank default,
* every submission passes ``--output=<dir>/%x-%j.out`` on the command line and
  that directory was created before ``sbatch`` ran (U7-9),
* the stage-1 dependency chain is unchanged.

The scripts are copied into ``tmp_path/scripts`` so ``REPO_ROOT`` resolves to
``tmp_path`` and nothing is written into the repository.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

# orchestrator -> (sbatch template, cluster default SLURM log dir for USER=testuser)
ORCHESTRATORS = {
    "run_stage1_horeka.sh": ("exp_horeka.sbatch", "/hkfs/work/workspace/scratch/testuser-zreg/logs/slurm"),
    "run_stage1_juwels.sh": ("exp_juwels.sbatch", "/p/project1/tissuetwin/testuser/logs/slurm"),
    "run_stage2_horeka.sh": ("exp_horeka.sbatch", "/hkfs/work/workspace/scratch/testuser-zreg/logs/slurm"),
    "run_stage2_juwels.sh": ("exp_juwels.sbatch", "/p/project1/tissuetwin/testuser/logs/slurm"),
}

SBATCH_STUB = r"""#!/bin/bash
# Stub sbatch: record argv, require that the --output directory exists.
{ printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >> "$STUB_LOG"
for a in "$@"; do
    case "$a" in
        --output=*)
            d="$(dirname "${a#--output=}")"
            case "$d" in
                "$STUB_ROOT"/*)
                    if [ ! -d "$d" ]; then
                        echo "stub sbatch: --output directory $d does not exist" >&2
                        exit 3
                    fi
                    ;;
                *)
                    if ! grep -qxF "mkdir -p $(printf '%q' "$d")" "$STUB_LOG"; then
                        echo "stub sbatch: --output directory $d was not created (no mkdir -p before sbatch)" >&2
                        exit 3
                    fi
                    ;;
            esac
            ;;
    esac
done
echo 4242
"""

MKDIR_STUB = r"""#!/bin/bash
# Stub mkdir: record argv; only really create directories under STUB_ROOT.
{ printf 'mkdir'; printf ' %q' "$@"; printf '\n'; } >> "$STUB_LOG"
for a in "$@"; do
    case "$a" in
        -*) ;;
        "$STUB_ROOT"/*) ;;
        *) exit 0 ;;
    esac
done
exec __REAL_MKDIR__ "$@"
"""


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _run_orchestrator(tmp_path: Path, script: str, log_dir_override: str | None):
    sbatch_tpl, _ = ORCHESTRATORS[script]
    sdir = tmp_path / "scripts"
    sdir.mkdir()
    shutil.copy2(SCRIPTS / script, sdir / script)
    shutil.copy2(SCRIPTS / sbatch_tpl, sdir / sbatch_tpl)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    real_mkdir = shutil.which("mkdir") or "/bin/mkdir"
    _write_exe(bindir / "sbatch", SBATCH_STUB)
    _write_exe(bindir / "mkdir", MKDIR_STUB.replace("__REAL_MKDIR__", real_mkdir))

    stub_log = tmp_path / "stub.log"
    stub_log.touch()
    env = dict(os.environ)
    env.pop("ZREG_SLURM_LOG_DIR", None)
    env.update(
        PATH=f"{bindir}{os.pathsep}{env.get('PATH', '')}",
        STUB_LOG=str(stub_log),
        STUB_ROOT=str(tmp_path),
        USER="testuser",
    )
    if log_dir_override is not None:
        env["ZREG_SLURM_LOG_DIR"] = log_dir_override

    proc = subprocess.run(
        ["bash", str(sdir / script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [shlex.split(ln) for ln in stub_log.read_text().splitlines() if ln.strip()]
    return proc, lines


def _sbatch_calls(lines):
    return [ln[1:] for ln in lines if ln and ln[0] == "sbatch"]


def _export(argv):
    for a in argv:
        if a.startswith("--export="):
            return a[len("--export="):]
    return ""


def _outputs(argv):
    return [a[len("--output="):] for a in argv if a.startswith("--output=")]


def _assert_rc0(proc):
    assert proc.returncode == 0, (
        f"orchestrator exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


@pytest.mark.parametrize("script", sorted(ORCHESTRATORS))
def test_stage_eval_single_rank_and_log_dir_override(tmp_path, script):
    log_dir = tmp_path / "slurm_logs"
    proc, lines = _run_orchestrator(tmp_path, script, str(log_dir))
    _assert_rc0(proc)

    calls = _sbatch_calls(lines)
    assert calls, "no sbatch submissions recorded"
    evals = [c for c in calls if "MODE=eval" in _export(c)]
    hpo = [c for c in calls if "MODE=eval" not in _export(c)]
    assert evals, "orchestrator made no eval submissions"
    assert hpo, "orchestrator made no HPO submissions"

    for c in evals:
        assert "--ntasks-per-node=1" in c, f"eval submission not single-rank: {c}"
        assert "--gres=gpu:1" in c, f"eval submission not single-GPU: {c}"
    for c in hpo:
        assert "--ntasks-per-node=1" not in c, f"HPO submission was reduced to one rank: {c}"
        assert "--gres=gpu:1" not in c, f"HPO submission was reduced to one GPU: {c}"

    for c in calls:
        outs = _outputs(c)
        assert len(outs) == 1, f"expected exactly one --output= on the command line: {c}"
        assert outs[0] == f"{log_dir}/%x-%j.out", outs[0]
    assert log_dir.is_dir()

    if script.startswith("run_stage1"):
        chained = [c for c in calls if any(a.startswith("--dependency=") for a in c)]
        # 3 methods x (semisynthetic + real) are chained on their predecessor
        assert len(chained) == 6, chained
        for c in chained:
            assert "--dependency=afterok:4242" in c, c
            assert "WARM_START_FROM=" in _export(c), c


@pytest.mark.parametrize("script", sorted(ORCHESTRATORS))
def test_stage_default_slurm_log_dir_created_before_sbatch(tmp_path, script):
    _, default_dir = ORCHESTRATORS[script]
    proc, lines = _run_orchestrator(tmp_path, script, None)
    _assert_rc0(proc)

    calls = _sbatch_calls(lines)
    assert calls, "no sbatch submissions recorded"
    for c in calls:
        assert _outputs(c) == [f"{default_dir}/%x-%j.out"], c

    first_sbatch = next(i for i, ln in enumerate(lines) if ln[0] == "sbatch")
    mkdir_idx = [
        i for i, ln in enumerate(lines)
        if ln[0] == "mkdir" and "-p" in ln and default_dir in ln[1:]
    ]
    assert mkdir_idx, f"no 'mkdir -p {default_dir}' recorded"
    assert mkdir_idx[0] < first_sbatch, "SLURM log dir created only after the first sbatch"
