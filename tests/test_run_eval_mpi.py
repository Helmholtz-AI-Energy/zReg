"""Real ``mpirun -n 2`` regression tests for ``run_eval.main`` (Phase 63 HPC-04).

Real run_eval.main under real mpirun -n 2; only EvaluationRunner.run is
wrapped to count calls.  The worker (``tests/_run_eval_mpi_worker.py``) delays
rank 1 by 1.2 s so per-process timestamps would differ.

Covers U7-7 (every rank picked its own ``datetime.now()`` stamp and wrote its
own output directory) and the in-process half of U7-8 (every rank ran
``EvaluationRunner`` in ``eval``/``full`` mode), plus the cycle-2 review item
that a rank-0 failure while creating the run directory must make every rank
raise instead of leaving non-root ranks waiting forever.

Skips when ``mpirun`` or ``mpi4py`` is unavailable.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).parent / "_run_eval_mpi_worker.py"
_TIMEOUT_S = 600


def _require_mpi() -> str:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not found on PATH")
    pytest.importorskip("mpi4py")
    return mpirun


def _run_worker(tmp_path: Path, scenario: str, timeout: float = _TIMEOUT_S) -> dict[int, dict]:
    mpirun = _require_mpi()
    assert _WORKER.exists(), f"MPI worker not found at {_WORKER}"
    proc = subprocess.run(
        [mpirun, "-n", "2", sys.executable, str(_WORKER), scenario, str(tmp_path)],
        capture_output=True,
        timeout=timeout,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, (
        f"mpirun failed: stdout={proc.stdout.decode(errors='replace')!r} "
        f"stderr={proc.stderr.decode(errors='replace')!r}"
    )
    ranks = {}
    for r in (0, 1):
        path = tmp_path / f"rank_{r}.json"
        assert path.exists(), f"rank {r} wrote no report; stderr={proc.stderr.decode(errors='replace')!r}"
        ranks[r] = json.loads(path.read_text())
    return ranks


def _assert_ranks_succeeded(ranks: dict[int, dict]) -> None:
    for r, rep in ranks.items():
        assert rep["error"] is None and rep["rc"] == 0, f"rank {r} failed: {rep}"


def _run_dirs(tmp_path: Path) -> list[Path]:
    return sorted(p for p in (tmp_path / "runs").iterdir() if p.is_dir())


def test_ranks_share_one_output_dir_optimize(tmp_path) -> None:
    """--mode optimize under 2 ranks writes exactly one timestamped directory."""
    ranks = _run_worker(tmp_path, "optimize")
    _assert_ranks_succeeded(ranks)
    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1, f"expected one shared run directory, got {[d.name for d in dirs]}"
    assert (dirs[0] / "run_config.yaml").exists()
    assert (dirs[0] / "best_params.json").exists()


def test_full_mode_evaluates_on_rank0_only(tmp_path) -> None:
    """--mode full: collective HPO on both ranks, EvaluationRunner on rank 0 only."""
    ranks = _run_worker(tmp_path, "full")
    _assert_ranks_succeeded(ranks)
    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1, f"expected one shared run directory, got {[d.name for d in dirs]}"
    counts = {r: rep["eval_runs"] for r, rep in ranks.items()}
    assert counts == {0: 1, 1: 0}, f"EvaluationRunner.run calls per rank: {counts}"
    reports = list((tmp_path / "runs").rglob("eval_report.json"))
    assert len(reports) == 1, reports


def test_eval_mode_evaluates_on_rank0_only(tmp_path) -> None:
    """--mode eval: only rank 0 evaluates; rank 1 returns 0 immediately."""
    ranks = _run_worker(tmp_path, "eval")
    _assert_ranks_succeeded(ranks)
    dirs = _run_dirs(tmp_path)
    assert len(dirs) == 1, f"expected one shared run directory, got {[d.name for d in dirs]}"
    counts = {r: rep["eval_runs"] for r, rep in ranks.items()}
    assert counts == {0: 1, 1: 0}, f"EvaluationRunner.run calls per rank: {counts}"


def test_rank0_setup_failure_makes_every_rank_raise(tmp_path) -> None:
    """Rank 0 cannot create the run directory: both ranks raise, nobody hangs.

    The worker makes ``<tmp>/runs`` read-only.  Rank 0 broadcasts its setup
    error, so rank 1 raises a RuntimeError naming it instead of blocking on a
    collective that rank 0 never reaches (Phase 63 cycle-2 review).
    """
    try:
        ranks = _run_worker(tmp_path, "rank0_setup_fails", timeout=180)
    finally:
        runs = tmp_path / "runs"
        if runs.exists():
            os.chmod(runs, stat.S_IRWXU)  # let pytest clean tmp_path
    assert ranks[0]["error"] is not None, ranks[0]
    assert "PermissionError" in ranks[0]["error"], ranks[0]
    assert ranks[1]["error"] is not None, ranks[1]
    assert ranks[1]["error"].startswith("RuntimeError"), ranks[1]
    assert "rank 0" in ranks[1]["error"] and "PermissionError" in ranks[1]["error"], ranks[1]
    assert ranks[1]["eval_runs"] == 0
    assert _run_dirs(tmp_path) == []
