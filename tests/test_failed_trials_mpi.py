"""Real ``mpirun -n 2`` integration tests for HPO failure bookkeeping (Phase 59 NUM-05).

Each test launches ``tests/_failed_trials_mpi_worker.py`` on two MPI ranks
with a per-rank search space in which ``k_neighbours=10000`` makes a trial
raise for real.  The worker records per rank whether ``run()`` raised; rank 0
writes ``best_params.json`` / ``search_history.json`` / ``failed_trials.json``
into the shared ``<tmp>/hpo`` directory.

Covers review cycle-1 HIGH 2 (failures are MPI-global; the run raises only
when no rank succeeded) and cycle-2 HIGH (rank 0 persists the successful
trials of other ranks even when all of its own trials failed).

Skips when ``mpirun`` or ``mpi4py`` is unavailable; the Propulate variant
additionally skips when ``propulate`` cannot be imported.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _mpi_launch import mpi_launcher

_WORKER = Path(__file__).parent / "_failed_trials_mpi_worker.py"
_TIMEOUT_S = 600


def _run_worker(tmp_path: Path, scenario: str, strategy: str) -> tuple[dict, dict]:
    launcher = mpi_launcher(2)
    assert _WORKER.exists(), f"MPI worker not found at {_WORKER}"
    proc = subprocess.run(
        [*launcher, sys.executable, str(_WORKER), scenario, strategy, str(tmp_path)],
        capture_output=True,
        timeout=_TIMEOUT_S,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, (
        f"{launcher[0]} failed: stdout={proc.stdout.decode(errors='replace')!r} "
        f"stderr={proc.stderr.decode(errors='replace')!r}"
    )
    ranks = {}
    for r in (0, 1):
        path = tmp_path / f"rank_{r}.json"
        assert path.exists(), f"rank {r} wrote no report; stderr={proc.stderr.decode(errors='replace')!r}"
        ranks[r] = json.loads(path.read_text())
    return ranks[0], ranks[1]


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _assert_rank0_persisted_rank1_success(tmp_path: Path) -> None:
    hpo = tmp_path / "hpo"
    assert _load(hpo / "best_params.json")["k_neighbours"] == 3
    history = _load(hpo / "search_history.json")
    assert history, "rank 0 must persist rank 1's successful trial"
    assert all(math.isfinite(t["score"]) for t in history)
    failed = _load(hpo / "failed_trials.json")
    assert any(r["rank"] == 0 for r in failed)


def test_mpirun_rank1_fails_globally_succeeds(tmp_path) -> None:
    """Rank 1 fails everything, rank 0 succeeds: no rank raises; rank 1's failure is gathered."""
    r0, r1 = _run_worker(tmp_path, "rank1_fails", "grid")
    assert r0["raised"] is False, r0
    assert r1["raised"] is False, r1
    failed = _load(tmp_path / "hpo" / "failed_trials.json")
    assert any(r["rank"] == 1 for r in failed)
    assert _load(tmp_path / "hpo" / "best_params.json")["k_neighbours"] == 3


def test_mpirun_rank0_fails_rank1_succeeds(tmp_path) -> None:
    """Cycle-2 HIGH: rank 0 fails everything, rank 1 succeeds -> rank 0 writes rank 1's best."""
    r0, r1 = _run_worker(tmp_path, "rank0_fails", "grid")
    assert r0["raised"] is False, r0
    assert r1["raised"] is False, r1
    assert r0["best_params"]["k_neighbours"] == 3
    _assert_rank0_persisted_rank1_success(tmp_path)


def test_mpirun_all_fail_raises_on_every_rank(tmp_path) -> None:
    """No rank succeeds: every rank raises and rank 0 lists both ranks' failures."""
    r0, r1 = _run_worker(tmp_path, "all_fail", "grid")
    assert r0["raised"] is True and "All" in r0["message"], r0
    assert r1["raised"] is True and "All" in r1["message"], r1
    failed = _load(tmp_path / "hpo" / "failed_trials.json")
    assert {r["rank"] for r in failed} == {0, 1}


def test_mpirun_propulate_rank0_fails_rank1_succeeds(tmp_path) -> None:
    """Optional real-Propulate variant of the cycle-2 HIGH inverse case."""
    mpi_launcher(2)  # skip early when no MPI launcher is available
    pytest.importorskip("propulate")
    r0, r1 = _run_worker(tmp_path, "rank0_fails", "propulate")
    assert r0["raised"] is False, r0
    assert r1["raised"] is False, r1
    assert r0["best_params"]["k_neighbours"] == 3
    _assert_rank0_persisted_rank1_success(tmp_path)
