"""Real ``mpirun -n 2`` checks that persisted HPO history carries real payloads (Phase 62, 62-06).

Each test launches ``tests/_propulate_payload_mpi_worker.py`` on two MPI ranks.
The worker runs ``HyperparamOptimizer`` with cpd_weighted label transfer and
one dependency substitution (a real ``AlignmentStage`` subclass that zeroes the
first receiver column of every posterior), so every successful trial carries
deterministic, non-empty label-transfer fallback flags.

Checked on rank 0's ``search_history.json`` (59-REVIEW IN-09b, 62-REVIEWS
cycle 3 MEDIUM): every entry carries the real fallback flags (one per frame,
no ``PROPULATE_PLACEHOLDER_FLAG``) and real, finite, non-zero metrics -- not
the zero-filled placeholder metrics the Propulate branch used to rebuild.

The grid variant runs wherever ``mpirun`` and ``mpi4py`` exist and validates the
worker itself. The Propulate variant is the production-backend check; it also
needs ``propulate`` and is listed with the HoreKa / real-Propulate checks.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).parent / "_propulate_payload_mpi_worker.py"
_TIMEOUT_S = 600
_PLACEHOLDER_FLAG_PREFIX = "propulate: individual returned without an evaluation record"


def _run_worker(tmp_path: Path, strategy: str) -> list[dict]:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not found on PATH")
    pytest.importorskip("mpi4py")
    proc = subprocess.run(
        [mpirun, "-n", "2", sys.executable, str(_WORKER), strategy, str(tmp_path)],
        capture_output=True,
        timeout=_TIMEOUT_S,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, (
        f"mpirun failed: stdout={proc.stdout.decode(errors='replace')!r} "
        f"stderr={proc.stderr.decode(errors='replace')!r}"
    )
    for r in (0, 1):
        report = json.loads((tmp_path / f"rank_{r}.json").read_text())
        assert report["raised"] is False, report
    with open(tmp_path / "hpo" / "search_history.json") as f:
        return json.load(f)


def _assert_real_flagged_payloads(history: list[dict]) -> None:
    assert history, "rank 0 persisted no trial"
    for entry in history:
        assert math.isfinite(entry["score"])
        flags = entry["flags"]
        # one fallback flag per frame from the real LabelTransferStage
        assert flags, f"empty flags in persisted trial {entry}"
        assert all(f.startswith("cpd_weighted: 1 of ") and " in frame " in f for f in flags), flags
        assert not any(f.startswith(_PLACEHOLDER_FLAG_PREFIX) for f in flags), flags
        metrics = entry["metrics"]
        for key in ("chamfer_distance", "hausdorff_distance"):
            assert math.isfinite(metrics[key]) and metrics[key] > 0.0, (key, metrics)


def test_mpirun_grid_history_carries_real_flags_and_metrics(tmp_path) -> None:
    """Worker self-check on the grid strategy (runs wherever mpirun + mpi4py exist)."""
    _assert_real_flagged_payloads(_run_worker(tmp_path, "grid"))


def test_mpirun_propulate_history_carries_real_flags_and_metrics(tmp_path) -> None:
    """Production backend: real Propulate persists real flags and metrics on rank 0."""
    pytest.importorskip("propulate")
    _assert_real_flagged_payloads(_run_worker(tmp_path, "propulate"))
