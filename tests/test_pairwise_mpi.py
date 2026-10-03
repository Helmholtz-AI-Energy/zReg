"""MPI collective-safety tests for the pairwise cost-matrix sweep (Phase 61 DIST-01).

Both ``create_pairwise_distance_matrix`` and
``create_pairwise_distance_matrix_given_rigid_rot`` distribute the (i, j) pairs
round-robin over the ranks and combine every row with one ``allgather``.
Collectives match by call order, so every rank must reach the allgather for
every row -- including rows in which it computed no pair.  Before the fix a rank
without local pairs in a row skipped the allgather, which made a 2-rank
x=2/y=1 sweep finish with a silently wrong matrix and an x=3/y=1 sweep hang.

Two layers of evidence:

* ``ThreadComm`` simulates N ranks with threads and a barrier-based
  ``allgather`` (payloads are copied on store and on return, mirroring the
  pickling of real MPI), so lock-step violations surface as a broken barrier or
  a timeout instead of a hang of the test session.
* ``test_mpirun_two_ranks`` runs ``tests/_pairwise_mpi_worker.py`` under a real
  ``mpirun -n 2`` with a subprocess timeout (skipped without mpirun/mpi4py).
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path
from unittest import mock

from zreg.algorithms import pairwise_distance_matrix as pm
from zreg.core.dataset import zRegPointCloud

import numpy as np
import pytest
import torch

_WORKER = Path(__file__).parent / "_pairwise_mpi_worker.py"
_MPIRUN_TIMEOUT_S = 120


class ThreadComm:
    """Minimal thread-backed stand-in for ``MPI.COMM_WORLD``.

    Each thread sets its own rank via ``_local.rank``.  ``allgather`` stores a
    copy of the payload in the caller's slot, waits for all ranks, builds the
    result from fresh copies, and waits again before returning so no rank can
    overwrite a slot while another one is still reading it.
    """

    def __init__(self, size: int, timeout: float = 10.0):
        self.size = size
        self._local = threading.local()
        self._barrier = threading.Barrier(size, timeout=timeout)
        self._slots = [None] * size

    @property
    def rank(self) -> int:
        return self._local.rank

    def allgather(self, obj):
        # The production row is a NumPy view of the tensor; copy so the later
        # tensor writeback can never mutate a stored payload (real MPI pickles).
        self._slots[self.rank] = np.array(obj, copy=True)
        self._barrier.wait()
        out = [np.array(s, copy=True) for s in self._slots]
        self._barrier.wait()
        return out


def run_ranks(fn, size: int, timeout: float = 10.0):
    """Run ``fn`` on ``size`` simulated ranks; return (results, errors, hung)."""
    comm = ThreadComm(size, timeout)
    results, errors = [None] * size, [None] * size

    def worker(r):
        comm._local.rank = r
        try:
            results[r] = fn()
        except BaseException as e:  # noqa: BLE001 - surfaced through `errors`
            errors[r] = e
            comm._barrier.abort()

    with mock.patch.object(pm, "hasmpi", True), mock.patch.object(
        pm, "MPI", types.SimpleNamespace(COMM_WORLD=comm)
    ):
        threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout * 3)
        hung = [t.is_alive() for t in threads]
        if any(hung):
            comm._barrier.abort()
    return results, errors, hung


def _frames(n: int) -> dict[int, zRegPointCloud]:
    return {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(n)}


def _call(which: str, x, y, window, mpi_distribute: bool) -> torch.Tensor:
    if which == "create":
        return pm.create_pairwise_distance_matrix(
            x, y, window=window, normalize=False, distance_metric="euclidean",
            mpi_distribute=mpi_distribute,
        ).cost_matrix
    return pm.create_pairwise_distance_matrix_given_rigid_rot(
        x, y, rotation=torch.eye(3), translation=torch.zeros(3), scale=1.0,
        window=window, normalize=False, distance_metric="euclidean",
        mpi_distribute=mpi_distribute,
    )


@pytest.mark.parametrize("which", ["create", "given_rigid_rot"])
@pytest.mark.parametrize(
    "nx, ny, window, size",
    [(2, 1, None, 2), (3, 1, None, 2), (4, 4, 0, 3)],
    ids=["x2y1-2ranks", "x3y1-2ranks", "x4y4w0-3ranks"],
)
def test_thread_ranks_equal_serial(which, nx, ny, window, size):
    """Every simulated rank finishes, raises nothing and holds the serial matrix (DIST-01)."""
    torch.manual_seed(0)
    x, y = _frames(nx), _frames(ny)
    serial = _call(which, x, y, window, mpi_distribute=False)

    results, errors, hung = run_ranks(lambda: _call(which, x, y, window, mpi_distribute=True), size)

    assert not any(hung), f"ranks hung (collective mismatch): {hung}"
    assert not any(errors), f"rank errors: {errors}"
    for r, res in enumerate(results):
        assert torch.equal(res, serial), f"rank {r} matrix {res} != serial {serial}"


def _require_mpi() -> str:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not found on PATH")
    pytest.importorskip("mpi4py")
    return mpirun


@pytest.mark.parametrize("which", ["create", "given_rigid_rot"])
def test_mpirun_two_ranks(which, tmp_path):
    """Real ``mpirun -n 2`` with x=3, y=1: no deadlock, both ranks equal the serial matrix."""
    mpirun = _require_mpi()
    assert _WORKER.exists(), f"MPI worker not found at {_WORKER}"
    try:
        proc = subprocess.run(
            [mpirun, "-n", "2", sys.executable, str(_WORKER), which, "3", str(tmp_path)],
            capture_output=True,
            timeout=_MPIRUN_TIMEOUT_S,
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"pairwise MPI sweep deadlocked (DIST-01): mpirun -n 2 ({which}, x=3, y=1) "
            f"did not finish within {_MPIRUN_TIMEOUT_S} s"
        )
    assert proc.returncode == 0, (
        f"mpirun failed: stdout={proc.stdout.decode(errors='replace')!r} "
        f"stderr={proc.stderr.decode(errors='replace')!r}"
    )
    for r in (0, 1):
        path = tmp_path / f"rank_{r}.json"
        assert path.exists(), f"rank {r} wrote no report; stderr={proc.stderr.decode(errors='replace')!r}"
        report = json.loads(path.read_text())
        assert report["rank"] == r
        assert report["equal"] is True, f"rank {r} matrix differs from serial: {report}"
