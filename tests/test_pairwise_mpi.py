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

import copy
import json
import os
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
from _mpi_launch import mpi_launcher
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
        # The production row is a NumPy view of the tensor; deep-copy so the later
        # tensor writeback can never mutate a stored payload (real MPI pickles).
        self._slots[self.rank] = copy.deepcopy(obj)
        self._barrier.wait()
        out = [copy.deepcopy(s) for s in self._slots]
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
            # No barrier abort here: a peer released from the last allgather barrier
            # may not have re-checked the barrier state yet, and an abort would turn
            # its expected exception into BrokenBarrierError (flaky CR-01).  A genuine
            # lock-step violation still surfaces via the Barrier timeout, and hung
            # ranks via the join-timeout abort below.

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


@pytest.mark.parametrize("which", ["create", "given_rigid_rot"])
def test_mpirun_two_ranks(which, tmp_path):
    """Real ``mpirun -n 2`` with x=3, y=1: no deadlock, both ranks equal the serial matrix."""
    launcher = mpi_launcher(2)
    assert _WORKER.exists(), f"MPI worker not found at {_WORKER}"
    try:
        proc = subprocess.run(
            [*launcher, sys.executable, str(_WORKER), which, "3", str(tmp_path)],
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
        f"{launcher[0]} failed: stdout={proc.stdout.decode(errors='replace')!r} "
        f"stderr={proc.stderr.decode(errors='replace')!r}"
    )
    for r in (0, 1):
        path = tmp_path / f"rank_{r}.json"
        assert path.exists(), f"rank {r} wrote no report; stderr={proc.stderr.decode(errors='replace')!r}"
        report = json.loads(path.read_text())
        assert report["rank"] == r
        assert report["equal"] is True, f"rank {r} matrix differs from serial: {report}"
        if which == "create":
            assert report["equal_transforms"] is True, f"rank {r} transforms differ from serial (WR-03)"


_FAIL_MARKER = 999.0


def _failing_metric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean-like metric that raises for the frame tagged with ``_FAIL_MARKER``."""
    if bool((a == _FAIL_MARKER).any()):
        raise ValueError("synthetic pair failure")
    return torch.linalg.norm(a.mean(0) - b.mean(0))


def _call_failing(which: str, x, y, mpi_distribute: bool):
    if which == "create":
        return pm.create_pairwise_distance_matrix(
            x, y, normalize=False, distance_metric=_failing_metric, mpi_distribute=mpi_distribute,
        )
    return pm.create_pairwise_distance_matrix_given_rigid_rot(
        x, y, rotation=torch.eye(3), translation=torch.zeros(3), scale=1.0,
        normalize=False, distance_metric=_failing_metric, mpi_distribute=mpi_distribute,
    )


@pytest.mark.parametrize("which", ["create", "given_rigid_rot"])
@pytest.mark.parametrize("fail_frame", [0, 1], ids=["fail-on-rank0", "fail-on-rank1"])
def test_thread_ranks_pair_failure_raises_on_every_rank(which, fail_frame):
    """A pair failing on one rank raises RuntimeError on every rank instead of deadlocking (WR-01).

    x=3, y=1, 2 ranks: pair (k, 0) is owned by rank k % 2, so tagging frame 0 or 1
    makes exactly one rank fail.
    """
    torch.manual_seed(0)
    x, y = _frames(3), _frames(1)
    x[fail_frame]["pos"][0, 0] = _FAIL_MARKER

    results, errors, hung = run_ranks(lambda: _call_failing(which, x, y, mpi_distribute=True), 2)

    assert not any(hung), f"ranks hung after a pair failure: {hung}"
    for r, err in enumerate(errors):
        assert isinstance(err, RuntimeError), f"rank {r} did not raise RuntimeError: {err!r}"
        assert f"rank {fail_frame % 2}" in str(err)
        assert f"pair ({fail_frame}, 0)" in str(err)
        assert "synthetic pair failure" in str(err)
    owner = errors[fail_frame % 2]
    assert isinstance(owner.__cause__, ValueError), "failing rank must chain the original exception"


@pytest.mark.parametrize("which", ["create", "given_rigid_rot"])
def test_serial_pair_failure_raises_original_exception(which):
    """Without MPI the original exception propagates unchanged (WR-01 keeps the serial path)."""
    torch.manual_seed(0)
    x, y = _frames(2), _frames(1)
    x[1]["pos"][0, 0] = _FAIL_MARKER
    with pytest.raises(ValueError, match="synthetic pair failure"):
        _call_failing(which, x, y, mpi_distribute=False)


def test_mpirun_two_ranks_pair_failure(tmp_path):
    """Real ``mpirun -n 2``: a pair failing on rank 1 makes both ranks raise, no deadlock (WR-01)."""
    launcher = mpi_launcher(2)
    try:
        proc = subprocess.run(
            [*launcher, sys.executable, str(_WORKER), "create_fail", "3", str(tmp_path)],
            capture_output=True,
            timeout=_MPIRUN_TIMEOUT_S,
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"pair failure on one rank deadlocked mpirun -n 2 (WR-01) within {_MPIRUN_TIMEOUT_S} s")
    assert proc.returncode == 0, f"{launcher[0]} failed: stderr={proc.stderr.decode(errors='replace')!r}"
    for r in (0, 1):
        report = json.loads((tmp_path / f"rank_{r}.json").read_text())
        assert report["raised"] == "RuntimeError", f"rank {r}: {report}"


def _transform_state(st) -> list[torch.Tensor]:
    tf = st.transform
    return [torch.as_tensor(tf.rot), torch.as_tensor(tf.t), torch.as_tensor(tf.scale),
            st.src_min, st.src_max, st.tgt_min, st.tgt_max]


@pytest.mark.parametrize("size", [2, 3])
def test_thread_ranks_gather_stored_transforms_and_rotations(size):
    """Under MPI every rank returns the serial stored_transforms and rotations (WR-03)."""
    torch.manual_seed(0)
    x, y = _frames(3), _frames(2)

    def run(mpi_distribute):
        return pm.create_pairwise_distance_matrix(
            x, y, normalize=True, distance_metric="euclidean", cpd_type="rigid",
            mpi_distribute=mpi_distribute,
        )

    serial = run(False)
    assert len(serial.stored_transforms) == 6 and serial.rotations.shape == (6, 3, 3)

    results, errors, hung = run_ranks(lambda: run(True), size, timeout=60.0)
    assert not any(hung), f"ranks hung: {hung}"
    assert not any(errors), f"rank errors: {errors}"
    for r, res in enumerate(results):
        assert torch.equal(res.cost_matrix, serial.cost_matrix), f"rank {r} cost matrix differs"
        assert list(res.stored_transforms) == list(serial.stored_transforms), f"rank {r} keys differ"
        for key, st in serial.stored_transforms.items():
            for got, want in zip(_transform_state(res.stored_transforms[key]), _transform_state(st)):
                assert torch.equal(got, want), f"rank {r} transform {key} differs"
        assert torch.equal(res.rotations, serial.rotations), f"rank {r} rotations differ"
