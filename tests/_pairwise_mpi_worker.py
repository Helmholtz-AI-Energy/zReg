"""MPI worker for tests/test_pairwise_mpi.py (Phase 61 DIST-01).

Run inside ``mpirun -n 2 python tests/_pairwise_mpi_worker.py <which> <nx> <out_dir>``.

``which`` is ``create`` (``create_pairwise_distance_matrix``) or
``given_rigid_rot`` (``create_pairwise_distance_matrix_given_rigid_rot``), or
``create_fail`` (a pair owned by rank 1 raises; each rank reports the exception type).
Every rank builds the same seeded x (``nx`` frames) and y (1 frame), computes
the serial matrix locally, then the MPI-distributed matrix, and writes
``<out_dir>/rank_<r>.json`` = ``{"equal": bool, "rank": r}``.
"""

import json
import sys
from pathlib import Path

# zreg.* -> torch -> mpi4py import order (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud
from zreg.algorithms import pairwise_distance_matrix as pm
import torch

from mpi4py import MPI


_FAIL_MARKER = 999.0


def _failing_metric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if bool((a == _FAIL_MARKER).any()):
        raise ValueError("synthetic pair failure")
    return torch.linalg.norm(a.mean(0) - b.mean(0))


def _call(which: str, x, y, mpi_distribute: bool) -> torch.Tensor:
    if which == "create":
        return pm.create_pairwise_distance_matrix(
            x, y, normalize=False, distance_metric="euclidean", mpi_distribute=mpi_distribute
        ).cost_matrix
    if which == "given_rigid_rot":
        return pm.create_pairwise_distance_matrix_given_rigid_rot(
            x, y, rotation=torch.eye(3), translation=torch.zeros(3), scale=1.0,
            normalize=False, distance_metric="euclidean", mpi_distribute=mpi_distribute,
        )
    raise SystemExit(f"unknown function selector {which!r}")


def main() -> None:
    which, nx, out_dir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    rank = MPI.COMM_WORLD.rank

    torch.manual_seed(0)
    x = {i: zRegPointCloud(pos=torch.randn(10, 3)) for i in range(nx)}
    y = {0: zRegPointCloud(pos=torch.randn(10, 3))}

    if which == "create_fail":
        # Pair (1, 0) is owned by rank 1 only; every rank must raise, none may hang (WR-01).
        x[1]["pos"][0, 0] = _FAIL_MARKER
        raised = None
        try:
            pm.create_pairwise_distance_matrix(
                x, y, normalize=False, distance_metric=_failing_metric, mpi_distribute=True
            )
        except Exception as exc:  # noqa: BLE001 - reported to the test
            raised = type(exc).__name__
        (out_dir / f"rank_{rank}.json").write_text(json.dumps({"raised": raised, "rank": rank}))
        return

    serial = _call(which, x, y, mpi_distribute=False)
    distributed = _call(which, x, y, mpi_distribute=True)

    report = {"equal": bool(torch.equal(serial, distributed)), "rank": rank}
    if which == "create":
        # WR-03: CPD transforms and rotations are gathered, not rank-local.
        rs = pm.create_pairwise_distance_matrix(x, y, distance_metric="euclidean", cpd_type="rigid")
        rd = pm.create_pairwise_distance_matrix(
            x, y, distance_metric="euclidean", cpd_type="rigid", mpi_distribute=True
        )
        same_tf = list(rs.stored_transforms) == list(rd.stored_transforms) and all(
            torch.equal(rs.stored_transforms[k].transform.rot, rd.stored_transforms[k].transform.rot)
            and torch.equal(rs.stored_transforms[k].transform.t, rd.stored_transforms[k].transform.t)
            for k in rs.stored_transforms
        )
        report["equal_transforms"] = bool(same_tf and torch.equal(rs.rotations, rd.rotations))
    (out_dir / f"rank_{rank}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
