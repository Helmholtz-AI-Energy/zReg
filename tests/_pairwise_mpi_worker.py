"""MPI worker for tests/test_pairwise_mpi.py (Phase 61 DIST-01).

Run inside ``mpirun -n 2 python tests/_pairwise_mpi_worker.py <which> <nx> <out_dir>``.

``which`` is ``create`` (``create_pairwise_distance_matrix``) or
``given_rigid_rot`` (``create_pairwise_distance_matrix_given_rigid_rot``).
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

    serial = _call(which, x, y, mpi_distribute=False)
    distributed = _call(which, x, y, mpi_distribute=True)

    report = {"equal": bool(torch.equal(serial, distributed)), "rank": rank}
    (out_dir / f"rank_{rank}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
