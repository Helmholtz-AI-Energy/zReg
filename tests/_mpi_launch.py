"""Launcher prefix for the real multi-rank MPI tests.

Inside a Slurm allocation the ranks are started with ``srun``: JSC systems such as JUPITER
ship ParaStationMPI without an ``mpirun``, and srun is the supported launcher there. Outside
Slurm, ``mpirun`` is used. The calling test is skipped when neither launcher or mpi4py is
available.
"""

from __future__ import annotations

import os
import shutil

import pytest


def mpi_launcher(n: int = 2) -> list[str]:
    """Return the argv prefix that starts ``n`` MPI ranks, or skip the calling test.

    Parameters
    ----------
    n : int, optional
        Number of ranks. Default 2.

    Returns
    -------
    list[str]
        e.g. ``["srun", "--ntasks=2", ...]`` or ``["/usr/bin/mpirun", "-n", "2"]``.
    """
    srun = shutil.which("srun") if os.environ.get("SLURM_JOB_ID") else None
    mpirun = shutil.which("mpirun")
    if srun is None and mpirun is None:
        pytest.skip("no MPI launcher: mpirun not on PATH and not inside a Slurm allocation")
    pytest.importorskip("mpi4py")
    if srun is not None:
        # --overlap shares the CPUs already held by the batch step that runs pytest; the job
        # needs at least n tasks (run_gpu_tests.sh requests them).
        return [srun, f"--ntasks={n}", "--nodes=1", "--overlap", "--cpus-per-task=1"]
    return [mpirun, "-n", str(n)]
