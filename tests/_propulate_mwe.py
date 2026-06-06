"""MPI worker for test_propulate.py::TestPropulateMPIIntegration.

Run inside `mpirun -n 2 python tests/_propulate_mwe.py <out_path>`.
Each rank runs PropulateSearch.search() on a trivial objective; rank 0
writes the results list (as JSON) to <out_path>.
"""

import json
import sys
import tempfile
from pathlib import Path

# zreg.* → torch → eval.* import order (macOS-ARM libomp SIGABRT rule)
from zreg.dataset import zRegPointCloud  # noqa: F401
import torch  # noqa: F401

from eval.search_strategies import PropulateSearch
from mpi4py import MPI


def main(out_path: str) -> None:
    # Pass lists (not tuples) — PropulateSearch performs the tuple conversion (D-07)
    search_space = {"x": [-5.0, 5.0], "y": [-5.0, 5.0]}

    def trivial_obj(params: dict) -> float:
        # Framework maximises; sphere objective returns negated sum-of-squares
        return -(params["x"] ** 2 + params["y"] ** 2)

    with tempfile.TemporaryDirectory() as tmp:
        results = PropulateSearch().search(
            search_space=search_space,
            objective_fn=trivial_obj,
            n_trials=4,
            output_dir=tmp,
        )

    # Only rank 0 writes results — other ranks return [] from search()
    if MPI.COMM_WORLD.Get_rank() == 0:
        Path(out_path).write_text(json.dumps([[p, s] for p, s in results]))


if __name__ == "__main__":
    main(sys.argv[1])
