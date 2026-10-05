"""MPI worker for tests/test_run_eval_mpi.py (Phase 63 HPC-04).

Run inside ``mpirun -n 2 python tests/_run_eval_mpi_worker.py <scenario> <tmp_dir>``.

Rank 0 writes the cheap synthetic sanity-tier config to ``<tmp_dir>/cfg.yaml``;
every rank then waits on ``MPI.COMM_WORLD.Barrier()`` before reading it, so the
harness itself has no write/read race.  Rank 1 sleeps 1.2 s after the barrier
so that per-process ``datetime.now()`` stamps differ by at least one second.
Both ranks then call the real ``run_eval.main`` with
``--output-dir <tmp_dir>/runs``.  Only ``EvaluationRunner.run`` is wrapped (it
still calls through to the real method) to count invocations per rank.

Every rank writes ``<tmp_dir>/rank_<r>.json`` =
``{"rc", "eval_runs", "error", "elapsed_s"}`` and exits 0, so a failure in
``run_eval.main`` is recorded instead of being lost.

Scenarios
---------
optimize          : ``--mode optimize``
full              : ``--mode full``
eval              : ``--mode eval`` (no best_params.json -> default params)
rank0_setup_fails : ``--mode optimize`` with ``<tmp_dir>/runs`` read-only, so
                    rank 0 cannot create the run directory; every rank must
                    raise instead of hanging.
"""

import json
import os
import stat
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# zreg.* -> torch -> eval.* -> mpi4py import order (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud  # noqa: F401
import torch  # noqa: F401

import yaml

import run_eval
from eval.runners.eval_runner import EvaluationRunner
from mpi4py import MPI

_MODES = {
    "optimize": "optimize",
    "full": "full",
    "eval": "eval",
    "rank0_setup_fails": "optimize",
}


def _config(out_dir: Path) -> dict:
    return {
        "data_path": str(out_dir / "unused.mat"),
        "output_dir": str(out_dir / "runs"),
        "pipeline_mode": "synthetic",
        "run_alignment": True,
        "run_label_transfer": True,
        "transform_spec": {
            "type": "subsample_pair",
            "synthesize": True,
            "seed": 1,
            "n_classes": 3,
            "n_points": 80,
            "source_fraction": 0.8,
            "target_fraction": 0.8,
            "rotation_deg": 0.0,
            "rotation_axis": [0.0, 0.0, 1.0],
            "scale_factor": 1.0,
        },
        "tier": "sanity",
        "n_trials": 1,
        "search_strategy": "grid",
        "search_space": {"k_neighbours": [3]},
        # EvaluationRunner needs every stage key (eval mode has no best_params.json)
        "default_params": {
            "window_size": 5,
            "step": 1,
            "cpd_penalty": None,
            "dtw_dist_fn": "euclidean",
            "n_breakpoints": 5,
            "k_neighbours": 3,
            "dist_metric": "euclidean",
            "smoothing": 0.0,
            "threshold": 0.0,
        },
    }


def main(scenario: str, tmp_dir: str) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    out = Path(tmp_dir)
    cfg_path = out / "cfg.yaml"
    runs = out / "runs"

    if rank == 0:
        cfg_path.write_text(yaml.safe_dump(_config(out)))
        if scenario == "rank0_setup_fails":
            runs.mkdir(parents=True, exist_ok=True)
            os.chmod(runs, stat.S_IRUSR | stat.S_IXUSR)  # read-only: mkdir of the stamp dir fails
    comm.Barrier()  # nobody reads cfg.yaml before rank 0 finished writing it

    if rank == 1:
        time.sleep(1.2)  # stagger the per-process datetime.now() stamps

    counter = {"n": 0}
    real_run = EvaluationRunner.run

    def _counting_run(self, *a, **kw):
        counter["n"] += 1
        return real_run(self, *a, **kw)

    EvaluationRunner.run = _counting_run

    report: dict = {"rc": None, "eval_runs": 0, "error": None, "elapsed_s": None}
    t0 = time.monotonic()
    try:
        report["rc"] = run_eval.main(
            ["--config", str(cfg_path), "--mode", _MODES[scenario], "--output-dir", str(runs)]
        )
    except BaseException as exc:  # record, do not lose, the failure
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        EvaluationRunner.run = real_run
    report["elapsed_s"] = time.monotonic() - t0
    report["eval_runs"] = counter["n"]
    (out / f"rank_{rank}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
