"""MPI worker for tests/test_failed_trials_mpi.py (Phase 59 NUM-05).

Run inside ``mpirun -n 2 python tests/_failed_trials_mpi_worker.py <scenario> <strategy> <out_dir>``.

Each rank builds the cheap synthetic sanity-tier config (shared ``output_dir``
``<out_dir>/hpo``) with a per-rank search space and calls
``HyperparamOptimizer(cfg).run()``.  Trials with ``k_neighbours=3`` succeed;
``k_neighbours=10000`` exceeds the frame point count, so ``LabelTransferStage``
raises a real ``ValueError`` (no mocks).  Every rank writes
``<out_dir>/rank_<r>.json`` = ``{"raised", "message", "best_params"}`` and exits 0.

Scenarios
---------
rank1_fails : rank 0 [3], rank 1 [10000]
rank0_fails : rank 0 [10000], rank 1 [3]
all_fail    : both ranks [10000]
"""

import json
import sys
from pathlib import Path

# zreg.* -> torch -> eval.* -> mpi4py import order (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud  # noqa: F401
import torch  # noqa: F401

from eval.config import EvalConfig
from eval.runners.optimizer import HyperparamOptimizer
from mpi4py import MPI

_OK_K = 3
_FAIL_K = 10000

_SCENARIOS = {
    "rank1_fails": ([_OK_K], [_FAIL_K]),
    "rank0_fails": ([_FAIL_K], [_OK_K]),
    "all_fail": ([_FAIL_K], [_FAIL_K]),
}


def _config(out_dir: Path, strategy: str, k_values: list[int]) -> EvalConfig:
    return EvalConfig(
        data_path=str(out_dir / "unused.mat"),
        output_dir=str(out_dir / "hpo"),
        pipeline_mode="synthetic",
        run_alignment=True,
        run_label_transfer=True,
        transform_spec={
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
        tier="sanity",
        n_trials=1,
        search_strategy=strategy,
        search_space={"k_neighbours": k_values},
    )


def main(scenario: str, strategy: str, out_dir: str) -> None:
    out = Path(out_dir)
    rank = MPI.COMM_WORLD.Get_rank()
    k_values = _SCENARIOS[scenario][rank]
    cfg = _config(out, strategy, k_values)
    report: dict = {"raised": False, "message": "", "best_params": None}
    try:
        result = HyperparamOptimizer(cfg).run()
        report["best_params"] = dict(result.best_params) or None
    except RuntimeError as exc:
        report["raised"] = True
        report["message"] = str(exc)
    (out / f"rank_{rank}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
