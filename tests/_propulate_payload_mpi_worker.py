"""MPI worker for tests/test_propulate_payload_mpi.py (Phase 62, 62-06).

Run inside ``mpirun -n 2 python tests/_propulate_payload_mpi_worker.py <strategy> <out_dir>``.

Each rank builds the same cheap synthetic sanity-tier config (shared
``output_dir`` ``<out_dir>/hpo``) with ``alignment_method="cpd"`` and
``label_transfer_method="cpd_weighted"`` and calls
``HyperparamOptimizer(cfg).run()``.

To make the label-transfer flags deterministically non-empty, the worker
substitutes one dependency, exactly like ``_injected_alignment_stage`` in
tests/test_optimizer_scoring_contract.py (62-05): a subclass of the real
``AlignmentStage`` runs the real DTW, rigid CPD and posterior capture, then
zeroes the first receiver column of every frame's posterior. The real
``LabelTransferStage`` therefore repairs one receiver row per frame and reports
one fallback flag per frame; ``_objective``, the MPI reduction and
``save_best_params`` run unmodified. Every rank writes
``<out_dir>/rank_<r>.json`` = ``{"raised", "message", "n_history"}`` and exits 0.
"""

import json
import sys
from pathlib import Path

# zreg.* -> torch -> eval.* -> mpi4py import order (macOS-ARM libomp SIGABRT rule)
from zreg.core.dataset import zRegPointCloud  # noqa: F401
import torch  # noqa: F401

from eval.config import EvalConfig
from eval.runners import optimizer as optimizer_module
from eval.runners.optimizer import HyperparamOptimizer
from eval.stages import AlignmentStage
from mpi4py import MPI

_N_ZERO = 1


class _InjectedAlignmentStage(AlignmentStage):
    """Real AlignmentStage whose posterior has its first receiver column zeroed."""

    def run(self, source, target, params):
        result = super().run(source, target, params)
        injected = {}
        for tk, est in result.estep_results.items():
            pmat = est.pmat.clone()
            pmat[:, :_N_ZERO] = 0.0
            injected[tk] = est._replace(pmat=pmat)
        return result.model_copy(update={"estep_results": injected})


def _config(out_dir: Path, strategy: str) -> EvalConfig:
    return EvalConfig(
        data_path=str(out_dir / "unused.mat"),
        output_dir=str(out_dir / "hpo"),
        pipeline_mode="synthetic",
        run_alignment=True,
        alignment_method="cpd",
        run_label_transfer=True,
        label_transfer_method="cpd_weighted",
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
        n_trials=2,
        search_strategy=strategy,
        search_space={"cpd_penalty": ["rigid"], "k_neighbours": [3, 4]},
        save_plots=False,
    )


def main(strategy: str, out_dir: str) -> None:
    out = Path(out_dir)
    rank = MPI.COMM_WORLD.Get_rank()
    optimizer_module.AlignmentStage = _InjectedAlignmentStage
    report: dict = {"raised": False, "message": "", "n_history": None}
    try:
        result = HyperparamOptimizer(_config(out, strategy)).run()
        report["n_history"] = len(result.history)
    except RuntimeError as exc:
        report["raised"] = True
        report["message"] = str(exc)
    (out / f"rank_{rank}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
