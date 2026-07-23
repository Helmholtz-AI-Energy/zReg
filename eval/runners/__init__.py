"""Runner classes for the zReg evaluation framework.

``EvaluationRunner`` orchestrates DataFactory → AlignmentStage →
LabelTransferStage → MetricsEngine → EvalReport (Phase 21).

``HyperparamOptimizer`` orchestrates tiered hyperparameter search via
GridSearch / RandomSearch / BayesianSearch strategies (Phase 22, FRAME-09).

``LabelTransferBenchmark`` orchestrates a multi-method comparison run across
all four ``LabelTransferStage`` methods (``knn_voting``, ``cpd_weighted``,
``pointnet2``, ``egnn``) against the same raw source/target pair — additive
to ``EvaluationRunner``, which remains unchanged (Phase 49).
"""

from .benchmark_runner import LabelTransferBenchmark
from .eval_runner import EvaluationRunner
from .optimizer import HyperparamOptimizer

__all__ = ["EvaluationRunner", "HyperparamOptimizer", "LabelTransferBenchmark"]
