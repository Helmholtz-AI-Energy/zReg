"""Runner classes for the zReg evaluation framework.

``EvaluationRunner`` orchestrates DataFactory → AlignmentStage →
LabelTransferStage → MetricsEngine → EvalReport (Phase 21).

``HyperparamOptimizer`` orchestrates tiered hyperparameter search via
GridSearch / RandomSearch / BayesianSearch strategies (Phase 22, FRAME-09).
"""

from .eval_runner import EvaluationRunner
from .optimizer import HyperparamOptimizer

__all__ = ["EvaluationRunner", "HyperparamOptimizer"]
