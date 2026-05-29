"""Runner classes for the zReg evaluation framework.

``EvaluationRunner`` orchestrates DataFactory → AlignmentStage →
LabelTransferStage → MetricsEngine → EvalReport (Phase 21).
"""

from .eval_runner import EvaluationRunner

__all__ = ["EvaluationRunner"]
