"""Pipeline-stage classes for the zReg evaluation framework.

Stages implement a fixed ABC (``PipelineStage``) so the
``EvaluationRunner`` (Phase 21) can treat them uniformly.

Phase 19 ships ``PipelineStage`` (this package's ABC) and
``AlignmentStage``.  Phase 20 adds ``LabelTransferStage``.
"""

from .base import PipelineStage
from .alignment import AlignmentStage
from .label_transfer import LabelTransferStage

__all__ = ["PipelineStage", "AlignmentStage", "LabelTransferStage"]
