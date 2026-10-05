"""PipelineStage abstract base class for evaluation-framework stages.

Defines the contract every concrete stage (AlignmentStage in Phase 19,
LabelTransferStage in Phase 20) must implement.  The single abstract
method ``run`` is required; ``validate_params`` is a concrete default
(no-op) that concrete subclasses override and that ``run`` calls as its
first line (D-09 — required by convention (D-09); not mechanically enforced
by the ABC).
"""

from abc import ABC, abstractmethod
from typing import Any

# zreg.core.dataset MUST precede any torch import (macOS-ARM libomp SIGABRT;
# enforced in tests/conftest.py:20-24, eval/data_factory.py:18-35,
# eval/metrics.py:53-67, eval/types.py:48-53).
from zreg.core.dataset import zRegPointCloud

from eval.config import EvalConfig
from eval.types import StageResult

__all__ = ["PipelineStage"]


class PipelineStage(ABC):
    """Abstract base class for evaluation-framework pipeline stages.

    Subclasses implement a single computation step (alignment or label
    transfer) over a dataset and produce a typed ``StageResult``.

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration.  Stored as ``self.config``
        at construction; no I/O is performed.
    """

    def __init__(self, config: EvalConfig) -> None:
        """Store the evaluation configuration.

        Parameters
        ----------
        config : EvalConfig
            Validated evaluation configuration.
        """
        self.config = config

    @abstractmethod
    def run(
        self,
        source: dict[int, zRegPointCloud],
        target: dict[int, zRegPointCloud],
        params: dict[str, Any],
    ) -> StageResult:
        """Execute the stage and return a typed result.

        Parameters
        ----------
        source : dict[int, zRegPointCloud]
            Source trajectory keyed by integer frame index.  The dataset
            whose coordinates DTW/label transfer aligns from.
        target : dict[int, zRegPointCloud]
            Target trajectory keyed by integer frame index.  The dataset
            DTW/label transfer aligns against.  May be the same as source
            for smoke-test scenarios; must be distinct for true paired
            alignment.
        params : dict[str, Any]
            Hyperparameters for this stage.  Validated by
            ``validate_params`` before use.

        Returns
        -------
        StageResult
            ``AlignResult`` or ``LabelResult`` depending on the concrete
            stage implementation.

        Notes
        -----
        Concrete subclasses MUST call ``self.validate_params(params)``
        as the first line of ``run`` (D-09).  This is required by
        convention (D-09); not mechanically enforced by the ABC.
        """
        ...

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate the hyperparameter dict for this stage.

        Default implementation is a no-op — subclasses that need
        validation MUST override this method.

        Parameters
        ----------
        params : dict[str, Any]
            Hyperparameter dict to validate.  The default implementation
            accepts any dict without inspection.

        Returns
        -------
        None
            Returns ``None`` implicitly on success.

        Raises
        ------
        ValueError
            On missing required keys or invalid values.  Raised by
            concrete subclass implementations (subclass-defined).

        Notes
        -----
        Default implementation is a no-op — subclasses that need
        validation MUST override.  ``run()`` calls this as its first
        line (D-09) so callers cannot bypass validation.
        """
        return None
