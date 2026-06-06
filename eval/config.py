"""Configuration model for the zReg evaluation framework.

``EvalConfig`` is a pydantic v2 ``BaseModel`` that loads and validates
YAML-based experiment configurations.  ``EvalConfigError`` (a ``ValueError``
subclass) wraps all ``pydantic.ValidationError`` and file-I/O exceptions so
that the Phase 23 CLI can display a single readable line without a stacktrace
(D-03).  ``EvalConfig.from_yaml`` is the canonical YAML loader (D-04).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = ["EvalConfig", "EvalConfigError"]


class EvalConfigError(ValueError):
    """Raised by EvalConfig.from_yaml for all configuration failure modes.

    Wraps ``pydantic.ValidationError`` (missing/unknown/bad-type fields),
    ``FileNotFoundError``, and ``yaml.YAMLError`` into a single human-readable
    message so that run_eval.py (Phase 23) can catch this and print the message
    without exposing a pydantic stacktrace to the user (D-03).
    """


class EvalConfig(BaseModel):
    """Typed, validated evaluation-framework configuration.

    Loaded from YAML via ``EvalConfig.from_yaml(path)``.  All fields except
    ``data_path`` carry sensible defaults (D-02); ``val_split`` is an extra
    field beyond FRAME-01 required by FRAME-02 ``prepare_split``.  Unknown YAML
    keys are rejected with ``extra="forbid"`` so typos are caught early.

    Parameters
    ----------
    data_path : str
        Path to the real dataset file (.mat tracklets or .csv).  The only
        required field — all others have defaults.

    Attributes
    ----------
    data_path : str
        Path to the real dataset.
    data_format : str
        Loader to use: ``"tracklets"`` (default) or ``"csv"``.
    ground_truth_path : str or None
        Optional external ground-truth file.  ``None`` means extract from
        the dataset's ``id`` field (D-10).
    n_synthetic : int
        Number of synthetic frames to generate (default 100).
    transform_degree : float
        Perturbation magnitude for synthetic data generation (default 0.1).
    augmentation_params : dict
        Extra keyword arguments passed to augmentation functions.
    run_alignment : bool
        Whether the AlignmentStage should be executed (default True).
    run_label_transfer : bool
        Whether the LabelTransferStage should be executed (default True).
    default_params : dict
        Fallback hyperparameter values merged with search-space trial params
        by ``HyperparamOptimizer._objective`` (Pitfall 4).  Default empty dict.
    search_space : dict
        Hyper-parameter grid / search space for the optimizer.
    search_strategy : str
        Optimizer strategy: ``"grid"``, ``"random"``, ``"bayesian"``,
        ``"propulate"`` (MPI-parallel evolutionary search via the propulate
        library; requires the ``zreg[propulate]`` optional extra), or
        ``"auto"`` (resolved at run-time by
        ``HyperparamOptimizer._detect_backend`` based on MPI world size and
        the ``SLURM_JOB_ID`` environment variable — returns ``"propulate"``
        when running under MPI with world_size > 1 or when ``SLURM_JOB_ID``
        is set, otherwise ``"bayesian"``).  EXT-03.
    tier : str
        Search tier: ``"sanity"``, ``"dev"``, or ``"full"``.
    n_trials : int
        Number of optimizer trials (default 10).
    output_dir : str
        Directory for experiment outputs (default ``"experiments/runs"``).
    save_plots : bool
        Whether to save matplotlib figures (default True).
    verbose : bool
        Enable verbose logging (default False).
    val_split : float
        Fraction of frames to hold out as validation set (default 0.2).
    metric_weights : dict[str, float]
        Relative weights for ``MetricsEngine.compute_score``.  Auto-rescaled
        by their sum so values need not sum to 1.0 exactly.  Default weights
        sum to 1.0; users may override any subset via YAML.  Canonical short-
        name keys: ``"chamfer"``, ``"hausdorff"``, ``"path_smoothness"``,
        ``"temporal_stability"``, ``"f1"``, ``"knn_consistency"``.
    label_names : dict[int, str] or None
        Optional mapping of integer label IDs to descriptive names (e.g.
        ``{0: 'T cell', 1: 'B cell'}``).  Used by ``plot_trajectory`` for
        legend labels in the label-trajectory figure.  Default ``None``.

    Notes
    -----
    Only ``data_path`` is required; every other field has a default (D-02).
    Loading is always done through ``EvalConfig.from_yaml`` — direct
    construction (``EvalConfig(data_path=...)``) is supported but not the
    primary use-case.
    """

    model_config = ConfigDict(extra="forbid")

    data_path: str
    data_format: str = "tracklets"
    ground_truth_path: str | None = None
    n_synthetic: int = 100
    transform_degree: float = 0.1
    augmentation_params: dict = Field(default_factory=dict)
    run_alignment: bool = True
    run_label_transfer: bool = True
    default_params: dict = Field(default_factory=dict)
    search_space: dict = Field(default_factory=dict)
    search_strategy: str = "grid"
    tier: str = "sanity"
    n_trials: int = 10
    output_dir: str = "experiments/runs"
    save_plots: bool = True
    verbose: bool = False
    val_split: float = 0.2
    metric_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "chamfer": 0.35,
            "hausdorff": 0.15,
            "path_smoothness": 0.10,
            "temporal_stability": 0.10,
            "f1": 0.20,
            "knn_consistency": 0.10,
        }
    )
    label_names: dict[int, str] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load and validate an EvalConfig from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        EvalConfig
            Validated configuration instance with defaults applied.

        Raises
        ------
        EvalConfigError
            If the file is not found (``"EvalConfig: file not found: ..."``),
            if the file contains invalid YAML syntax, if a required field
            (``data_path``) is missing, if an unknown field is present, or if
            a field value cannot be coerced to the declared type.  In all
            cases the raw ``pydantic.ValidationError`` is never allowed to
            escape (D-03).
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except FileNotFoundError as e:
            raise EvalConfigError(f"EvalConfig: file not found: {path}") from e
        except yaml.YAMLError as e:
            raise EvalConfigError(f"EvalConfig: invalid YAML in {path}: {e}") from e
        except ValidationError as e:
            first = e.errors()[0]
            field = ".".join(str(x) for x in first["loc"])
            raise EvalConfigError(f"EvalConfig: field '{field}': {first['msg']}") from e
