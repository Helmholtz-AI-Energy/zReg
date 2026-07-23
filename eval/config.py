"""Configuration model for the zReg evaluation framework.

``EvalConfig`` is a pydantic v2 ``BaseModel`` that loads and validates
YAML-based experiment configurations.  ``EvalConfigError`` (a ``ValueError``
subclass) wraps all ``pydantic.ValidationError`` and file-I/O exceptions so
that the Phase 23 CLI can display a single readable line without a stacktrace
(D-03).  ``EvalConfig.from_yaml`` is the canonical YAML loader (D-04).
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = ["EvalConfig", "EvalConfigError", "AlignmentPreprocessingConfig", "DataPreprocessingConfig"]


class EvalConfigError(ValueError):
    """Raised by EvalConfig.from_yaml for all configuration failure modes.

    Wraps ``pydantic.ValidationError`` (missing/unknown/bad-type fields),
    ``FileNotFoundError``, and ``yaml.YAMLError`` into a single human-readable
    message so that run_eval.py (Phase 23) can catch this and print the message
    without exposing a pydantic stacktrace to the user (D-03).
    """


class AlignmentPreprocessingConfig(BaseModel):
    """Pre-DTW alignment preprocessing options (Phase 41).

    Selects a preprocessing strategy applied before DTW inside
    ``AlignmentStage.run`` (wired in plan 41-02).  ``method`` is required;
    the velocity-landmark parameters carry defaults and are only consulted
    when ``method='velocity_landmarks'``.  Unknown keys are rejected with
    ``extra='forbid'`` so config typos are caught at parse time.

    Parameters
    ----------
    method : {"principal_axes", "velocity_landmarks"}
        Preprocessing strategy.  ``"principal_axes"`` applies a PCA rotation
        (``zreg.preprocessing.compute_pca_rotation``); ``"velocity_landmarks"``
        flags high-velocity frames (``detect_velocity_landmarks``).
    velocity_threshold : float
        Velocity above which a frame is recorded as a landmark.  Default 0.5.
    velocity_metric : {"mean", "max"}
        Reduction applied to per-point displacement norms.  Default "mean".
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["principal_axes", "velocity_landmarks"]
    velocity_threshold: float = 0.5
    velocity_metric: Literal["mean", "max"] = "mean"


class DataPreprocessingConfig(BaseModel):
    """Per-trajectory data preprocessing options (Phase 43).

    Selects a scaling strategy applied to the ``pos`` field of each trajectory
    after subsampling in ``DataFactory.load_real()`` and
    ``DataFactory.load_target()``.  Only the ``pos`` field is scaled; ``label``,
    ``id``, and ``fps-idx`` are passed through unchanged.  Statistics are
    computed globally across all frames of the trajectory (concatenated).

    Three methods are supported:

    - ``"standardize"`` (default, D-01): z-score normalization — subtract the
      per-dimension mean and divide by the per-dimension standard deviation
      (Bessel-corrected).  After scaling, each dimension has mean ≈ 0 and
      std ≈ 1.
    - ``"normalize"``: min-max scaling — maps each dimension to the range
      ``[0, 1]`` using ``(pos - min) / (max - min + eps)``.
    - ``"robust"``: robust scaling — subtract the per-dimension median, divide
      by the per-dimension inter-quartile range (IQR = Q75 - Q25), then clip to
      ``±robust_outlier_threshold``.  Resistant to outliers.

    Unknown keys are rejected with ``extra='forbid'`` so config typos are caught
    at parse time.

    Parameters
    ----------
    method : {"standardize", "normalize", "robust"}
        Scaling strategy.  Default ``"standardize"`` (z-score, D-01).
    robust_outlier_threshold : float
        Clipping threshold used only when ``method="robust"``.  Values are
        clipped to ``[-robust_outlier_threshold, +robust_outlier_threshold]``
        after IQR scaling.  Silently ignored for other methods — not an error.
        Default ``3.0`` (D-07).
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["standardize", "normalize", "robust"] = "standardize"
    robust_outlier_threshold: float = 3.0


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
        Optimizer strategy: ``"grid"``, ``"random"``, ``"sobol"``
        (quasi-random Sobol sequence sampling — low-discrepancy, reproducible
        via ``sobol_seed``; falls back to ``RandomSearch`` when
        ``n_trials < 8``), ``"bayesian"``, ``"propulate"`` (MPI-parallel
        evolutionary search via the propulate library; requires the
        ``zreg[propulate]`` optional extra), or ``"auto"`` (resolved at
        run-time by ``HyperparamOptimizer._detect_backend`` based on MPI
        world size and the ``SLURM_JOB_ID`` environment variable — returns
        ``"propulate"`` when running under MPI with world_size > 1 or when
        ``SLURM_JOB_ID`` is set, otherwise ``"bayesian"``).  EXT-03.
        Default ``"sobol"`` (OPT-04-02).
    tier : str
        Search tier: ``"sanity"``, ``"dev"``, or ``"full"``.
    n_trials : int
        Number of optimizer trials (default 10).
    sobol_seed : int
        Seed passed to ``scipy.stats.qmc.Sobol`` when ``sobol_randomize=True``
        (scrambled Owen sequence); silently ignored when
        ``sobol_randomize=False`` (classical Van der Corput, D-04).
        Default 42, consistent with ``BayesianSearch`` (``TPESampler(seed=42)``).
    sobol_randomize : bool
        When ``True`` (default), uses the scrambled Owen sequence (better
        uniformity, fully reproducible via ``sobol_seed``).  When ``False``,
        uses the classical Van der Corput sequence and ``sobol_seed`` is
        silently ignored (D-04).  Default ``True``.
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
    pipeline_mode : str
        Pipeline mode: ``'paired'`` loads ``target_data_path`` via
        ``DataFactory.load_target()``; ``'synthetic'`` is reserved for Phase 31
        and is currently a no-op.  Default ``'paired'``.
    target_data_path : str or None
        Optional path to the second dataset.  Required (non-None) when
        ``pipeline_mode='paired'`` and ``DataFactory.load_target()`` is
        invoked; ignored otherwise.  ``EvalConfigError`` is raised at
        ``load_target()`` call time, not at ``EvalConfig`` construction (D-05).
        Default ``None``.
    transform_spec : dict or None
        Transform specification dict for synthetic mode.  Required (non-None)
        when ``pipeline_mode='synthetic'`` and
        ``DataFactory.generate_target()`` is invoked; ignored otherwise.
        ``ValueError`` is raised at ``generate_target()`` call time (not at
        ``EvalConfig`` construction) per the error-at-use-time pattern (D-08).
        Dict must contain at least one recognised augmentation key beyond the
        ``"type"`` discriminator (e.g. ``{"type": "rigid", "rotation_deg":
        30.0, "rotation_axis": [0, 0, 1]}`` or ``{"type": "noise", "sigma":
        0.1}``).  Default ``None``.
    target_data_format : str or None
        Format override for the target dataset loader.  Accepted values:
        ``"tracklets"`` or ``"csv"``.  ``None`` (default) inherits
        ``data_format`` so all existing configs without this key remain valid.
        Validated at ``load_target()`` call time, not at construction.
    max_points_per_frame : int or None
        If set, subsample each frame of the loaded source and target datasets
        to at most this many points (random without replacement, seed 42).
        ``None`` (default) disables subsampling — all points are kept.
        Applied by ``DataFactory.load_real()`` and ``DataFactory.load_target()``
        immediately after loading so all downstream stages see the reduced cloud.
    alignment_method : str
        Registration method for spatial alignment: ``'cpd'`` (Coherent Point Drift),
        ``'icp'`` (Open3D ICP, point-to-point rigid), or ``'swd'`` (Sliced Wasserstein
        Distance with variant selection). Default ``'cpd'``. Independent of ``dtw_dist_fn``
        (Phase 40 Pitfall 1: alignment_method and dtw_dist_fn are independent choices).
    swd_variant : str
        SWD variant for spatial alignment when ``alignment_method='swd'``. Accepted values:
        ``'swd'``, ``'aswd'``, ``'oswd'``, ``'gswd'``, ``'pswd'``, ``'maxswd'``.
        Default ``'aswd'``. Only validated when ``alignment_method='swd'``.
    label_transfer_method : str
        Label transfer method used by ``LabelTransferStage`` when the stage's
        ``params`` dict omits ``"method"``: ``'knn_voting'`` (k-nearest-neighbour
        majority vote, existing default behaviour) or ``'cpd_weighted'``
        (CPD E-step posterior-weighted average, requires ``AlignResult.estep_results``
        for the frame pair being processed). Default ``'knn_voting'``.
    data_preprocessing : DataPreprocessingConfig or None
        Per-trajectory data scaling applied after subsampling in
        ``DataFactory.load_real()`` and ``DataFactory.load_target()``.  Only
        the ``pos`` field is scaled; ``label``, ``id``, and ``fps-idx`` are
        passed through unchanged.  Defaults to ``DataPreprocessingConfig()``
        (z-score standardization, D-01) — standardization is ON for all
        existing configs without any YAML change.  Set to ``None`` to disable
        preprocessing entirely (D-02).

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
    search_strategy: Literal["grid", "random", "bayesian", "propulate", "sobol", "auto"] = "sobol"
    tier: Literal["sanity", "dev", "full"] = "sanity"
    n_trials: int = 10
    sobol_seed: int = 42
    sobol_randomize: bool = True
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
    pipeline_mode: Literal["paired", "synthetic"] = "paired"
    target_data_path: str | None = None
    transform_spec: dict | None = None
    target_data_format: str | None = None
    max_points_per_frame: int | None = None
    alignment_method: str = Field(default="cpd", description="Registration method: 'cpd', 'icp', or 'swd'")
    swd_variant: str = Field(
        default="aswd",
        description="SWD variant for alignment_method='swd': 'swd', 'aswd', 'oswd', 'gswd', 'pswd', or 'maxswd'"
    )
    label_transfer_method: str = Field(
        default="knn_voting",
        description="Label transfer method: 'knn_voting', 'cpd_weighted', 'pointnet2', or 'egnn'"
    )
    device: str = Field(
        default="cpu",
        description="Compute device: 'cpu', 'cuda', 'cuda:0', 'cuda:1', or 'mps'"
    )
    egnn_checkpoint_path: str | None = Field(
        default=None,
        description=(
            "Path to a Phase 47 eGNN checkpoint (.pt), required when "
            "label_transfer_method='egnn'. Validated at LabelTransferStage.run() call "
            "time, not at EvalConfig construction (mirrors target_data_path, D-05)."
        ),
    )
    pointnet2_checkpoint_path: str | None = Field(
        default=None,
        description=(
            "Path to a Phase 47 PointNet++ checkpoint (.pt), required when "
            "label_transfer_method='pointnet2'. Validated at LabelTransferStage.run() call "
            "time, not at EvalConfig construction (mirrors target_data_path, D-05)."
        ),
    )
    alignment_preprocessing: AlignmentPreprocessingConfig | None = None
    data_preprocessing: DataPreprocessingConfig | None = Field(
        default_factory=DataPreprocessingConfig
    )

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

    @field_validator("alignment_method")
    @classmethod
    def validate_alignment_method(cls, v: str) -> str:
        """Validate that alignment_method is 'cpd', 'icp', or 'swd'.

        Parameters
        ----------
        v : str
            The alignment_method value to validate.

        Returns
        -------
        str
            The validated alignment_method value.

        Raises
        ------
        ValueError
            If alignment_method is not 'cpd', 'icp', or 'swd'.
        """
        if v not in ("cpd", "icp", "swd"):
            raise ValueError(f"alignment_method must be 'cpd', 'icp', or 'swd'; got {v!r}")
        return v

    @field_validator("swd_variant")
    @classmethod
    def validate_swd_variant(cls, v: str, info) -> str:
        """Validate swd_variant only when alignment_method='swd'.

        Parameters
        ----------
        v : str
            The swd_variant value to validate.
        info : ValidationInfo
            Pydantic validation info containing data from other fields.

        Returns
        -------
        str
            The validated swd_variant value.

        Raises
        ------
        ValueError
            If alignment_method='swd' and swd_variant is not one of the allowed values.
        """
        # Only validate swd_variant if alignment_method is explicitly 'swd'
        # data contains all fields being validated
        if info.data.get("alignment_method") == "swd":
            if v not in ("swd", "aswd", "oswd", "gswd", "pswd", "maxswd"):
                raise ValueError(
                    f"swd_variant must be one of {{'swd', 'aswd', 'oswd', 'gswd', 'pswd', 'maxswd'}}; got {v!r}"
                )
        return v

    @field_validator("label_transfer_method")
    @classmethod
    def validate_label_transfer_method(cls, v: str) -> str:
        """Validate that label_transfer_method is one of the four supported methods.

        Parameters
        ----------
        v : str
            The label_transfer_method value to validate.

        Returns
        -------
        str
            The validated label_transfer_method value.

        Raises
        ------
        ValueError
            If label_transfer_method is not 'knn_voting', 'cpd_weighted',
            'pointnet2', or 'egnn'.
        """
        if v not in ("knn_voting", "cpd_weighted", "pointnet2", "egnn"):
            raise ValueError(
                "label_transfer_method must be 'knn_voting', 'cpd_weighted', "
                f"'pointnet2', or 'egnn'; got {v!r}"
            )
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate that device is one of the supported compute device identifiers.

        Parameters
        ----------
        v : str
            The device value to validate.

        Returns
        -------
        str
            The validated device value.

        Raises
        ------
        ValueError
            If device is not one of 'cpu', 'cuda', 'cuda:0', 'cuda:1', or 'mps'.
        """
        _VALID_DEVICES = {"cpu", "cuda", "cuda:0", "cuda:1", "mps"}
        if v not in _VALID_DEVICES:
            raise ValueError(
                f"device must be one of {{'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'}}; got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def validate_device_icp_compat(self) -> "EvalConfig":
        """Raise ValueError when device is not 'cpu' and alignment_method is 'icp'.

        Open3D ICP requires CPU-resident tensors and performs explicit
        .cpu().numpy() round-trips in icp.py (lines 114-115 and 196-197).
        Requesting a non-CPU device with ICP is therefore always an error.

        Returns
        -------
        EvalConfig
            The validated model instance (self).

        Raises
        ------
        ValueError
            If device is not 'cpu' and alignment_method is 'icp'.
        """
        if self.device != "cpu" and self.alignment_method == "icp":
            raise ValueError(
                f"device='{self.device}' is incompatible with alignment_method='icp': "
                "Open3D ICP requires CPU-resident tensors "
                "(explicit .cpu().numpy() round-trips in icp.py:114-115, 196-197)"
            )
        return self
