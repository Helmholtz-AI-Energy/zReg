"""DataFactory: lazy, cached orchestrator for real and synthetic point cloud trajectories.

Wraps the four existing data primitives (``load_data_from_tracklets``,
``load_shah_from_csv``, the ``zreg.generators`` package, and the two corruption
functions) behind a single class driven by an ``EvalConfig``.  All five public
methods return ``dict[int, zRegPointCloud]`` (or a split tuple).  Construction
is cheap (no I/O) per D-08; data is loaded/generated lazily on first method
call per D-09.  The returned dicts must not be mutated in place — downstream
phases should treat them as read-only (copy.deepcopy them first if mutation is
required).  The generators in ``zreg.generators`` already deep-copy their
inputs, so chaining ``augment(load_real())`` is inherently safe.
"""

# stdlib first
import logging
import math
import random
from pathlib import Path  # noqa: F401  (available for future use)

# zreg.dataset MUST precede import torch (libomp SIGABRT lesson from Phase 12;
# enforced in tests/conftest.py:20-24)
from zreg.core.dataset import (
    load_data_from_tracklets,
    load_shah_from_csv,
    zRegPointCloud,
)
from zreg.data_generation import (
    add_gaussian_noise,
    add_outliers,
    apply_affine,  # noqa: F401  (available; not used in Phase 17 minimal generate_synthetic)
    apply_rigid,
    generate_labels,
    generate_trajectory,
    sample_ball,
    sample_bowl,
)
from zreg.core.transforms import RigidTransformation

# torch AFTER zreg.* imports
import torch

# local sibling module last
from eval.config import EvalConfig, EvalConfigError
from eval.types import TrainingTriple

__all__ = ["DataFactory", "split_seeds"]


def split_seeds(
    n_train: int,
    n_val: int,
    base_seed: int = 0,
) -> tuple[range, range]:
    """Disjoint-by-construction train/val seed ranges (D-03 seed-level holdout).

    Unlike :meth:`DataFactory.prepare_split` (which randomly samples a subset
    of FRAME indices within a single trajectory), this function partitions
    the SEED space itself into two non-overlapping contiguous ranges, so a
    triple generated from a train seed and one generated from a val seed can
    never share a seed. Disjointness is guaranteed by construction (adjacent,
    non-overlapping ranges) — no ``isdisjoint()`` check is needed.

    Parameters
    ----------
    n_train : int
        Number of training seeds. Must be >= 1.
    n_val : int
        Number of validation seeds. Must be >= 0.
    base_seed : int, optional
        First seed of the training range (default 0).

    Returns
    -------
    tuple[range, range]
        ``(train_seeds, val_seeds)`` — ``train_seeds = range(base_seed,
        base_seed + n_train)``; ``val_seeds = range(base_seed + n_train,
        base_seed + n_train + n_val)``.

    Raises
    ------
    ValueError
        If ``n_train < 1`` or ``n_val < 0``.
    """
    if n_train < 1:
        raise ValueError(f"split_seeds: n_train must be >= 1, got {n_train}")
    if n_val < 0:
        raise ValueError(f"split_seeds: n_val must be >= 0, got {n_val}")
    train_seeds = range(base_seed, base_seed + n_train)
    val_seeds = range(base_seed + n_train, base_seed + n_train + n_val)
    return train_seeds, val_seeds


class DataFactory:
    """Lazy, cached factory for real and synthetic trajectories.

    Construction performs NO I/O (D-08).  Data is loaded only on the first
    method call and then cached by reference (D-09).

    **DO NOT mutate** the returned dicts in place — downstream phases must treat
    them as immutable.  If mutation is required, ``copy.deepcopy`` the returned
    dict before modifying it.  The generators in ``zreg.generators`` already
    deep-copy their inputs (verified in ``src/zreg/generators/corruption.py``),
    so the common pattern ``augment(load_real())`` is inherently safe: augment()
    returns a new dict whose frames are deep-copies of the originals.

    The ``augment()`` method dispatches on ``config.augmentation_params`` dict
    keys: ``"sigma"`` applies ``add_gaussian_noise``; ``"n_outliers"`` applies
    ``add_outliers``; both may be present (noise is applied first, outliers
    second — composition order matters).  This locks RESEARCH §Open Question 1.

    Parameters
    ----------
    config : EvalConfig
        Validated evaluation configuration.  Only stored — no I/O is performed.
    """

    def __init__(self, config: EvalConfig) -> None:
        """Store config.  No I/O is performed (D-08).

        Parameters
        ----------
        config : EvalConfig
            The validated experiment configuration produced by
            ``EvalConfig.from_yaml`` or direct construction.
        """
        self.config = config
        # D-03: fast-fail availability check — before any I/O so no resource can be leaked.
        if config.device != "cpu":
            if config.device.startswith("cuda"):
                # Covers "cuda", "cuda:0", "cuda:1" (Pitfall 4 — startswith, not ==)
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"DataFactory: config.device={config.device!r} requested "
                        "but torch.cuda.is_available() is False"
                    )
            elif config.device == "mps":  # pragma: no cover
                if not torch.backends.mps.is_available():
                    raise RuntimeError(
                        f"DataFactory: config.device={config.device!r} requested "
                        "but torch.backends.mps.is_available() is False"
                    )
        self._real_dataset: dict[int, zRegPointCloud] | None = None
        self._synthetic_dataset: dict[int, zRegPointCloud] | None = None
        self._target_dataset: dict[int, zRegPointCloud] | None = None
        self._synthetic_target: dict[int, zRegPointCloud] | None = None
        self._source_dataset: dict[int, zRegPointCloud] | None = None
        self._transform_spec: dict | None = None
        self._preprocessing_stats: dict | None = None

    def load_real(self) -> dict[int, zRegPointCloud]:
        """Load the real dataset from disk (lazy, cached).

        Dispatches to ``load_data_from_tracklets`` or ``load_shah_from_csv``
        depending on ``config.data_format``.  The loaded dict is cached; every
        subsequent call returns the same object reference (D-09).

        Returns
        -------
        dict[int, zRegPointCloud]
            Dataset keyed by integer frame index.  Cached by reference — do
            not mutate in place.

        Raises
        ------
        ValueError
            If ``config.data_format`` is neither ``"tracklets"`` nor ``"csv"``.
        """
        if self._real_dataset is not None:
            return self._real_dataset

        if self.config.data_format == "tracklets":
            # Pitfall 4: discard raw tracklets dict (second tuple element)
            dataset, _ = load_data_from_tracklets(self.config.data_path, device=self.config.device)
        elif self.config.data_format == "csv":
            # Pitfall 5: device has NO default in load_shah_from_csv
            dataset = load_shah_from_csv(self.config.data_path, device=self.config.device)
        else:
            raise ValueError(
                f"DataFactory: unknown data_format {self.config.data_format!r}; "
                f"expected 'tracklets' or 'csv'"
            )

        dataset = self._subsample_to_max(dataset)
        dataset = self._standardize(dataset)
        logging.info(
            "Loaded %s dataset on %s (%d frames)", "source", self.config.device, len(dataset)
        )
        self._real_dataset = dataset
        return dataset

    def load_target(self) -> dict[int, zRegPointCloud]:
        """Load the target dataset from disk (lazy, cached).

        Mirrors ``load_real()`` exactly, reading from ``config.target_data_path``
        instead of ``config.data_path``.  Dispatches to
        ``load_data_from_tracklets`` or ``load_shah_from_csv`` depending on
        ``config.target_data_format`` when set, falling back to
        ``config.data_format``.  The loaded dict is cached; every subsequent
        call returns the same object reference (D-09).

        Returns
        -------
        dict[int, zRegPointCloud]
            Target dataset keyed by integer frame index.  Cached by
            reference — do not mutate in place.

        Raises
        ------
        EvalConfigError
            If ``config.target_data_path is None`` — raised at call time
            per D-05.  Message contains ``"target_data_path"`` as a
            substring so callers can grep for it.
        ValueError
            If ``target_data_format or data_format`` is neither
            ``"tracklets"`` nor ``"csv"``.
        """
        if self._target_dataset is not None:
            return self._target_dataset

        if self.config.target_data_path is None:
            raise EvalConfigError(
                "DataFactory.load_target: target_data_path is required in paired mode "
                "but was None. Set target_data_path in the YAML config."
            )

        if (
            self.config.data_preprocessing is not None
            and self._preprocessing_stats is None
            and self._real_dataset is None
        ):
            import warnings
            warnings.warn(
                "DataFactory.load_target() called before load_real() in paired mode. "
                "Statistics will be computed from the TARGET dataset, not the source. "
                "Call load_real() first to share coordinate space.",
                stacklevel=2,
            )

        fmt = self.config.target_data_format or self.config.data_format
        if fmt == "tracklets":
            # Pitfall 4: discard raw tracklets dict (second tuple element)
            dataset, _ = load_data_from_tracklets(self.config.target_data_path, device=self.config.device)
        elif fmt == "csv":
            # Pitfall 5: device has NO default in load_shah_from_csv
            dataset = load_shah_from_csv(self.config.target_data_path, device=self.config.device)
        else:
            raise ValueError(
                f"DataFactory: unknown data_format {fmt!r}; "
                f"expected 'tracklets' or 'csv'"
            )

        dataset = self._subsample_to_max(dataset)
        dataset = self._standardize(dataset, stats=self._preprocessing_stats)
        logging.info(
            "Loaded %s dataset on %s (%d frames)", "target", self.config.device, len(dataset)
        )
        self._target_dataset = dataset
        return dataset

    def generate_target(
        self,
        dataset: dict[int, zRegPointCloud],
        transform_spec: dict,
    ) -> dict[int, zRegPointCloud]:
        """Apply a transform spec to ``dataset`` and return a synthetic target trajectory.

        Delegates to :meth:`augment` internally (D-01), reusing its dispatch
        table and deep-copy semantics.  The ``"type"`` discriminator key in
        ``transform_spec`` is stripped before the override so that ``augment``
        does not encounter an unrecognised key (D-02).

        The original ``self.config.augmentation_params`` value is saved before
        the call and unconditionally restored in a ``finally`` block — even if
        :meth:`augment` raises (D-02, Pitfall 2).

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Source trajectory.  Stored as ``self._source_dataset`` after the call.
        transform_spec : dict
            Transform specification.  Must be non-empty and must contain at
            least one key beyond the optional ``"type"`` discriminator.  Any
            keys recognised by :meth:`augment` (``"sigma"``, ``"rotation_deg"``,
            ``"rotation_axis"``, ``"scale_factor"``, ``"dropout_fraction"``,
            ``"n_outliers"``, ``"n_new_points"``) are forwarded.

        Returns
        -------
        dict[int, zRegPointCloud]
            Augmented trajectory with positions differing from ``dataset``.
            Also stored as ``self._synthetic_target``.

        Raises
        ------
        ValueError
            If ``transform_spec`` is ``None`` or an empty dict — the result
            would be a no-op (augment returns input by reference; see
            Pitfall 1 in RESEARCH.md).
        ValueError
            If ``transform_spec`` contains only the ``"type"`` discriminator
            key and no augmentation keys — ``augment_params`` would be empty
            after stripping, producing the same no-op result.

        Notes
        -----
        - D-01: delegates augmentation dispatch to :meth:`augment`.
        - D-02: temporary override of ``self.config.augmentation_params``
          inside ``try/finally`` ensures the original value is always restored.
        - D-03: stores ``self._synthetic_target``, ``self._source_dataset``,
          ``self._transform_spec`` for use by
          :meth:`get_synthetic_ground_truth`.
        """
        # Guard: transform_spec must be non-empty
        if not transform_spec:
            raise ValueError(
                "DataFactory.generate_target: transform_spec must be non-empty. "
                "Passing None or {} would produce a no-op (augment returns input "
                "by reference when augmentation_params is empty — Pitfall 1)."
            )
        # Strip the 'type' discriminator key before passing to augment() dispatch (D-02)
        augment_params = {k: v for k, v in transform_spec.items() if k != "type"}
        if not augment_params:
            raise ValueError(
                "DataFactory.generate_target: transform_spec contains only the "
                "'type' key — no augmentation keys remain after stripping 'type'. "
                "Provide at least one augmentation key (e.g. 'sigma', 'rotation_deg')."
            )
        # D-02: save original augmentation_params and restore in finally
        original_params = self.config.augmentation_params
        try:
            self.config.augmentation_params = augment_params
            result = self.augment(dataset)
        finally:
            self.config.augmentation_params = original_params
        # D-03: store instance state after successful augmentation
        self._synthetic_target = result
        self._source_dataset = dataset
        self._transform_spec = transform_spec
        return result

    def generate_training_triple(
        self,
        seed: int,
        n_classes: int = 6,
        shape: str | None = None,
        n_points: int | None = None,
    ) -> TrainingTriple:
        """Generate ONE seed-driven (source, target) training triple (D-01/D-02/D-04).

        Composes three existing/new pieces per 46-RESEARCH.md Pattern 1: (1) a
        single-frame ``sample_ball``/``sample_bowl`` base cloud, (2) categorical
        Voronoi labels via ``generate_labels``, (3) an exact-correspondence
        target via :meth:`generate_target` (whose transform machinery already
        propagates ``label`` through every step). Every call with a different
        ``seed`` produces a genuinely different triple; the same ``seed``
        called twice is bitwise-reproducible.

        Geometry selection: when ``shape`` is not given, ``"ball"`` is used for
        even seeds and ``"bowl"`` for odd seeds (``seed % 2 == 0`` -> ball) —
        this documented rule is how callers/tests determine which geometry a
        given seed produced, rather than an internal attribute.

        Point count: when ``n_points`` is not given, it is drawn uniformly from
        ``[100, 300]`` using a per-seed ``random.Random(seed)`` instance (D-01
        small-variant regime).

        CRITICAL (Pitfall 2): this method deliberately does NOT read or write
        ``self._synthetic_dataset`` — it has no early-return cache, unlike
        :meth:`generate_synthetic`'s D-08/D-09 "construct once, cache by
        reference" contract. Every call recomputes from scratch so that N
        seeds produce N distinct results. :meth:`generate_target` (called
        internally) does write ``self._synthetic_target``/``_source_dataset``
        as a benign side effect of its own contract — those attributes have no
        early-return caching themselves and are simply overwritten on every
        call, so this is safe. Do NOT "fix" this method to cache its result;
        that would silently return the first seed's triple for every
        subsequent seed (see 46-RESEARCH.md Pitfall 2).

        The target's ``transform_spec`` deliberately excludes ``"n_new_points"``
        (46-RESEARCH.md Anti-Patterns): points added by ``sample_new_points``
        get a ``-1`` label sentinel, which is not a valid supervised target.

        Parameters
        ----------
        seed : int
            Seed driving base-cloud geometry, label assignment, transform
            parameters, and augmentation RNG (threaded as ``"augment_seed"``).
        n_classes : int, optional
            Voronoi label vocabulary size (default 6).
        shape : {"ball", "bowl"} or None, optional
            Force a specific geometry; ``None`` (default) alternates by seed
            parity (see above).
        n_points : int or None, optional
            Force a specific point count; ``None`` (default) draws uniformly
            from ``[100, 300]`` per-seed.

        Returns
        -------
        TrainingTriple
            ``source_cloud`` (labeled, ``id=None``), ``target_cloud``
            (transformed, correctly-labeled), and ``seed``.
        """
        rng = random.Random(seed)
        if n_points is None:
            n_points = rng.randint(100, 300)
        if shape is None:
            shape = "ball" if seed % 2 == 0 else "bowl"

        if shape == "ball":
            pos = sample_ball(n_points, seed=seed)
        elif shape == "bowl":
            pos = sample_bowl(n_points, seed=seed)
        else:
            raise ValueError(
                f"DataFactory.generate_training_triple: unknown shape {shape!r}; "
                f"expected 'ball' or 'bowl'"
            )

        base = {0: zRegPointCloud(pos=pos)}
        source = generate_labels(base, n_classes=n_classes, seed=seed)

        transform_spec = {
            "type": "rigid",
            "rotation_deg": rng.uniform(0, 360),
            "rotation_axis": [0.0, 0.0, 1.0],
            "scale_factor": rng.uniform(0.8, 1.2),
            "dropout_fraction": rng.uniform(0.0, 0.2),
            "augment_seed": seed,
        }
        target = self.generate_target(source, transform_spec)

        return TrainingTriple(source_cloud=source[0], target_cloud=target[0], seed=seed)

    def generate_training_set(
        self,
        seeds,
        n_classes: int = 6,
    ) -> list[TrainingTriple]:
        """Batch :meth:`generate_training_triple` over an iterable of seeds.

        Convenient companion to :func:`split_seeds` for constructing a full
        train or val set: ``factory.generate_training_set(train_seeds)``.

        Parameters
        ----------
        seeds : Iterable[int]
            Seeds to generate triples for, e.g. a ``range`` from
            :func:`split_seeds`.
        n_classes : int, optional
            Forwarded to every :meth:`generate_training_triple` call
            (default 6).

        Returns
        -------
        list[TrainingTriple]
            One triple per seed, in the order ``seeds`` was iterated.
        """
        return [self.generate_training_triple(s, n_classes=n_classes) for s in seeds]

    def generate_synthetic(self) -> dict[int, zRegPointCloud]:
        """Generate a synthetic trajectory (lazy, cached).

        Produces a ``dict[int, zRegPointCloud]`` with ``config.n_synthetic``
        frames and 100 points per frame (n_points is hardcoded — there is no
        per-config field for point count in the locked FRAME-01 spec; it may
        be added in a future phase).  Uses ``seed=42`` for reproducibility.
        Subsequent calls return the same object reference (D-09).

        Phase 17 keeps this as a minimal wrapper around ``generate_trajectory``.
        Downstream phases (e.g. Phase 23 scenario configs) may extend it with
        optional ``generate_labels`` / ``apply_rigid`` / ``apply_affine`` driven
        by config — out of scope for Phase 17.

        Returns
        -------
        dict[int, zRegPointCloud]
            Synthetic trajectory keyed by integer frame index.  Cached by
            reference — do not mutate in place.
        """
        if self._synthetic_dataset is not None:
            return self._synthetic_dataset

        traj = generate_trajectory(
            n_points=100,
            n_frames=self.config.n_synthetic,
            seed=42,
        )

        self._synthetic_dataset = traj
        return traj

    def augment(self, dataset: dict[int, zRegPointCloud]) -> dict[int, zRegPointCloud]:
        """Apply augmentations per ``config.augmentation_params``.

        Reads ``self.config.augmentation_params`` (a plain dict).  Recognised
        keys (in dispatch order):

        - ``"sigma"`` (float): applies ``add_gaussian_noise(dataset, sigma=..., seed=augment_seed)``
        - ``"n_outliers"`` (int): applies ``add_outliers(dataset, n_outliers=..., seed=augment_seed)``
        - ``"scale"`` (float, optional): outlier scale for ``add_outliers``, default 3.0
        - ``"scale_factor"`` (float): ``self.scale`` — multiplies every frame's pos by
          ``scale_factor``; applied after noise/outliers
        - ``"rotation_deg"`` (float): ``self.rotate`` via Rodrigues' rotation formula;
          optional ``"rotation_axis"`` (list[float], default [0, 0, 1]) selects the
          axis of rotation; applied after scale_factor
        - ``"dropout_fraction"`` (float): ``self.drop_points`` — randomly removes the
          specified fraction of points per frame; applied after scale/rotate
        - ``"n_new_points"`` (int): ``self.sample_new_points`` — appends uniform-in-bbox
          points per frame; applied last
        - ``"augment_seed"`` (int, optional): per-seed RNG override (Pitfall 3,
          46-RESEARCH.md) forwarded to all four stochastic sub-calls above
          (``add_gaussian_noise``, ``add_outliers``, ``drop_points``,
          ``sample_new_points``) in place of a hardcoded ``42``.  Default: 42
          when absent, so every existing caller that does not set
          ``"augment_seed"`` gets byte-identical legacy behavior.  This key is
          NOT itself a transform — it never appears in the dispatch ``if``
          chain and does not, by itself, trigger any augmentation step.

        Dispatch order: sigma → n_outliers → scale_factor → rotation_deg →
        dropout_fraction → n_new_points.

        Missing keys skip the corresponding step.  An empty dict (or a dict
        containing only ``"augment_seed"``) is a no-op and returns the input
        dataset unchanged.

        Both ``add_gaussian_noise`` and ``add_outliers`` deep-copy their inputs
        (verified in ``src/zreg/generators/corruption.py``), so the input
        ``dataset`` is never mutated.

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Input trajectory to augment.

        Returns
        -------
        dict[int, zRegPointCloud]
            Augmented trajectory.  Equals ``dataset`` by identity when
            ``augmentation_params`` is empty (or only contains
            ``"augment_seed"``) — the no-op path.
        """
        params = self.config.augmentation_params
        augment_seed = params.get("augment_seed", 42)
        result = dataset
        # Step 1: Gaussian noise
        if "sigma" in params:
            result = add_gaussian_noise(result, sigma=params["sigma"], seed=augment_seed)
        # Step 2: outlier injection
        if "n_outliers" in params:
            result = add_outliers(
                result,
                n_outliers=params["n_outliers"],
                scale=params.get("scale", 3.0),
                seed=augment_seed,
            )
        # Step 3: uniform scaling
        if "scale_factor" in params:
            result = self.scale(result, params["scale_factor"])
        # Step 4: rotation via Rodrigues' rotation formula
        if "rotation_deg" in params:
            axis = torch.tensor(
                params.get("rotation_axis", [0.0, 0.0, 1.0]),
                dtype=torch.float32,
            )
            angle_rad = math.radians(params["rotation_deg"])
            axis = axis / (axis.norm() + 1e-8)
            # Rodrigues' formula: R = I + sin(θ)·K + (1−cos(θ))·K²
            K = torch.zeros(3, 3)
            K[0, 1] = -axis[2]; K[0, 2] = axis[1]   # noqa: E702
            K[1, 0] = axis[2];  K[1, 2] = -axis[0]   # noqa: E702
            K[2, 0] = -axis[1]; K[2, 1] = axis[0]    # noqa: E702
            R = torch.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)
            result = self.rotate(result, R)
        # Step 5: point dropout
        if "dropout_fraction" in params:
            result = self.drop_points(result, params["dropout_fraction"], seed=augment_seed)
        # Step 6: new point sampling
        if "n_new_points" in params:
            result = self.sample_new_points(result, params["n_new_points"], seed=augment_seed)
        return result

    def prepare_split(
        self,
        dataset: dict[int, zRegPointCloud],
        seed: int = 42,
    ) -> tuple[dict[int, zRegPointCloud], dict[int, zRegPointCloud]]:
        """Split a dataset into train and validation subsets.

        Uses ``random.sample`` to choose which frame indices go into the val
        set (size = ``int(len(keys) * config.val_split)``).  Both returned
        dicts have keys in ascending sorted order — temporal ordering is
        preserved within each subset (D-05).

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Input dataset.  Keys are integer frame indices.

        Returns
        -------
        tuple[dict[int, zRegPointCloud], dict[int, zRegPointCloud]]
            ``(train, val)`` — two disjoint dicts whose key union equals the
            input key set.  Both dicts have sorted keys.

        Notes
        -----
        - ``val_count`` uses ``int(...)`` floor truncation (Pitfall 8): a
          4-frame dataset with the default ``val_split=0.2`` yields
          ``int(4 * 0.2) = 0``, producing an empty val set.  The guard
          ``if n <= 1 or val_count == 0`` handles both the documented D-07
          single-frame case and this silent-empty-val edge case.
        - Single-frame datasets (``n <= 1``) return ``(dataset, {})`` silently
          per D-07 — no exception raised.
        - Both returned dicts preserve sorted-key order per D-05.
        """
        keys = sorted(dataset.keys())  # Pitfall 7: random.sample requires a sequence
        n = len(keys)
        val_count = int(n * self.config.val_split)  # Pitfall 8: floor truncation
        if n <= 1 or val_count == 0:  # D-07 single-frame guard + zero-count guard
            return dataset, {}
        rng = random.Random(seed)  # WR-04: seeded RNG for reproducible splits
        val_keys = set(rng.sample(keys, k=val_count))
        train_keys = [k for k in keys if k not in val_keys]  # sorted by construction
        val_keys_sorted = sorted(val_keys)
        train = {k: dataset[k] for k in train_keys}
        val = {k: dataset[k] for k in val_keys_sorted}
        return train, val

    def get_ground_truth(
        self,
        dataset: dict[int, zRegPointCloud],
    ) -> dict[int, torch.Tensor]:
        """Extract cell-identity labels from a loaded dataset.

        Returns one id tensor per frame, keyed by frame index.  Mirrors the
        structure of the dataset itself (D-11).

        By default (``config.ground_truth_path is None``) extracts ``pc["id"]``
        from each frame in the supplied ``dataset`` (D-10).  If
        ``config.ground_truth_path`` is explicitly set, the external file is
        loaded using the same loader as ``config.data_format`` and its ``id``
        fields are returned instead.

        This method is intended for **real data** (where ``pc["id"]`` is the
        canonical cell id from upstream tracking/annotation).  Synthetic data
        flows should use ``pc["label"]`` from ``generate_labels`` instead.

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            In-memory dataset.  Ignored when ``config.ground_truth_path`` is
            set (the external file is loaded and its ``id`` fields are used).

        Returns
        -------
        dict[int, torch.Tensor]
            One 1-D id tensor per frame, keyed by integer frame index.
        """
        if self.config.ground_truth_path is not None:
            if self.config.data_format == "tracklets":
                gt_ds, _ = load_data_from_tracklets(self.config.ground_truth_path, device=self.config.device)
            else:
                gt_ds = load_shah_from_csv(self.config.ground_truth_path, device=self.config.device)
            return {i: pc["id"] for i, pc in gt_ds.items()}
        return {i: pc["id"] for i, pc in dataset.items()}

    def get_synthetic_ground_truth(self) -> dict[int, torch.Tensor]:
        """Return per-frame cell-identity labels for synthetic mode (D-04, D-05, D-06).

        For rigid, affine, and noise transforms the correspondence between source
        and target is identity: ``source[k][i]`` maps to ``target[k][i]``.
        This method encodes that identity correspondence as per-frame label
        tensors derived from ``self._source_dataset`` (set by
        :meth:`generate_target`).

        This method is for **synthetic mode only**.  For real-data paired mode,
        use :meth:`get_ground_truth` (which accepts an explicit dataset argument).

        Parameters
        ----------
        None

        Returns
        -------
        dict[int, torch.Tensor]
            Per-frame 1-D tensors of dtype ``torch.long``, keyed by integer
            frame index.  Values are either:

            - ``pc["id"].to(torch.long)`` when ``pc["id"]`` is not ``None``
              (D-04); or
            - ``torch.arange(n_points, dtype=torch.long)`` as an ordinal
              fallback when ``pc["id"] is None`` (D-05).

            Dtype is always ``torch.long`` regardless of the source
            ``pc["id"]`` dtype (D-06), consistent with ``compute_f1``
            expectations.

        Raises
        ------
        RuntimeError
            If called before :meth:`generate_target` — ``_synthetic_target``
            is ``None`` and there is no stored source dataset to derive labels
            from.  Call :meth:`generate_target` first.

        Notes
        -----
        - D-04: identity correspondence for rigid/affine/noise transforms.
        - D-05: ordinal fallback when source frames have no cell-id labels.
        - D-06: ``torch.long`` dtype guarantee — consistent with
          ``compute_f1`` label expectations.
        - Guard checks ``self._synthetic_target is None`` (D-03 contract).
        """
        if self._synthetic_target is None:
            raise RuntimeError(
                "DataFactory.get_synthetic_ground_truth: generate_target() must be "
                "called first to populate _synthetic_target and _source_dataset."
            )
        result: dict[int, torch.Tensor] = {}
        for k, pc in self._source_dataset.items():
            if pc["id"] is not None:
                result[k] = pc["id"].to(torch.long)  # D-04 + D-06: id cast to torch.long
            else:
                result[k] = torch.arange(pc["pos"].shape[0], dtype=torch.long)  # D-05 + D-06: ordinal fallback
        return result

    def scale(
        self,
        dataset: dict[int, zRegPointCloud],
        factor: float,
    ) -> dict[int, zRegPointCloud]:
        """Scale every frame's pos by ``factor`` and return a new dataset.

        Immutability contract: the input ``dataset`` is never mutated.
        ``pos * factor`` creates a new tensor, so color and id shared
        references are acceptable — they are passed through as-is.
        ``fps-idx`` is preserved per-frame (passed through by reference;
        it is never indexed or modified).

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Input trajectory.  Not modified.
        factor : float
            Scalar multiplier applied to each frame's ``pos`` field.

        Returns
        -------
        dict[int, zRegPointCloud]
            New trajectory with scaled ``pos``; all other fields preserved.
        """
        result: dict[int, zRegPointCloud] = {}
        for i, pc in dataset.items():
            result[i] = zRegPointCloud(
                pos=pc["pos"] * factor,
                label=pc["label"],
                id=pc["id"],
            )
            result[i]["fps-idx"] = pc["fps-idx"]
        return result

    def rotate(
        self,
        dataset: dict[int, zRegPointCloud],
        rotation_matrix: torch.Tensor,
    ) -> dict[int, zRegPointCloud]:
        """Rotate every frame's pos by ``rotation_matrix`` and return a new dataset.

        Delegates to ``apply_rigid`` which calls ``copy.deepcopy`` internally
        via ``_apply_matrix``.  The deep-copy contract is therefore guaranteed
        by the underlying function — no additional copying is required here.

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Input trajectory.  Not modified.
        rotation_matrix : torch.Tensor
            3x3 rotation matrix (float32).

        Returns
        -------
        dict[int, zRegPointCloud]
            Deep-copied trajectory with rotated ``pos``; all other fields preserved.
        """
        tf = RigidTransformation(
            rot=rotation_matrix,
            t=torch.zeros(3, dtype=rotation_matrix.dtype),
            scale=1.0,
        )
        return apply_rigid(dataset, tf)

    def _standardize(
        self,
        dataset: dict[int, zRegPointCloud],
        stats: dict | None = None,
    ) -> dict[int, zRegPointCloud]:
        """Apply per-trajectory scaling to the ``pos`` field of every frame.

        When ``self.config.data_preprocessing`` is ``None``, returns ``dataset``
        unchanged by reference (no-op early return).

        Statistics are computed globally across all frames (concatenated pos
        tensors) when ``stats`` is ``None``; in that case the computed stats are
        stored in ``self._preprocessing_stats`` for reuse by
        ``load_target()`` in paired mode.  When ``stats`` is provided (not
        ``None``), it is used directly without updating ``self._preprocessing_stats``
        — the target reuses source statistics by reference (D-03).

        Only the ``pos`` field is scaled; ``label``, ``id``, and ``fps-idx``
        are passed through unchanged.

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Input trajectory.  Not modified.
        stats : dict or None
            Pre-computed statistics dict with keys ``"mean"``, ``"std"``,
            ``"median"``, ``"iqr"``, ``"min"``, ``"max"`` (each a Tensor of
            shape ``[3]``).  When ``None``, statistics are computed from
            ``dataset`` and stored in ``self._preprocessing_stats``.

        Returns
        -------
        dict[int, zRegPointCloud]
            New trajectory with scaled ``pos``; all other fields preserved.
            Returns the input ``dataset`` by reference when
            ``config.data_preprocessing`` is ``None``.
        """
        if self.config.data_preprocessing is None:
            return dataset

        cfg = self.config.data_preprocessing
        eps = 1e-8

        if stats is None:
            if not dataset:
                return dataset
            # Compute statistics globally across all frames (concatenated).
            # Cast to float32 first so all stat tensors share a consistent dtype
            # (torch.quantile requires float and would otherwise produce mixed
            # float32/float64 outputs when pos tensors are float64).
            all_pos = torch.cat([pc["pos"] for pc in dataset.values()], dim=0).float()
            mean = all_pos.mean(dim=0)
            std = all_pos.std(dim=0)
            std = torch.where(torch.isnan(std), torch.zeros_like(std), std)
            median = torch.quantile(all_pos, 0.5, dim=0)
            q25 = torch.quantile(all_pos, 0.25, dim=0)
            q75 = torch.quantile(all_pos, 0.75, dim=0)
            iqr = q75 - q25
            min_vals = all_pos.min(dim=0).values
            max_vals = all_pos.max(dim=0).values
            stats = {
                "mean": mean,
                "std": std,
                "median": median,
                "iqr": iqr,
                "min": min_vals,
                "max": max_vals,
            }
            self._preprocessing_stats = stats

        logging.debug(
            "DataFactory._standardize: method=%s mean=%s", cfg.method, stats["mean"]
        )

        result: dict[int, zRegPointCloud] = {}
        for i, pc in dataset.items():
            pos = pc["pos"]
            if cfg.method == "standardize":
                scaled_pos = (pos - stats["mean"]) / (stats["std"] + eps)
            elif cfg.method == "normalize":
                scaled_pos = (pos - stats["min"]) / (stats["max"] - stats["min"] + eps)
            else:  # robust
                scaled_pos = (pos - stats["median"]) / (stats["iqr"] + eps)
                scaled_pos = scaled_pos.clamp(
                    -cfg.robust_outlier_threshold, cfg.robust_outlier_threshold
                )
            result[i] = zRegPointCloud(
                pos=scaled_pos,
                label=pc["label"],
                id=pc["id"],
            )
            result[i]["fps-idx"] = pc["fps-idx"]
        return result

    def _subsample_to_max(
        self,
        dataset: dict[int, zRegPointCloud],
        seed: int = 42,
    ) -> dict[int, zRegPointCloud]:
        """Subsample each frame to at most ``config.max_points_per_frame`` points.

        No-op when ``config.max_points_per_frame`` is ``None`` or when a frame
        already has fewer points than the limit.  Uses ``torch.randperm`` for
        reproducible random selection (same contract as ``drop_points``).

        Returns the input dict unchanged by reference when no subsampling is
        required — the caller must not mutate the result in place.
        """
        limit = self.config.max_points_per_frame
        if limit is None:
            return dataset
        torch.manual_seed(seed)
        result: dict[int, zRegPointCloud] = {}
        for i, pc in dataset.items():
            n = pc["pos"].shape[0]
            if n <= limit:
                result[i] = pc
                continue
            idx = torch.randperm(n, device=pc["pos"].device)[:limit].sort().values
            result[i] = zRegPointCloud(
                pos=pc["pos"][idx],
                label=pc["label"][idx] if pc["label"] is not None else None,
                id=pc["id"][idx] if pc["id"] is not None else None,
            )
            result[i]["fps-idx"] = pc["fps-idx"][idx] if pc["fps-idx"] is not None else None
        return result

    def drop_points(
        self,
        dataset: dict[int, zRegPointCloud],
        fraction: float,
        seed: int = 42,
    ) -> dict[int, zRegPointCloud]:
        """Remove a random fraction of points from every frame.

        Uses ``torch.randperm`` (device-safe) seeded once before the loop so
        different frames receive different random subsets while the overall
        result is reproducible given the same ``seed``.

        Immutability contract: the input ``dataset`` is never mutated.
        All fields (``id``, ``color``, ``fps-idx``) are indexed with the
        same permutation-derived index as ``pos``.

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Input trajectory.  Not modified.
        fraction : float
            Fraction of points to drop.  ``0.0`` is a no-op; ``0.3`` drops
            30 % of points (keeping ``round(n * 0.7)``).
        seed : int, optional
            RNG seed for reproducibility (default 42).  Seeded once before
            the loop — NOT inside the loop.

        Returns
        -------
        dict[int, zRegPointCloud]
            New trajectory with fewer points per frame.
        """
        torch.manual_seed(seed)
        result: dict[int, zRegPointCloud] = {}
        for i, pc in dataset.items():
            n = pc["pos"].shape[0]
            keep = max(1, round(n * (1.0 - fraction)))
            idx = torch.randperm(n, device=pc["pos"].device)[:keep].sort().values
            result[i] = zRegPointCloud(
                pos=pc["pos"][idx],
                label=pc["label"][idx] if pc["label"] is not None else None,
                id=pc["id"][idx] if pc["id"] is not None else None,
            )
            result[i]["fps-idx"] = pc["fps-idx"][idx] if pc["fps-idx"] is not None else None
        return result

    def sample_new_points(
        self,
        dataset: dict[int, zRegPointCloud],
        n_extra: int,
        seed: int = 42,
    ) -> dict[int, zRegPointCloud]:
        """Append ``n_extra`` uniform-in-bbox points to every frame.

        Per-frame bounding box (not dataset-global) is used so each frame's
        new points are consistent with its own point cloud extent.
        ``torch.manual_seed`` is called once before the loop — seeding inside
        the loop would make all frames receive identical points.

        Sentinel fill convention (mirrors ``add_outliers`` in
        ``src/zreg/generators/corruption.py``):
        - 1-D fields (``id``, 1-D ``color``, ``fps-idx``): sentinel -1
        - 2-D fields (RGB ``color``): zero rows

        Immutability contract: the input ``dataset`` is never mutated.

        Parameters
        ----------
        dataset : dict[int, zRegPointCloud]
            Input trajectory.  Not modified.
        n_extra : int
            Number of new uniform-random points to append per frame.
        seed : int, optional
            RNG seed for reproducibility (default 42).  Seeded once before
            the loop.

        Returns
        -------
        dict[int, zRegPointCloud]
            New trajectory with ``n_extra`` additional points per frame.
        """
        torch.manual_seed(seed)
        result: dict[int, zRegPointCloud] = {}
        for i, pc in dataset.items():
            pos = pc["pos"]
            bbox_min = pos.min(dim=0).values
            bbox_max = pos.max(dim=0).values
            rand = torch.rand(n_extra, 3, dtype=pos.dtype, device=pos.device)
            new_pts = bbox_min + rand * (bbox_max - bbox_min)
            new_pos = torch.cat([pos, new_pts], dim=0)

            def _extend(t, fill=-1):
                if t is None:
                    return None
                if t.dim() == 1:
                    return torch.cat([t, torch.full((n_extra,), fill, dtype=t.dtype, device=t.device)])
                else:  # 2-D (RGB color)
                    return torch.cat([t, torch.zeros((n_extra, t.shape[1]), dtype=t.dtype, device=t.device)])

            result[i] = zRegPointCloud(
                pos=new_pos,
                label=_extend(pc["label"]),
                id=_extend(pc["id"]),
            )
            result[i]["fps-idx"] = _extend(pc["fps-idx"])
        return result
