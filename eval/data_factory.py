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
import math
import random
from pathlib import Path  # noqa: F401  (available for future use)

# zreg.dataset MUST precede import torch (libomp SIGABRT lesson from Phase 12;
# enforced in tests/conftest.py:20-24)
from zreg.dataset import (
    load_data_from_tracklets,
    load_shah_from_csv,
    zRegPointCloud,
)
from zreg.generators import (
    add_gaussian_noise,
    add_outliers,
    apply_affine,  # noqa: F401  (available; not used in Phase 17 minimal generate_synthetic)
    apply_rigid,
    generate_labels,  # noqa: F401  (available; not used in Phase 17 minimal generate_synthetic)
    generate_trajectory,
)
from zreg.transforms import RigidTransformation

# torch AFTER zreg.* imports
import torch

# local sibling module last
from eval.config import EvalConfig

__all__ = ["DataFactory"]


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
        self._real_dataset: dict[int, zRegPointCloud] | None = None
        self._synthetic_dataset: dict[int, zRegPointCloud] | None = None

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
            dataset, _ = load_data_from_tracklets(self.config.data_path, device="cpu")
        elif self.config.data_format == "csv":
            # Pitfall 5: device has NO default in load_shah_from_csv
            dataset = load_shah_from_csv(self.config.data_path, device="cpu")
        else:
            raise ValueError(
                f"DataFactory: unknown data_format {self.config.data_format!r}; "
                f"expected 'tracklets' or 'csv'"
            )

        self._real_dataset = dataset
        return dataset

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

        - ``"sigma"`` (float): applies ``add_gaussian_noise(dataset, sigma=..., seed=42)``
        - ``"n_outliers"`` (int): applies ``add_outliers(dataset, n_outliers=..., seed=42)``
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

        Dispatch order: sigma → n_outliers → scale_factor → rotation_deg →
        dropout_fraction → n_new_points.

        Missing keys skip the corresponding step.  An empty dict is a no-op
        and returns the input dataset unchanged.

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
            ``augmentation_params`` is empty (no-op path).
        """
        params = self.config.augmentation_params
        result = dataset
        # Step 1: Gaussian noise
        if "sigma" in params:
            result = add_gaussian_noise(result, sigma=params["sigma"], seed=42)
        # Step 2: outlier injection
        if "n_outliers" in params:
            result = add_outliers(
                result,
                n_outliers=params["n_outliers"],
                scale=params.get("scale", 3.0),
                seed=42,
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
            result = self.drop_points(result, params["dropout_fraction"])
        # Step 6: new point sampling
        if "n_new_points" in params:
            result = self.sample_new_points(result, params["n_new_points"])
        return result

    def prepare_split(
        self,
        dataset: dict[int, zRegPointCloud],
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
        val_keys = set(random.sample(keys, k=val_count))
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
        flows should use ``pc["color"]`` from ``generate_labels`` instead.

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
                gt_ds, _ = load_data_from_tracklets(self.config.ground_truth_path, device="cpu")
            else:
                gt_ds = load_shah_from_csv(self.config.ground_truth_path, device="cpu")
            return {i: pc["id"] for i, pc in gt_ds.items()}
        return {i: pc["id"] for i, pc in dataset.items()}

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
                color=pc["color"],
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
                color=pc["color"][idx] if pc["color"] is not None else None,
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
                color=_extend(pc["color"]),
                id=_extend(pc["id"]),
            )
            result[i]["fps-idx"] = _extend(pc["fps-idx"])
        return result
