#!/usr/bin/env python3
"""Generate fully synthetic and semi-synthetic point cloud datasets.

Fully synthetic (data/synthetic/fully_synthetic/):
  {ball,bowl}_{small,realistic_kobitski,realistic_shah}/
  Growing ball: sphere expanding by fixed Δr/frame; existing points scaled,
  new points added in the outer shell to keep density (n ∝ R³) constant.
  Growing bowl: lower hemisphere of spherical shell — outer sphere R, carving
  sphere √(R²+d²) centred at (0,0,d) with d = 0.5R.  Same growth logic.

Semi-synthetic (data/synthetic/semi_synthetic/):
  {source}_{aug_type}_{param}{value}/
  Augmentations applied independently (one type at a time):
    noise   sigma  ∈ {1.0, 5.0, 10.0}
    scaling factor ∈ {0.8, 1.2, 1.5}
    dropout frac   ∈ {0.1, 0.2, 0.3}

Output format: CSV with columns x,y,z,t,layer,id  (t is 1-indexed).
Compatible with load_shah_from_csv.

Run:
    python scripts/generate_datasets.py
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# zreg imports must precede torch (libomp SIGABRT workaround — see data_factory.py)
from zreg.core.dataset import load_data_from_tracklets, load_shah_from_csv, zRegPointCloud
from eval.data_factory import DataFactory
from eval.config import EvalConfig

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

FULLY_DIR = ROOT / "data" / "synthetic" / "fully_synthetic"
SEMI_DIR  = ROOT / "data" / "synthetic" / "semi_synthetic"

# Radius grows from R_0 to R_0 * GROWTH_FACTOR over the trajectory.
# GROWTH_FACTOR = 1.47 ≈ (3.17)^(1/3) matches the empirical kobitski point-
# count growth ratio of ~3.17× over 370 frames (n ∝ R³).
R_0 = 1.0
GROWTH_FACTOR = 1.47
BOWL_D_RATIO  = 0.5   # d = 0.5 * R at every frame; d ∝ R → bowl volume ∝ R³

SIZE_VARIANTS: dict[str, dict] = {
    "small":               {"n_frames": 20,  "n_base": 100},
    "realistic_kobitski":  {"n_frames": 370, "n_base": 15_000},
    "realistic_shah":      {"n_frames": 420, "n_base":  4_000},
}

REAL_SOURCES: dict[str, dict] = {
    "kobitski_ew06": {
        "path": ROOT / "data/external/sample/kobitski_data"
                     / "12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets",
        "fmt": "tracklets",
    },
    "kobitski_ew08": {
        "path": ROOT / "data/external/sample/kobitski_data"
                     / "12_11_27_embryo_ew_08_Cleaned_BackTracked_Oriented.tracklets",
        "fmt": "tracklets",
    },
    "kobitski_ew11": {
        "path": ROOT / "data/external/sample/kobitski_data"
                     / "12_11_30_embryo_ew_11_Cleaned_BackTracked_Oriented.tracklets",
        "fmt": "tracklets",
    },
    "kobitski_ew12": {
        "path": ROOT / "data/external/sample/kobitski_data"
                     / "12_12_04_embryo_ew_12_Cleaned_BackTracked_Oriented.tracklets",
        "fmt": "tracklets",
    },
    "shah_sample1": {
        "path": ROOT / "data/external/sample/shah_data/sample-1/sample-1-cell-tracks.csv",
        "fmt": "csv",
    },
}

# One augmentation type at a time; (param_name, value) pairs per type.
AUGMENTATION_GRID: dict[str, list[tuple[str, float]]] = {
    "noise":   [("sigma",    1.0), ("sigma",    5.0), ("sigma",   10.0)],
    "scaling": [("factor",   0.8), ("factor",   1.2), ("factor",   1.5)],
    "dropout": [("fraction", 0.1), ("fraction", 0.2), ("fraction", 0.3)],
}

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def in_bowl(pts: np.ndarray, R: float, d: float) -> np.ndarray:
    """Boolean mask: True for points inside bowl(R, d).

    Bowl = { p : ‖p‖ ≤ R  AND  ‖p−(0,0,d)‖² > R²+d²  AND  z ≤ 0 }

    The carving sphere (radius √(R²+d²), centred at (0,0,d)) intersects the
    outer sphere exactly at z = 0, so the bowl opening sits at the equatorial
    plane with zero wall thickness there.
    """
    norm_sq  = (pts ** 2).sum(axis=1)
    carve_sq = pts[:, 0] ** 2 + pts[:, 1] ** 2 + (pts[:, 2] - d) ** 2
    return (norm_sq <= R ** 2) & (carve_sq > R ** 2 + d ** 2) & (pts[:, 2] <= 0.0)


def _delta_r(n_frames: int) -> float:
    """Per-frame radius increment so R grows from R_0 to R_0*GROWTH_FACTOR."""
    return (GROWTH_FACTOR - 1.0) * R_0 / max(n_frames - 1, 1)


# ---------------------------------------------------------------------------
# Point samplers
# ---------------------------------------------------------------------------

def sample_ball_shell(
    n: int,
    R_inner: float,
    R_outer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform sample in spherical shell [R_inner, R_outer] (volume-correct)."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    r     = rng.uniform(R_inner ** 3, R_outer ** 3, n) ** (1.0 / 3)
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi   = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, None))
    return np.stack(
        [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t],
        axis=1,
    ).astype(np.float32)


def sample_bowl_frame(
    n: int,
    R: float,
    d: float,
    rng: np.random.Generator,
    batch_mult: int = 10,
) -> np.ndarray:
    """Uniform sample inside bowl(R, d) via rejection sampling."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    collected: list[np.ndarray] = []
    total = 0
    batch = max(n * batch_mult, 2_000)
    while total < n:
        cands = rng.uniform(-R, R, (batch, 3))
        keep  = cands[in_bowl(cands, R, d)]
        collected.append(keep)
        total += len(keep)
    return np.concatenate(collected, axis=0)[:n].astype(np.float32)


def sample_bowl_shell(
    n: int,
    R_inner: float,
    R_outer: float,
    d_outer: float,
    rng: np.random.Generator,
    batch_mult: int = 20,
) -> np.ndarray:
    """Uniform sample in spherical shell [R_inner, R_outer] ∩ bowl(R_outer, d_outer)."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    collected: list[np.ndarray] = []
    total = 0
    batch = max(n * batch_mult, 2_000)
    while total < n:
        cands = sample_ball_shell(batch, R_inner, R_outer, rng)
        keep  = cands[in_bowl(cands, R_outer, d_outer)]
        collected.append(keep)
        total += len(keep)
    return np.concatenate(collected, axis=0)[:n].astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory builders  (return dict[int, {"pos", "id", "color"}])
# ---------------------------------------------------------------------------

def _make_trajectory(
    shape: str,
    n_frames: int,
    n_base: int,
    rng: np.random.Generator,
) -> dict[int, dict]:
    """Build a growing-{ball,bowl} trajectory frame by frame.

    Existing points are scaled outward each frame; new points are born in the
    expanded outer shell to keep point density (points/volume) constant.
    """
    dr = _delta_r(n_frames)

    # --- seed frame 0 ---
    if shape == "ball":
        pts = sample_ball_shell(n_base, 0.0, R_0, rng)
    else:
        pts = sample_bowl_frame(n_base, R_0, BOWL_D_RATIO * R_0, rng)

    ids     = np.arange(n_base, dtype=np.int32)
    next_id = n_base
    traj: dict[int, dict] = {}

    for i in range(n_frames):
        R_curr = R_0 + i * dr
        d_curr = BOWL_D_RATIO * R_curr

        if i > 0:
            R_prev = R_0 + (i - 1) * dr
            scale  = R_curr / R_prev
            pts    = pts * scale

            n_target = round(n_base * (R_curr / R_0) ** 3)
            n_new    = max(0, n_target - len(pts))
            if n_new:
                if shape == "ball":
                    new_pts = sample_ball_shell(n_new, R_prev, R_curr, rng)
                else:
                    new_pts = sample_bowl_shell(n_new, R_prev, R_curr, d_curr, rng)
                pts = np.vstack([pts, new_pts])
                ids = np.concatenate(
                    [ids, np.arange(next_id, next_id + n_new, dtype=np.int32)]
                )
                next_id += n_new

        traj[i] = {
            "pos":   pts.copy(),
            "id":    ids.copy(),
            "label": np.ones(len(pts), dtype=np.int32),
        }

        if (i + 1) % max(n_frames // 10, 1) == 0 or i == n_frames - 1:
            print(f"    frame {i+1:>4}/{n_frames}  pts={len(pts):>7,}", flush=True)

    return traj


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _frame_to_df(frame_idx: int, frame: dict | zRegPointCloud) -> pd.DataFrame:
    """Convert one frame to a DataFrame row-set (x,y,z,t,layer,id)."""
    pos_raw = frame["pos"]
    if isinstance(pos_raw, np.ndarray):
        # plain numpy dict produced by _make_trajectory
        pos   = pos_raw
        ids   = frame["id"]
        color = frame["label"]
    else:
        # zRegPointCloud: fields are torch Tensors
        pos       = pos_raw.numpy()
        raw_id    = frame["id"]
        raw_color = frame["label"]
        ids = (
            raw_id.numpy()
            if raw_id is not None
            else np.arange(pos.shape[0], dtype=np.int32)
        )
        if raw_color is None:
            color = np.ones(pos.shape[0], dtype=np.int32)
        elif raw_color.dim() == 1:
            color = raw_color.numpy().astype(np.int32)
        else:
            # 2-D RGB (e.g. kobitski): use 1 as layer placeholder
            color = np.ones(pos.shape[0], dtype=np.int32)

    return pd.DataFrame({
        "x":     pos[:, 0].astype(np.float32),
        "y":     pos[:, 1].astype(np.float32),
        "z":     pos[:, 2].astype(np.float32),
        "t":     np.full(len(ids), frame_idx + 1, dtype=np.int32),
        "layer": color,
        "id":    ids.astype(np.int32),
    })


def save_as_csv(
    traj: dict,
    path: Path,
    chunk_frames: int = 50,
) -> None:
    """Write trajectory to CSV, flushing every chunk_frames frames to cap RAM.

    Writes to a sibling .csv.tmp file first, then renames atomically on
    success.  This prevents a partially written file from being mistaken for
    a complete one on re-run if the process is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path   = path.with_suffix(".csv.tmp")
    frames     = sorted(traj.keys())
    total_rows = 0
    first      = True

    try:
        for start in range(0, len(frames), chunk_frames):
            chunk = frames[start : start + chunk_frames]
            df    = pd.concat([_frame_to_df(fi, traj[fi]) for fi in chunk])
            df.to_csv(tmp_path, index=False, mode="w" if first else "a", header=first)
            total_rows += len(df)
            first = False
        tmp_path.rename(path)  # atomic on POSIX; same filesystem guaranteed
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"    → {path.relative_to(ROOT)}  ({total_rows:,} rows)")


# ---------------------------------------------------------------------------
# Top-level generators
# ---------------------------------------------------------------------------

def generate_fully_synthetic(rng: np.random.Generator) -> None:
    print("\n=== Fully synthetic datasets ===")
    for shape in ("ball", "bowl"):
        for size_name, cfg in SIZE_VARIANTS.items():
            name     = f"{shape}_{size_name}"
            out_path = FULLY_DIR / name / f"{name}.csv"
            if out_path.exists():
                print(f"  skip (exists): {name}")
                continue
            print(
                f"\n  {name}  "
                f"({cfg['n_frames']} frames, {cfg['n_base']:,} base pts)"
            )
            traj = _make_trajectory(shape, cfg["n_frames"], cfg["n_base"], rng)
            save_as_csv(traj, out_path)


def _derive_seed(src_name: str, aug_type: str, value: float) -> int:
    """Derive a deterministic but source/variant-specific seed.

    Uses MD5 over the canonical key string so that each (source, aug_type,
    value) triple gets an independent seed, preventing identical noise or
    dropout patterns across sources.
    """
    digest = hashlib.md5(f"{src_name}_{aug_type}_{value}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFF_FFFF  # positive int32


def generate_semi_synthetic() -> None:
    print("\n=== Semi-synthetic datasets ===")
    for src_name, src_cfg in REAL_SOURCES.items():
        # Determine which (aug_type, param_name, value) pairs still need work
        # before paying the I/O cost of loading the source dataset.
        pending: list[tuple[str, str, float, Path]] = []
        for aug_type, param_list in AUGMENTATION_GRID.items():
            for param_name, value in param_list:
                label    = f"{param_name}{value:.1f}"
                name     = f"{src_name}_{aug_type}_{label}"
                out_path = SEMI_DIR / name / f"{name}.csv"
                if not out_path.exists():
                    pending.append((aug_type, param_name, value, out_path))

        if not pending:
            print(f"\n  skip (all exist): {src_name}")
            continue

        print(f"\n  loading {src_name} …")
        if src_cfg["fmt"] == "tracklets":
            dataset, _ = load_data_from_tracklets(str(src_cfg["path"]), device="cpu")
        else:
            dataset = load_shah_from_csv(src_cfg["path"], device="cpu")
        print(f"  loaded {len(dataset)} frames")

        for aug_type, param_name, value, out_path in pending:
            print(f"    {aug_type} {param_name}={value} …", end="  ", flush=True)
            # Note: dropout RNG changed from np.random (pre-Phase 28) to torch.randperm via DataFactory.drop_points; dropout point selection differs but fraction and reproducibility are preserved (D-04/D-05).
            _aug_key = {"noise": "sigma", "scaling": "scale_factor", "dropout": "dropout_fraction"}[aug_type]
            _cfg = EvalConfig(data_path="", augmentation_params={_aug_key: value})
            # Set a source/variant-specific seed so that noise and dropout
            # patterns are independent across sources and aug types (WR-03).
            torch.manual_seed(_derive_seed(src_name, aug_type, value))
            augmented = DataFactory(_cfg).augment(dataset)
            save_as_csv(augmented, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    generate_fully_synthetic(rng)
    generate_semi_synthetic()
    print("\nDone.")
