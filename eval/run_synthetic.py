"""Synthetic noise/corruption sweep orchestrator for the zReg evaluation framework.

This module is a *script*, not a package member. It must remain standalone so
that ``python eval/run_synthetic.py`` can be invoked directly without an
``eval/__init__.py`` being present (``eval/`` is a namespace directory).

The sweep iterates over Gaussian noise sigmas and outlier counts, generating a
synthetic trajectory once per (sigma, n_outliers) cell, applying the
corruption stack, computing chamfer/hausdorff against the clean reference,
and persisting every run via ``eval.tracking.log_run``. Generators are
imported from ``zreg.generators``.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from zreg.generators import generate_trajectory, add_gaussian_noise, add_outliers
from eval.tracking import log_run
from zreg.metrics import chamfer, hausdorff

# --- Module-level sweep constants ---
SIGMAS = [0.0, 0.01, 0.05, 0.1]
N_OUTLIERS_LIST = [0, 5, 20]
N_POINTS = 200
N_FRAMES = 5
SEED = 42


def _count_points(trajectory: dict) -> int:
    """Return the number of points in the first frame of *trajectory*."""
    return int(next(iter(trajectory.values()))["pos"].shape[0])


def run_sweep(output_dir: str = "experiments/runs") -> None:
    """Run the synthetic noise + outlier sweep, logging every cell.

    Parameters
    ----------
    output_dir : str, optional
        Directory passed through to ``log_run`` for each cell. Tests redirect
        this to ``tmp_path`` to avoid polluting the real ``experiments/runs/``.
    """
    traj_clean = generate_trajectory(N_POINTS, N_FRAMES, seed=SEED)
    for sigma in SIGMAS:
        for n_out in N_OUTLIERS_LIST:
            traj_noisy = add_gaussian_noise(traj_clean, sigma=sigma, seed=SEED)
            traj_noisy = add_outliers(traj_noisy, n_outliers=n_out, seed=SEED)

            frame_idx = 0
            cd = chamfer(
                traj_clean[frame_idx]["pos"], traj_noisy[frame_idx]["pos"]
            ).item()
            hd = hausdorff(
                traj_clean[frame_idx]["pos"], traj_noisy[frame_idx]["pos"]
            ).item()

            run_id = f"synthetic_sigma{sigma}_out{n_out}_seed{SEED}"
            log_run(
                run_id=run_id,
                dataset_path="synthetic",
                frame_indices=list(range(N_FRAMES)),
                seed=SEED,
                n_points_before=N_POINTS,
                n_points_after=int(traj_noisy[frame_idx]["pos"].shape[0]),
                output_dir=output_dir,
                chamfer=cd,
                hausdorff=hd,
            )
            print(f"{run_id}: chamfer={cd:.4f}, hausdorff={hd:.4f}")


if __name__ == "__main__":
    run_sweep()
