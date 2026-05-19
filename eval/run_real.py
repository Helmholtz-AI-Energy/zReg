"""Real-data scale/density sweep orchestrator for the zReg evaluation framework.

This module is a *script*, not a package member. It must remain standalone so
that ``python eval/run_real.py`` can be invoked directly without an
``eval/__init__.py`` being present (``eval/`` is a namespace directory).

The sweep iterates over scale and density fractions, logging every run to
output_dir via ``eval.tracking.log_run``. When DATASET_PATH does not exist
the script exits gracefully with an informative "not found" message.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from eval.tracking import log_run
from zreg.dataset import load_data_from_tracklets

# --- Module-level sweep constants ---
DATASET_PATH = "data/raw/example.mat"
SCALES = [1.0, 0.5, 0.25]
DENSITY_FRACTIONS = [1.0, 0.75, 0.5]
SEED = 42


def run_sweep(output_dir: str = "evaluation/runs") -> None:
    """Run the real-data scale/density sweep, logging every cell.

    Parameters
    ----------
    output_dir : str, optional
        Directory passed through to ``log_run`` for each cell. Tests redirect
        this to ``tmp_path`` to avoid polluting the real ``evaluation/runs/``.
    """
    if not Path(DATASET_PATH).exists():
        print(f"Dataset not found: {DATASET_PATH}. Skipping real sweep.")
        return

    trajectory, _ = load_data_from_tracklets(DATASET_PATH, device="cpu")
    n_points_base = int(next(iter(trajectory.values()))["pos"].shape[0])

    for scale in SCALES:
        n_points_scaled = int(n_points_base * scale)
        for density in DENSITY_FRACTIONS:
            run_id = f"real_scale{scale}_density{density}_seed{SEED}"
            n_after = int(n_points_scaled * density)
            log_run(
                run_id=run_id,
                dataset_path=DATASET_PATH,
                frame_indices=list(trajectory.keys()),
                seed=SEED,
                n_points_before=n_points_scaled,
                n_points_after=n_after,
                output_dir=output_dir,
            )
            print(f"{run_id}: n_before={n_points_scaled}, n_after={n_after}")


if __name__ == "__main__":
    run_sweep()
