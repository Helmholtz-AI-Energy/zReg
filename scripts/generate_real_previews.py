"""Generate 1x3 triptych previews for real datasets (Shah + Kobitski).

Outputs PNG files to reports/dataset_previews/real/.
"""

import sys
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from zreg.core.dataset import load_data_from_tracklets  # noqa: E402 — must precede torch
import torch  # noqa: F401, E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.viz import render_dataset_triptych  # noqa: E402

OUT_DIR = _repo / "reports" / "dataset_previews" / "real"

COLOR = "#2a6496"
MAX_PTS = 4_000


def _triptych_from_dict(
    dataset: "dict[int, object]",
    name: str,
    out_dir: Path,
    dpi: int = 150,
) -> Path:
    """Render a 1×3 triptych PNG from a dict[int, zRegPointCloud]."""
    frames = sorted(dataset.keys())
    n = len(frames)
    t_first, t_mid, t_last = frames[0], frames[n // 2], frames[-1]

    fig = plt.figure(figsize=(13, 4.2))
    fig.suptitle(name, fontsize=12, fontweight="bold", y=1.01)

    rng = np.random.default_rng(0)
    for col, (fk, label) in enumerate(
        [(t_first, "first"), (t_mid, "mid"), (t_last, "last")]
    ):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        pts = dataset[fk]["pos"].detach().cpu().numpy()
        n_orig = len(pts)
        if n_orig > MAX_PTS:
            idx = rng.choice(n_orig, MAX_PTS, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=1.5, alpha=0.45, c=COLOR, linewidths=0)
        ax.set_title(f"t = {fk}  ({label})\nn = {n_orig:,}", fontsize=9, pad=4)
        for lbl in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            lbl.set_fontsize(6)
        ax.set_xlabel("x", fontsize=7, labelpad=2)
        ax.set_ylabel("y", fontsize=7, labelpad=2)
        ax.set_zlabel("z", fontsize=7, labelpad=2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    out = out_dir / f"{name}.png"
    try:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out


KOBITSKI_FILES = {
    "kobitski_ew06": "data/external/sample/kobitski_data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets",
    "kobitski_ew08": "data/external/sample/kobitski_data/12_11_27_embryo_ew_08_Cleaned_BackTracked_Oriented.tracklets",
    "kobitski_ew11": "data/external/sample/kobitski_data/12_11_30_embryo_ew_11_Cleaned_BackTracked_Oriented.tracklets",
    "kobitski_ew12": "data/external/sample/kobitski_data/12_12_04_embryo_ew_12_Cleaned_BackTracked_Oriented.tracklets",
}

SHAH_CSV = "data/external/sample/shah_data/sample-1/sample-1-cell-tracks.csv"


def main(out_dir: Path = OUT_DIR, repo: Path = _repo) -> None:
    out_dir = Path(out_dir)
    repo = Path(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing previews to {out_dir}")

    # --- Shah (CSV — render_dataset_triptych handles it natively) ---
    shah_path = repo / SHAH_CSV
    out = render_dataset_triptych(shah_path, name="shah_sample1", output_dir=out_dir)
    print(f"  {out}")

    # --- Kobitski (tracklets) ---
    for name, rel_path in KOBITSKI_FILES.items():
        abs_path = repo / rel_path
        dataset, _ = load_data_from_tracklets(str(abs_path), device="cpu")
        out = _triptych_from_dict(dataset, name, out_dir, dpi=150)
        print(f"  {out}")

    print("Done.")


if __name__ == "__main__":
    main()
