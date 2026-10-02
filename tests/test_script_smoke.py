"""main() smoke tests for the example / HPC scripts (RUN-02).

Import-only tests prove importability, not that the scripts run.  These tests
execute each script's real ``main()`` on tiny synthetic data.  Only I/O
boundaries are stubbed:

* the data-loader attributes on the loaded script module
  (``load_data_from_tracklets`` / ``load_shah_from_csv``) — the real files live
  on the HPC file system;
* ``o3d.visualization.draw_plotly`` — it opens a browser;
* output directories — pointed at ``tmp_path`` via ``main()`` parameters;
* ``SLURM_PROCID`` / ``SLURM_NTASKS`` — via ``monkeypatch.setenv``.

CPD registration, downsampling, transforms, label transfer, pairwise distance
computation and plotting all run for real.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# zreg before torch (libomp ordering on macOS ARM).
from zreg.core.dataset import zRegPointCloud
from zreg.data_generation import generate_labels, generate_trajectory

import torch

from test_script_imports import REPO_ROOT, _load_script

N_FRAMES = 3


def _tiny_trajectory(n_points: int, seed: int, device: str = "cpu") -> dict[int, zRegPointCloud]:
    """Return a dict[int, zRegPointCloud] shaped like load_data_from_tracklets output."""
    traj = generate_labels(generate_trajectory(n_points, N_FRAMES, seed=seed), n_labels=3, seed=seed)
    out = {}
    for t, pc in traj.items():
        out[t] = zRegPointCloud(
            pos=pc["pos"].to(device=device, dtype=torch.float32),
            label=pc["label"].to(device),
            id=torch.arange(n_points, device=device),
        )
    return out


def _tracklets_stub(n_points: int = 40):
    calls: list[str] = []

    def load_data_from_tracklets(filepath, device="cpu"):
        calls.append(str(filepath))
        return _tiny_trajectory(n_points, seed=len(calls), device=device), []

    return load_data_from_tracklets, calls


class _DrawRecorder:
    def __init__(self):
        self.calls: list[list] = []

    def __call__(self, geometries, *args, **kwargs):
        self.calls.append(list(geometries))


def _repo_untouched_snapshot() -> dict[str, bool]:
    paths = [REPO_ROOT / "reports" / "dataset_previews" / "real", REPO_ROOT / "experiments"]
    return {str(p): p.exists() for p in paths}


# ---------------------------------------------------------------------------
# example_plots.py
# ---------------------------------------------------------------------------


def test_example_plots_main_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module = _load_script("example_plots.py")
    loader, calls = _tracklets_stub(n_points=40)
    monkeypatch.setattr(module, "load_data_from_tracklets", loader)
    recorder = _DrawRecorder()
    monkeypatch.setattr(module.o3d.visualization, "draw_plotly", recorder)

    module.main(tracklets_path="unused", device="cpu")

    assert calls == ["unused"]
    assert len(recorder.calls) >= 4
    assert all(len(geoms) == 2 for geoms in recorder.calls)


# ---------------------------------------------------------------------------
# label_transfer_example.py
# ---------------------------------------------------------------------------


def test_label_transfer_example_main_runs(capsys):
    module = _load_script("label_transfer_example.py")

    module.main()

    out = capsys.readouterr().out
    assert "Color transfer demonstration complete!" in out
    assert out.count("Transferred colors shape: torch.Size([4, 3])") == 2


# ---------------------------------------------------------------------------
# dtw_testing.py
# ---------------------------------------------------------------------------


def test_dtw_testing_main_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    module = _load_script("dtw_testing.py")
    loader, calls = _tracklets_stub(n_points=30)
    shah_calls: list[str] = []

    def load_shah_from_csv(filepath, device):
        shah_calls.append(str(filepath))
        return _tiny_trajectory(36, seed=99, device=device)

    monkeypatch.setattr(module, "load_data_from_tracklets", loader)
    monkeypatch.setattr(module, "load_shah_from_csv", load_shah_from_csv)
    before = _repo_untouched_snapshot()

    module.main(
        data_dir=tmp_path,
        shah_csv=tmp_path / "x.csv",
        out_dir=tmp_path / "out",
        device="cpu",
        distance_metrics=["euclidean"],
        downsampling_metrics=["random"],
        pairs=[[1, 5]],
    )

    assert len(calls) == 4 and all(c.startswith(str(tmp_path)) for c in calls)
    assert shah_calls == [str(tmp_path / "x.csv")]
    assert list((tmp_path / "out").rglob("*.npy"))
    assert _repo_untouched_snapshot() == before


# ---------------------------------------------------------------------------
# generate_real_previews.py
# ---------------------------------------------------------------------------


def _write_shah_csv(path: Path, n_points: int = 30) -> None:
    """Write a tiny CSV in the Shah cell-tracks format (x, y, z, t, layer, id)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    traj = _tiny_trajectory(n_points, seed=7)
    lines = ["x,y,z,t,layer,id"]
    for t, pc in traj.items():
        for p, lab, ident in zip(pc["pos"].tolist(), pc["label"].tolist(), pc["id"].tolist()):
            lines.append(f"{p[0]},{p[1]},{p[2]},{t + 1},{lab},{ident}")
    path.write_text("\n".join(lines) + "\n")


def test_generate_real_previews_main_runs(monkeypatch, tmp_path):
    module = _load_script("generate_real_previews.py")
    _write_shah_csv(tmp_path / module.SHAH_CSV)
    loader, calls = _tracklets_stub(n_points=50)
    monkeypatch.setattr(module, "load_data_from_tracklets", loader)
    before = _repo_untouched_snapshot()
    out_dir = tmp_path / "previews"

    module.main(out_dir=out_dir, repo=tmp_path)

    pngs = sorted(p.name for p in out_dir.glob("*.png"))
    assert pngs == sorted(["shah_sample1.png", *(f"{k}.png" for k in module.KOBITSKI_FILES)])
    assert len(pngs) == 5
    assert len(calls) == 4 and all(c.startswith(str(tmp_path)) for c in calls)
    assert _repo_untouched_snapshot() == before
