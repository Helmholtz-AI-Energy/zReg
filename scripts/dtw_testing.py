from zreg.core.dataset import load_data_from_tracklets, load_shah_from_csv
from zreg.algorithms.pairwise_distance_matrix import create_pairwise_distance_matrix
import torch
from pathlib import Path
import copy
import time
import os
# from mpi4py import MPI

import numpy as np

# want to load all 4 tracklet point clouds
# then will run them in a matrix to

FILES = [
    "12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets",
    "12_11_27_embryo_ew_08_Cleaned_BackTracked_Oriented.tracklets",
    "12_11_30_embryo_ew_11_Cleaned_BackTracked_Oriented.tracklets",
    "12_12_04_embryo_ew_12_Cleaned_BackTracked_Oriented.tracklets",
]
DEFAULT_DATA_DIR = Path("/p/project/tissuetwin/ZebraTwin/data")
DEFAULT_SHAH_CSV = Path(
    "/p/project/tissuetwin/zebra/ftp.ebi.ac.uk/pub/databases/IDR/idr0068-shah-zebrafishlightsheet/"
    "20191001-ftp/sample-1/sample-1-cell-tracks.csv"
)
DEFAULT_OUT_DIR = Path("/p/project/tissuetwin/coquelin1/zReg/figures") / "down-before-cpd"

DEFAULT_DISTANCE_METRICS = [
    "swd",
    # "aswd",
    # "oswd",
    # "gswd",
    # "pswd",
    "euclidean",
    # "manhattan",
    # "minkowski",
    "cpd",
]
# TODO: set up furthest sampling to be faster somehow?
DEFAULT_DOWNSAMPLING_METRICS = [
    "random",
    "farthest",
]  #  "uniform", ]  #  None,
DEFAULT_PAIRS = [[1, 5], [2, 5], [3, 5], [4, 5]]


def _proc_run(pcs, out_dir, rank, device, distance, downsampling, i, j, cpd_type):
    t0 = time.perf_counter()
    print(f"Working on distances: {distance}, downsampling: {downsampling}, combi: {i, j} on {device}")
    testx = copy.deepcopy(pcs[i])
    testy = copy.deepcopy(pcs[j])

    for k in testx:
        testx[k] = testx[k].to(device)
    for m in testy:
        testy[m] = testy[m].to(device)

    # for testing with a fixed rotation uncomment the following:
    # mat = zreg.algorithms.pairwise_distance_matrix.create_pairwise_distance_matrix_given_rigid_rot(
    #     x=testx, y=testy,
    #     rotation=torch.tensor([[-0.5953, -0.7996,  0.0796],
    #                            [ 0.7202, -0.4870,  0.4941],
    #                            [-0.3563,  0.3515,  0.8657]]),
    #     translation=torch.tensor([-0.1428, -0.0739,  0.0719]),
    #     scale=0.9348,
    #     window=None,
    #     normalize=norm,
    #     distance_metric=distance,
    #     distance_kwargs=None,
    #     downsample_method=downsampling,
    #     mpi_distribute=True,
    # )
    # rots = None
    # ------------------------------------------------------------
    # Otherwise, use this for the general CPD fitting.
    # create_pairwise_distance_matrix returns a PairwiseResult dataclass (it
    # used to return a (matrix, rotations) tuple).
    result = create_pairwise_distance_matrix(
        x=testx,
        y=testy,
        window=None,
        distance_metric=distance,
        distance_kwargs=None,
        normalize=True,
        downsample_method=downsampling,
        cpd_type=cpd_type,
        mpi_distribute=True,
    )
    mat, rots = result.cost_matrix, result.rotations
    print(f"finished iteration, time required: {time.perf_counter() - t0}")
    for c, dist in enumerate(distance):
        out_file = out_dir / f"{dist}_{downsampling}_{i}_{j}.npy"
        np.save(out_file, mat[c].cpu().numpy())
    if rots is not None and rots.numel() > 0:
        rots_out = out_dir / f"rots_{downsampling}_{i}_{j}"
        rots_out.mkdir(parents=True, exist_ok=True)
        out_file = rots_out / f"rank_{rank}.npy"
        np.save(out_file, rots.cpu().numpy())
    del testx, testy


def main(
    data_dir=DEFAULT_DATA_DIR,
    shah_csv=DEFAULT_SHAH_CSV,
    out_dir=DEFAULT_OUT_DIR,
    device: str = "cuda:0",
    distance_metrics=None,
    downsampling_metrics=None,
    pairs=None,
) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    if distance_metrics is None:
        distance_metrics = list(DEFAULT_DISTANCE_METRICS)
    if downsampling_metrics is None:
        downsampling_metrics = list(DEFAULT_DOWNSAMPLING_METRICS)
    if pairs is None:
        pairs = [list(p) for p in DEFAULT_PAIRS]

    pcs = {}
    for idx, fname in enumerate(FILES, start=1):
        pcs[idx], _alltrackles = load_data_from_tracklets(
            filepath=str(data_dir / fname),
            device=device,
        )
    pcs[5] = load_shah_from_csv(
        filepath=str(shah_csv),
        device=device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    print(f"rank: {rank}, TASKS: {tasks}")
    print(f"device count: {torch.cuda.device_count()}")

    for downsampling in downsampling_metrics:
        for i, j in pairs:
            _proc_run(
                pcs=pcs,
                out_dir=out_dir,
                rank=rank,
                device=device,
                distance=distance_metrics,
                downsampling=downsampling,
                i=i,
                j=j,
                cpd_type="rigid",
            )


if __name__ == "__main__":
    main()
