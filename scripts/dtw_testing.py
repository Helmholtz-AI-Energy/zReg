import zreg
import torch
from pathlib import Path
import copy
import time
import os
# from mpi4py import MPI

import numpy as np

# want to load all 4 tracklet point clouds
# then will run them in a matrix to

files = [
    "12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets",
    "12_11_27_embryo_ew_08_Cleaned_BackTracked_Oriented.tracklets",
    "12_11_30_embryo_ew_11_Cleaned_BackTracked_Oriented.tracklets",
    "12_12_04_embryo_ew_12_Cleaned_BackTracked_Oriented.tracklets",
]
parent_dir = Path("/p/project1/tissuetwin/ZebraTwin/data")

pcs = {}
pcs[1], _alltrackles = zreg.dataset.load_data_from_tracklets(
    filepath="/p/project/tissuetwin/ZebraTwin/data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets",
    device="cuda:0",
)
pcs[2], _alltrackles = zreg.dataset.load_data_from_tracklets(
    filepath="/p/project/tissuetwin/ZebraTwin/data/12_11_27_embryo_ew_08_Cleaned_BackTracked_Oriented.tracklets",
    device="cuda:0",
)
pcs[3], _alltrackles = zreg.dataset.load_data_from_tracklets(
    filepath="/p/project/tissuetwin/ZebraTwin/data/12_11_30_embryo_ew_11_Cleaned_BackTracked_Oriented.tracklets",
    device="cuda:0",
)
pcs[4], _alltrackles = zreg.dataset.load_data_from_tracklets(
    filepath="/p/project/tissuetwin/ZebraTwin/data/12_12_04_embryo_ew_12_Cleaned_BackTracked_Oriented.tracklets",
    device="cuda:0",
)
pcs[5] = zreg.dataset.load_shah_from_csv(
    filepath="/p/project/tissuetwin/zebra/ftp.ebi.ac.uk/pub/databases/IDR/idr0068-shah-zebrafishlightsheet/20191001-ftp/sample-1/sample-1-cell-tracks.csv",
    device="cuda:0",
)

distance_metrics = [
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
downsampling_metrics = [
    "random",
    "farthest",
]  #  "uniform", ]  #  None,
out_dir = Path("/p/project/tissuetwin/coquelin1/zReg/figures") / "down-before-cpd"
out_dir.mkdir(parents=True, exist_ok=True)

rank = int(os.environ["SLURM_PROCID"])
tasks = int(os.environ["SLURM_NTASKS"])
print(f"rank: {rank}, TASKS: {tasks}")
print(f"device count: {torch.cuda.device_count()}")


def _proc_run(gpu_id, distance, downsampling, i, j, cpd_type):
    t0 = time.perf_counter()
    print(f"Working on distances: {distance}, downsampling: {downsampling}, combi: {i, j} on gpu {gpu_id}")
    testx = copy.deepcopy(pcs[i])
    testy = copy.deepcopy(pcs[j])

    for k in testx:
        testx[k] = testx[k].to(f"cuda:{gpu_id}")
    for m in testy:
        testy[m] = testy[m].to(f"cuda:{gpu_id}")

    # for testing with a fixed rotation uncomment the following:
    # mat = zreg.dtw.create_dtw_matrix_given_rigid_rot(
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
    # Otherwise, use this for the general CPD fitting
    mat, rots = zreg.dtw.create_dtw_matrix(
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


for downsampling in downsampling_metrics:
    for i, j in [[1, 5], [2, 5], [3, 5], [4, 5]]:
        _proc_run(
            gpu_id=0,
            distance=distance_metrics,
            downsampling=downsampling,
            i=i,
            j=j,
            cpd_type="rigid",
        )
