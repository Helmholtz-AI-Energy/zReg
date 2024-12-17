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

# precompute farthest points before doing anything
t0 = time.perf_counter()
# print("getting all farthest points, will take some time...")
for a in range(1, 5):
    # if (a - 1) % MPI.COMM_WORLD.rank != 0:
    #     continue
    for k in pcs[a]:
        if "fps-idx" not in pcs[a][k]:
            pcs[a][k] = zreg.downsampling.precompute_fps(pcs[a][k])
        # print(f"done with sample {k} in file {a}")

# for a in range(1, 5):
#     for k in pcs[a]:
#         pcs[a][k] = zreg.mpi_tools.broadcast_pc(pcs[a][k], root=(a - 1) % MPI.COMM_WORLD.rank)

# print(f"Finished with farthest point precompute, time required: {time.perf_counter() - t0}")

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
    "farthest",
    "random",
]  #  "uniform", ]  #  None,
out_dir = Path("/p/project/tissuetwin/coquelin1/zReg/figures")


def _proc_run(gpu_id, distance, downsampling, norm, i, j, cpd_type):
    t0 = time.perf_counter()
    print(
        f"Working on distances: {distance}, downsampling: {downsampling}, norm: {norm}, combi: {i, j} on gpu {gpu_id}"
    )
    testx = copy.deepcopy(pcs[i])
    testy = copy.deepcopy(pcs[j])

    for k in testx:
        testx[k] = testx[k].to(f"cuda:{gpu_id}")
    for m in testy:
        testy[m] = testy[m].to(f"cuda:{gpu_id}")
    mat = zreg.dtw.create_dtw_matrix(
        x=testx,
        y=testy,
        window=None,
        distance_metric=distance,
        distance_kwargs=None,
        normalize=norm,
        downsample_method=downsampling,
        cpd_type=cpd_type,
        mpi_distribute=True,
        # save_filename=out_file,
    )
    print(f"finished iteration, time required: {time.perf_counter() - t0}")
    for c, dist in enumerate(distance):
        out_file = out_dir / "multi" / f"{dist}_{downsampling}_{norm}_{i}_{j}.txt"
        np.savetxt(out_file, mat[c].cpu().numpy())
    del testx, testy


rank = int(os.environ["SLURM_PROCID"])
tasks = int(os.environ["SLURM_NTASKS"])
print(f"rank: {rank}, TASKS: {tasks}")
print(f"device count: {torch.cuda.device_count()}")

# want to run a test for all
d = 0
# waits = []
# with ProcessPoolExecutor(max_workers=4) as executor:
# for distance in distance_metrics:
downmets = downsampling_metrics
# if distance in ["euclidean", "manhattan", "minkowski"]:
#     downmets = downsampling_metrics + [None]
for downsampling in downmets:
    for norm in [
        True,
    ]:
        for i, j in [[1, 5], [2, 5], [3, 5], [4, 5]]:
            # combinations([1, 2, 3, 4, 5], 2):
            # if d != rank:
            #     d += 1
            #     d = d % tasks
            #     continue
            # w = executor.submit(
            _proc_run(
                gpu_id=0,
                distance=distance_metrics,
                downsampling=downsampling,
                norm=norm,
                i=i,
                j=j,
                cpd_type="rigid",
            )
            d += 1
            d = d % 4
            # waits.append(w)
            # w.result()
    # for w in waits:
    #     w.result()
