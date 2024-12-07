import zreg
import torch
from pathlib import Path
import copy
from itertools import combinations
import time
import os


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
    device="cpu",
)
pcs[2], _alltrackles = zreg.dataset.load_data_from_tracklets(
    filepath="/p/project/tissuetwin/ZebraTwin/data/12_11_27_embryo_ew_08_Cleaned_BackTracked_Oriented.tracklets",
    device="cpu",
)
pcs[3], _alltrackles = zreg.dataset.load_data_from_tracklets(
    filepath="/p/project/tissuetwin/ZebraTwin/data/12_11_30_embryo_ew_11_Cleaned_BackTracked_Oriented.tracklets",
    device="cpu",
)
pcs[4], _alltrackles = zreg.dataset.load_data_from_tracklets(
    filepath="/p/project/tissuetwin/ZebraTwin/data/12_12_04_embryo_ew_12_Cleaned_BackTracked_Oriented.tracklets",
    device="cpu",
)

# precompute farthest points before doing anything
t0 = time.perf_counter()
print("getting all farthest points, will take some time...")
for a in range(1, 5):
    for k in pcs[a]:
        if "fps-idx" not in pcs[a][k]:
            pcs[a][k] = zreg.downsampling.precompute_fps(pcs[a][k])
print(f"Finished with farthest point precompute, time required: {time.perf_counter() - t0}")

distance_metrics = [
    # "swd",
    # "aswd",
    # "oswd",
    # "gswd",
    # "pswd",
    # "euclidean",
    "manhattan",
    "minkowski",
]
# TODO: set up furthest sampling to be faster somehow?
downsampling_metrics = [
    "farthest",
]  # "random", "uniform", None,
out_dir = Path("/p/project/tissuetwin/coquelin1/zReg/figures")


def _proc_run(gpu_id, distance, downsampling, norm, i, j):
    t0 = time.perf_counter()
    out_file = out_dir / f"{distance}_{downsampling}_{norm}_{i}_{j}.txt"
    print(f"Working on distance: {distance}, downsampling: {downsampling}, norm: {norm}, combi: {i, j} on gpu {gpu_id}")
    # testx = {i: copy.deepcopy(pcs[i]) for i in range(5)}
    # testy = {i: copy.deepcopy(pcs2[i]) for i in range(5)}
    testx = copy.deepcopy(pcs[i])
    testy = copy.deepcopy(pcs[j])

    for k in testx:
        testx[k] = zreg.dataset.to(testx[k], f"cuda:{gpu_id}")
    for m in testy:
        testy[m] = zreg.dataset.to(testy[m], f"cuda:{gpu_id}")
    mat = zreg.dtw.create_dtw_matrix(
        x=testx,
        y=testy,
        window=None,
        distance_metric=distance,
        distance_kwargs=None,
        normalize=norm,
        downsample_method=downsampling,
    )
    print(f"finished iteration, time required: {time.perf_counter() - t0}")
    np.savetxt(out_file, mat.cpu().numpy())
    del testx, testy


rank = int(os.environ["SLURM_PROCID"])
print(f"rank: {rank}")
print(f"device count: {torch.cuda.device_count()}")

# want to run a test for all
d = 0
# waits = []
# with ProcessPoolExecutor(max_workers=4) as executor:
for distance in distance_metrics:
    downmets = downsampling_metrics
    # if distance in ["euclidean", "manhattan", "minkowski"]:
    #     downmets = downsampling_metrics + [None]
    for downsampling in downmets:
        for norm in [True, False]:
            for i, j in combinations([1, 2, 3, 4], 2):
                if d != rank:
                    d += 1
                    d = d % 4
                    continue
                # w = executor.submit(
                _proc_run(
                    gpu_id=0,
                    distance=distance,
                    downsampling=downsampling,
                    norm=norm,
                    i=i,
                    j=j,
                )
                d += 1
                d = d % 4
                # waits.append(w)
                # w.result()
    # for w in waits:
    #     w.result()
