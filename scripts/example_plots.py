from zreg.core.dataset import load_data_from_tracklets, zRegPointCloud, zreg_to_open3d
from zreg.algorithms.cpd import cpd_registration
from zreg.preprocessing.downsampling import farthest_point_down_sample
from zreg.core.transforms import NonRigidTransformation, transform_points_homogeneous
import copy
import open3d as o3d
import numpy as np
import torch

DEFAULT_TRACKLETS_PATH = (
    "/p/project/tissuetwin/ZebraTwin/data/12_11_15_embryo_ew_06_Cleaned_BackTracked_Oriented.tracklets"
)


# define a visualization function to display the results
def draw_registration_result(source, target):
    # deep copies used to avoid overwriting
    if isinstance(source, zRegPointCloud):
        # convert to open3d point cloud for visualization
        spc = zreg_to_open3d(copy.deepcopy(source))
    else:
        spc = copy.deepcopy(source)
    if isinstance(target, zRegPointCloud):
        tpc = zreg_to_open3d(copy.deepcopy(target))
    else:
        tpc = copy.deepcopy(target)
    spc.paint_uniform_color([1, 0.706, 0])
    tpc.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_plotly([spc.to_legacy(), tpc.to_legacy()])

def draw_single_pc(pc, coloring_type='source'):
    if coloring_type == 'source':
        color = [1, 0.706, 0]
    elif coloring_type == 'target':
        color = [0, 0.651, 0.929]
    else:
        raise ValueError("coloring_type must be 'source' or 'target'")
    # deep copies used to avoid overwriting
    if isinstance(pc, zRegPointCloud):
        # convert to open3d point cloud for visualization
        spc = zreg_to_open3d(copy.deepcopy(pc))
    else:
        spc = copy.deepcopy(pc)
    spc.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_plotly([spc.to_legacy()])


def main(tracklets_path: str = DEFAULT_TRACKLETS_PATH, device: str = "cpu") -> None:
    # load a time series of point clouds
    pcs, _alltrackles = load_data_from_tracklets(
        filepath=tracklets_path,
        device=device,
    )

    true_transform = np.asarray(
        [[0.866, -0.5, 0.0, 0.0], [0.5, 0.866, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    source_idx = 0
    target_idx = 0

    # copy the point clouds so they dont need to re-loaded
    source = copy.deepcopy(pcs[source_idx])
    target = copy.deepcopy(pcs[target_idx])

    # downsample whichever is larger, if the same, they are returned as is
    target, source = farthest_point_down_sample(target, source)

    # apply the transform to the source
    source = transform_points_homogeneous(source, true_transform, return_o3d=False)
    pcsource = zreg_to_open3d(source)
    pctarget = zreg_to_open3d(target)
    draw_registration_result(pcsource, pctarget)

    # Do a rigid registration. If another type of registration is desired, then call it
    result = cpd_registration(
        source=copy.deepcopy(source),
        target=copy.deepcopy(target),
        tf_type_name="rigid",
        use_color=False,
    )
    # print(result)
    print(f"Rotation:\n{result.transformation.rot}\nTranslation:\n{result.transformation.t}")

    # Plot results
    source2 = copy.deepcopy(pcs[source_idx])
    target2 = copy.deepcopy(pcs[target_idx])

    source2["pos"] = result.transformation.transform(source2["pos"])

    pc1new = zreg_to_open3d(source2)
    pc0new = zreg_to_open3d(target2)
    draw_registration_result(pc1new, pc0new)

    # example non-rigid transformation + example of how the other transformations work
    x = copy.deepcopy(pcs[0])
    y = copy.deepcopy(pcs[0])
    x, y = farthest_point_down_sample(x, y)

    NonRigidTrans = NonRigidTransformation(
        w=torch.ones_like(pcs[0]["pos"]), points=pcs[0]["pos"], beta=0.005
    )

    y["pos"] = NonRigidTrans.transform(y["pos"])

    pc1new = zreg_to_open3d(x)
    pc0new = zreg_to_open3d(y)

    draw_registration_result(pc1new, pc0new)

    # example non-rigid registration
    result = cpd_registration(
        source=copy.deepcopy(x),
        target=copy.deepcopy(y),
        tf_type_name="nonrigid_constrained",
        use_color=False,
        maxiter=100,
        beta=0.01,
        lmd=0.1,
    )
    print(result)

    print(result.transformation.g)

    x1 = copy.deepcopy(x)
    y1 = copy.deepcopy(y)

    x1["pos"] = result.transformation.transform(x1["pos"])

    pc1new = zreg_to_open3d(x1)
    pc0new = zreg_to_open3d(y1)
    draw_registration_result(pc1new, pc0new)


if __name__ == "__main__":
    main()
