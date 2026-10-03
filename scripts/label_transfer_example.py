#!/usr/bin/env python
"""
Example script demonstrating color transfer functionality in zReg.

This script shows how to:
1. Load point cloud data
2. Perform CPD registration
3. Transfer colors from source to target using different methods
"""

from zreg.core.dataset import zRegPointCloud
from zreg.algorithms.cpd import RigidCPD, cpd_registration
from zreg.label_transfer import transfer_labels as transfer_colors, LabelTransferMethod as ColorTransferMethod

import torch  # imported after zreg (libomp ordering on macOS ARM)


def main():
    """Demonstrate color transfer functionality."""
    print("zReg Color Transfer Example")
    print("=" * 40)

    # Create sample data
    print("Creating sample point clouds...")

    # Source point cloud
    source_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=torch.float32)

    source_colors = torch.tensor([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0]   # Yellow
    ], dtype=torch.float32)

    source_pc = zRegPointCloud(pos=source_pos, label=source_colors)

    # Target point cloud (slightly offset and rotated)
    target_pos = torch.tensor([
        [0.1, 0.1, 0.0],  # Close to source[0]
        [0.9, 0.1, 0.0],  # Close to source[1]
        [0.1, 0.9, 0.0],  # Close to source[2]
        [0.9, 0.9, 0.0]   # Close to source[3]
    ], dtype=torch.float32)

    target_pc = zRegPointCloud(pos=target_pos)

    print(f"Source: {source_pc['pos'].shape[0]} points")
    print(f"Target: {target_pc['pos'].shape[0]} points")

    # Perform CPD registration
    print("\nPerforming CPD registration...")
    cpd_result = cpd_registration(
        source_pc, target_pc,
        tf_type_name="rigid",
        maxiter=50,
        tol=1e-4,
        log_freq=-1  # Disable logging
    )

    print("Registration complete!")

    # Apply color transfer using different methods
    print("\nApplying color transfer...")

    # Method 1: Nearest Neighbor
    print("1. Nearest Neighbor method:")
    colors_nn = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.NEAREST_NEIGHBOR
    )
    print(f"   Transferred colors shape: {colors_nn.shape}")
    print(f"   Sample colors: {colors_nn[:2]}")

    # Method 2: CPD-weighted (requires EstepResult)
    print("2. CPD-weighted method:")

    # We need to run the expectation step to get probabilities
    # For this example, we'll create a simple CPD object and get the E-step result
    cpd_obj = RigidCPD(source_pc["pos"], log_freq=-1)
    cpd_obj.transformation = cpd_result.transformation  # Use the result

    # Run one expectation step to get probabilities
    transformed_source = cpd_obj.transformation.transform(source_pc["pos"])
    estep_result = cpd_obj.expectation_step(
        transformed_source, target_pc["pos"],
        cpd_result.sigma2, 0.0, w=0.0
    )

    # expectation_step() returns pmat shaped (n_source, n_target), but the
    # CPD-weighted transfer expects (n_target, n_source).  Pass a transposed
    # copy (same D-04 fix as eval/stages/label_transfer.py).  The demo clouds
    # are 4x4, so without the transpose this would silently use the wrong
    # orientation instead of raising.
    colors_cpd = transfer_colors(
        source_pc, target_pc,
        method=ColorTransferMethod.CPD_WEIGHTED,
        estep_result=estep_result._replace(pmat=estep_result.pmat.T),
        # The transposed pmat is (n_target, n_source): declare it, never inferred.
        pmat_layout="receiver_provider",
    )
    print(f"   Transferred colors shape: {colors_cpd.shape}")
    print(f"   Sample colors: {colors_cpd[:2]}")

    # Compare results
    print("\nComparison:")
    print("Original source colors:")
    for i, color in enumerate(source_colors):
        print(f"  Point {i}: [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]")

    print("\nTransferred colors (Nearest Neighbor):")
    for i, color in enumerate(colors_nn):
        print(f"  Point {i}: [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]")

    print("\nTransferred colors (CPD-weighted):")
    for i, color in enumerate(colors_cpd):
        print(f"  Point {i}: [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]")

    print("\nColor transfer demonstration complete!")


if __name__ == "__main__":
    main()