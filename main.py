from Aligner.Aligner import Aligner
from Preprocessor.Preprocessor import Preprocessor
from Preprocessor.RandomDownsampler import RandomDownsampler
from Preprocessor.VoxelDownsampler import VoxelDownsampler
from Optimizer.FastGlobalOptimizer import FastGlobalOptimizer
import numpy as np
import open3d as o3d
import copy


def main():
    # Load point clouds
    source = o3d.io.read_point_cloud("data/fragment.ply")
    target = o3d.io.read_point_cloud("data/fragment.ply")

    # Downsample point clouds
    source = source.voxel_down_sample(voxel_size=0.05)
    target = target.voxel_down_sample(voxel_size=0.05)

    random_downsample = RandomDownsampler(sample_size=1000)
    voxel_downsample = VoxelDownsampler(target_points=1000)

    # Preprocess point clouds
    source_preprocessor = Preprocessor([random_downsample, voxel_downsample])
    target_preprocessor = Preprocessor([random_downsample, voxel_downsample])

    # TODO insert parameters
    # Create an optimizer
    optimizer = FastGlobalOptimizer()

    # Create an aligner
    aligner = Aligner(source_preprocessor, target_preprocessor, optimizer)

    # Perform multistart registration
    T = aligner.multistart_registration(source, target)

    # Apply transformation to source point cloud
    source.transform(T)

    # Visualize
    o3d.visualization.draw_geometries([source, target])


if __name__ == "__main__":
    main()