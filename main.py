import os
import time

import numpy as np
import open3d as o3d
import yaml

from src.Aligner.Aligner import Aligner
from src.Optimizer.generalizedICP import GeneralizedICP
from src.Preprocessor.farthestDownsampler import FarthestDownsampler
from src.Preprocessor.preprocessor import Preprocessor
from src.Preprocessor.randomDownsampler import RandomDownsampler
from src.Preprocessor.scaler import Scaler
from src.Preprocessor.voxelDownsampler import VoxelDownsampler
from src.Visualizer.Visualizer import visualize_point_clouds, draw_registration_result
from src.utils.create_cloud import create_cloud
from src.utils.constants import (
    __FARTHEST_SAMPLE_SIZE__,
    __RANDOM_SAMPLE_SIZE__,
    __VOXEL_SAMPLE_SIZE__,
    __BASE_VOXEL_SIZE__,
    __MIN_VOXEL_SIZE__,
    __DELTA__,
    __EPS__,
    __ALIGNER_DELTA__,
    __ALIGNER_DEG__,
    __ALIGNER_MU__,
    __ALIGNER_STD__,
    __MULTISTART_ATTEMPTS__,
    __ALIGNER_EPS__,
)


def main():
    YAML_PATH = os.path.join(os.getcwd(), "yml", "mcs_registration.yml")

    with open(YAML_PATH, "r") as f:
        config_dict = yaml.safe_load(f)
        f.close()

    "READ mcs_registration.yml"
    source_path = config_dict["source_path"]
    target_path = config_dict["target_path"]
    is_npz = config_dict["is_npz"]

    "LOAD CLOUDS"
    if is_npz:
        print("Loading Clouds...", end="")
        source = np.load(source_path, allow_pickle=True)
        target = np.load(target_path, allow_pickle=True)

        source_array = np.array(source["points"])[:, :3]
        target_array = np.array(target["points"])[:, :3]

        print("done")
    else:
        print("Loading Clouds...", end="")
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)

        source_array = np.array(source.points)[:, :3]
        target_array = np.array(target.points)[:, :3]

        print("done")

    "PREPROCESSING"
    print("Preprocessing...", end="")
    random_downsampler = RandomDownsampler(__RANDOM_SAMPLE_SIZE__)
    fps_downsampler = FarthestDownsampler(__FARTHEST_SAMPLE_SIZE__)
    source_voxel_downsampler = VoxelDownsampler(
        __VOXEL_SAMPLE_SIZE__,
        __BASE_VOXEL_SIZE__,
        __MIN_VOXEL_SIZE__,
        __DELTA__,
        __EPS__,
    )

    target_voxel_downsampler = VoxelDownsampler(
        __VOXEL_SAMPLE_SIZE__,
        __BASE_VOXEL_SIZE__,
        __MIN_VOXEL_SIZE__,
        __DELTA__,
        __EPS__,
    )
    scaler = Scaler()

    source_preprocessor = Preprocessor(
        [random_downsampler, fps_downsampler, scaler, source_voxel_downsampler],
    )
    target_preprocessor = Preprocessor(
        [random_downsampler, fps_downsampler, scaler, target_voxel_downsampler],
    )

    # TODO remove magic numbers
    optimizer = GeneralizedICP(
        max_correspondence_distance=0.5,
        max_iterations=100,
    )

    aligner = Aligner(
        source_preprocessor,
        target_preprocessor,
        optimizer,
        attempts=__MULTISTART_ATTEMPTS__,
        deg=__ALIGNER_DEG__,
        mu=__ALIGNER_MU__,
        std=__ALIGNER_STD__,
        delta=__ALIGNER_DELTA__,
        eps=__ALIGNER_EPS__,
        visualize_intermediate_steps=True,
    )
    start = time.time()

    optimal_transformation, optimal_metric, optimal_scale_factor, errors = (
        aligner.align(source_array, target_array)
    )
    print("done")
    print(
        f"Optimal transformation: \n{optimal_transformation}\nOptimal metric: {optimal_metric}\nErrors: {errors}\nOptimal Scale Factors: {optimal_scale_factor}"
    )
    print(f"Elapsed time: {time.time() - start}")

    visualize_point_clouds(
        [source_array, target_array * optimal_scale_factor], [(0, 0, 1), (1, 0, 0)]
    )
    source_processed = source_preprocessor.preprocess(source_array)
    target_processed = target_preprocessor.preprocess(target_array)
    source = create_cloud(source_processed)
    target = create_cloud(target_processed * optimal_scale_factor)

    draw_registration_result(source, target, optimal_transformation)


if __name__ == "__main__":
    main()
