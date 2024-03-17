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
    __VOXEL_SAMPLE_SIZE__,
    __BASE_VOXEL_SIZE__,
    __MIN_VOXEL_SIZE__,
    __DELTA__,
    __EPS__,
)


def main():
    YAML_PATH = os.path.join(os.getcwd(), "yml", "mcs_registration.yml")

    with open(YAML_PATH, "r") as f:
        config_dict = yaml.safe_load(f)
        f.close()

    "READ mcs_registration.yml"
    source_path = config_dict["source_path"]
    target_path = config_dict["target_path"]
    source_path_out = config_dict["source_path_out"]
    target_path_out = config_dict["target_path_out"]

    deg = config_dict["deg"]
    mu = config_dict["mu"]
    std = config_dict["std"]
    n_points = config_dict["n_points"]

    clean_cloud = config_dict["clean_cloud"]
    std_ratio = config_dict["std_ratio"]
    nb_neighbors = config_dict["nb_neighbors"]

    n_points_source_ds = config_dict["n_points_source_ds"]
    n_points_target_ds = config_dict["n_points_target_ds"]

    radius_normal = config_dict["radius_normal"]
    radius_feature = config_dict["radius_feature"]

    target_points_source = config_dict["target_points_source"]
    target_points_target = config_dict["target_points_target"]
    starting_voxel_size = config_dict["starting_voxel_size"]
    delta = config_dict["delta"]
    stopping_delta = config_dict["stopping_delta"]

    n_attempts = config_dict["n_attempts"]
    compass_delta = config_dict["compass_delta"]
    compass_eps = config_dict["compass_eps"]
    compass_max_iter = config_dict["compass_max_iter"]

    refinement_iterations = config_dict["refinement_iterations"]
    refinement_distnace_threshold = config_dict["refinement_distnace_threshold"]
    reg_type = config_dict["reg_type"]
    icp_type = config_dict["icp_type"]

    visualize_initial_alignment = config_dict["visualize_initial_alignment"]
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
        attempts=n_attempts,
        deg=deg,
        mu=mu,
        std=std,
        delta=compass_delta,
        eps=compass_eps,
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
