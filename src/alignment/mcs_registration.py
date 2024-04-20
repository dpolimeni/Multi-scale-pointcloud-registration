# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:03:54 2022

@author: Alessandro & Diego
"""
from src.utils.util_mcs_registration import *
import time

"IMPORT LIBRARIES"
import os
import sys
from copy import deepcopy
from datetime import datetime

import yaml

sys.path.append("../../")

if __name__ == "__main__":

    # USER_DIR = os.path.dirname(os.getcwd())
    YAML_PATH = "/Users/dpolimeni/Desktop/projects/pointcloud_registration/Multi-scale-pointcloud-registration/yml/mcs_registration.yml"
    N_POINTS_RANDOM_DS = 15000

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
    data_path = os.path.dirname(os.getcwd())
    source_path = os.path.join(data_path, source_path)
    target_path = os.path.join(data_path, target_path)
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

    "PERTURB CLOUD"
    n_points = int(n_points * target_array.shape[0])
    print(n_points, "of noise added")
    # target_array, R, t, initial_distortion = perturb_cloud(
    #     target_array, deg=deg, mu=mu, std=std, n_points=n_points
    # )
    # print("Initial distortion: ", initial_distortion)
    # print("Initial rotation:", R)
    # print("Initial translation", t)

    "GET FIRST POINT FOR EVALUATION"
    first_source = source_array
    first_target = target_array[:-n_points, :]

    if visualize_initial_alignment:
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target_array)

        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source_array)

        pcd_target.paint_uniform_color([0.0, 0.0, 1.0])
        pcd_source.paint_uniform_color([1.0, 0.0, 0.0])

        o3d.visualization.draw_geometries([pcd_source, pcd_target])

    "CLEAN CLOUD"
    target_array_no_sor = deepcopy(target_array)
    source_array_no_sor = deepcopy(source_array)

    if clean_cloud == True:
        print("Removing noise from target cloud...", end="")
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target_array)
        # Create copy of target array before outlier removal
        # Use SOR to remove outliers
        inliers, ind = pcd_target.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # Remove # to see outliers according to SOR
        # outliers = pcd_target.select_by_index(ind, invert = True)
        # inliers.paint_uniform_color([0., 0.4, 0.])
        # outliers.paint_uniform_color([1., 0.2, 0.])
        # o3d.visualization.draw_geometries([inliers,outliers])

        # Keep inliers only for the purpose of registration
        target_array = np.asarray(inliers.points)

        print("done")

        print("Removing noise from source cloud...", end="")
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source_array)
        inliers, ind = pcd_source.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # Remove # to see outliers according to SOR
        # outliers = pcd_target.select_by_index(ind, invert = True)
        # inliers.paint_uniform_color([0., 0.4, 0.])
        # outliers.paint_uniform_color([1., 0.2, 0.])
        # o3d.visualization.draw_geometries([inliers,outliers])
        source_array = np.asarray(inliers.points)

    "SUBSAMPLE source"
    print("Subsampling clouds...")
    # Uniform subsamplig without reinsertion
    source_ds = subsample(source_array, n_points=N_POINTS_RANDOM_DS)
    target_ds = subsample(target_array, n_points=N_POINTS_RANDOM_DS)
    # Further subsampling with FPS (subsampling above to speed up this step)
    target_ds = subsample_fps(target_ds, n_points=n_points_target_ds)
    source_ds = subsample_fps(source_ds, n_points=n_points_source_ds)

    "SCALE AND MOVE POINTCLOUDS"
    print("Normalizing clouds...", end="")
    source_center_ds = np.mean(source_ds, axis=0, keepdims=True)
    target_center_ds = np.mean(target_ds, axis=0, keepdims=True)

    dist_source_ds = np.max(cdist(source_center_ds, source_ds))
    dist_target_ds = np.max(cdist(target_center_ds, target_ds))

    source_array = (source_array - source_center_ds) / dist_source_ds
    target_array = (target_array - target_center_ds) / dist_target_ds
    target_array_no_sor = (target_array_no_sor - target_center_ds) / dist_target_ds
    source_array_no_sor = (source_array_no_sor - source_center_ds) / dist_source_ds
    source_ds = (source_ds - source_center_ds) / dist_source_ds
    target_ds = (target_ds - target_center_ds) / dist_target_ds

    first_source = (first_source - source_center_ds) / dist_source_ds
    first_target = first_target - target_center_ds

    print("done")

    "CREATE AND VISUALIZE POINTCLOUDS"
    pcd_target_no_sor = o3d.geometry.PointCloud()
    pcd_target_no_sor.points = o3d.utility.Vector3dVector(target_array_no_sor)

    pcd_source_no_sor = o3d.geometry.PointCloud()
    pcd_source_no_sor.points = o3d.utility.Vector3dVector(source_array_no_sor)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_array)

    pcd_target_ds = o3d.geometry.PointCloud()
    pcd_target_ds.points = o3d.utility.Vector3dVector(target_ds)

    pcd_source_ds = o3d.geometry.PointCloud()
    pcd_source_ds.points = o3d.utility.Vector3dVector(source_ds)

    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source_array)

    pcd_target_no_sor.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_source_no_sor.paint_uniform_color([0.2, 0.2, 0.2])
    pcd_target.paint_uniform_color([0.0, 0.0, 1.0])
    pcd_source.paint_uniform_color([1.0, 0.0, 0.0])
    pcd_target_ds.paint_uniform_color([0.0, 0.0, 1.0])
    pcd_source_ds.paint_uniform_color([1.0, 0.0, 0.0])

    if visualize_initial_alignment:
        o3d.visualization.draw_geometries([pcd_source, pcd_target])

    "GET VOXEL SIZE WITH COMPASS SEARCH"
    print("Finding correct voxel size...", end="")
    voxel_size_source = get_voxel_size(
        pcd_source_ds,
        target_points=target_points_source,
        vs_0=starting_voxel_size,
        delta=delta,
        eps=stopping_delta,
    )
    voxel_size_target = get_voxel_size(
        pcd_target_ds,
        target_points=target_points_target,
        vs_0=starting_voxel_size,
        delta=delta,
        eps=stopping_delta,
    )
    print("done")

    "PREPROCESS POINT CLOUD"
    print("Preprocessing point clouds...", end="")
    # Just for sake of visualization
    pcd_source_viz = copy.deepcopy(pcd_source_ds)
    print("DEEPCOPY OF SOURCE")
    pcd_target_viz = copy.deepcopy(pcd_target_ds)
    # Subsampling + normal estimation + fpfh
    print(
        "SOURCE VOXEL DOWNSAMPLED + FPFH",
        radius_normal,
        radius_feature,
        voxel_size_source,
    )
    pcd_source_ds, pcd_source_ds_fpfh = preprocess_point_cloud(
        pcd_source_ds,
        radius_normal=radius_normal,
        radius_feature=radius_feature,
        voxel_size=voxel_size_source,
    )
    pcd_target_ds, pcd_target_ds_fpfh = preprocess_point_cloud(
        pcd_target_ds,
        radius_normal=radius_normal,
        radius_feature=radius_feature,
        voxel_size=voxel_size_target,
    )

    print("done")
    print("source:", pcd_source_ds, "requested:", target_points_source)
    print("target:", pcd_target_ds, "requested:", target_points_target)

    "RUN COMPASS SEARCH"
    print("Compass Search Started")
    start = time.time()
    print(
        "COMPASS SEARCH",
        n_attempts,
        compass_delta,
        compass_eps,
        compass_max_iter,
        reg_type,
    )
    T_star, coeff_star, rmse_star, errors = compass_search(
        pcd_source_ds,
        pcd_target_ds,
        pcd_source_ds_fpfh,
        pcd_target_ds_fpfh,
        n_attempts=n_attempts,
        delta=compass_delta,
        eps=compass_eps,
        max_iter=compass_max_iter,
        reg_type="gicp",
    )

    print("time requested:", (time.time() - start) / 60, "minutes")

    "SCALE CLOUDS"
    pcd_target.points = o3d.utility.Vector3dVector(np.array(pcd_target.points) * coeff_star)
    pcd_target_no_sor.points = o3d.utility.Vector3dVector(np.array(pcd_target_no_sor.points) * coeff_star)
    pcd_target_ds.points = o3d.utility.Vector3dVector(np.array(pcd_target_ds.points) * coeff_star)
    pcd_target_viz.points = o3d.utility.Vector3dVector(np.array(pcd_target_viz.points) * coeff_star)
    draw_registration_result(pcd_target_viz, pcd_source_viz, T_star)
    "REFINE REGISTRATION"
    print("Refining registration...", end="")
    print(
        "REFINEMENT",
        refinement_iterations,
        refinement_distnace_threshold,
        icp_type,
        T_star,
    )
    T_refined = refine_registration(
        pcd_source_ds,
        pcd_target_ds,
        max_iteration=refinement_iterations,
        initial_transform=T_star,
        distance_threshold=refinement_distnace_threshold,
        icp_type=icp_type,
    )
    print("done")

    "GET INVERSE TRANSFORMATION MATRIX"
    T = np.copy(T_refined)
    T[:3, :3] = T_refined[:3, :3].T
    T[:3, 3] = -np.dot(T_refined[:3, 3], T_refined[:3, :3])
    print("Transformation matrix:", T)
    "VISUALIZE BEST ALIGNMENT"
    draw_registration_result(pcd_target_viz, pcd_source_viz, T)
    draw_registration_result(pcd_target_no_sor, pcd_source_no_sor, T)

    # evaluate_pipe(R, t, T_refined)
    # evaluate_distortion(first_source, first_target, coeff_star, T, dist_target_ds)

    # path = r"C:/Users/dpolimeni/Desktop/paper_project/{}_{}".format(
    #     reg_type, datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")
    # )
    # with open(path, "w") as f:
    #     for elem in errors:
    #         f.write("\n" + str(elem))

# 'SAVE ALIGNED AND SCALED POINTCLOUDS AS NPZ FOR SEMANTIC SEGMENTATION'
# pcd_target_no_sor.transform(T)
# print('Saving new clouds...', end = '')
# np.savez(target_path_out, points = np.asarray(pcd_target_no_sor.points))
# np.savez(source_path_out, points = source_array)
# print('done')
