# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:52:53 2022

@author: Alessandro & Diego
"""

"IMPORT LIBRARIES"
import copy

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from tqdm import tqdm

"SHOW TWO CLOUDS AGAINST EACH OTHER"


# Input: 2 o3d pointcloud, 4x4 array
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.0, 0.0, 1.0])
    target_temp.paint_uniform_color([1.0, 0.0, 0.0])

    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp])


"UNIFORM SUBSAMPLING OF POINTCLOUD, WITHOUT REINSERTION"


def subsample(X, n_points):
    X_new = []

    random_indexes = np.random.choice(X.shape[0], size=(n_points,), replace=False)

    Cloud = X[random_indexes]

    X_new.append(Cloud)

    return np.array(X_new)[0]


"FARTHEST POINT SAMPLING"


def subsample_fps(X, n_points):
    X_new = []

    index_list = np.zeros(n_points, dtype=int)
    index_list[0] = np.random.randint(low=0, high=X.shape[0])

    D = np.ones((1, X.shape[0])) * 1e6

    for i in tqdm(range(n_points - 1)):
        h = X[index_list[i]].reshape((1, X.shape[1]))

        D_corrente = cdist(h, X)
        D = np.min((D_corrente, D), axis=0)

        fp_index = np.argmax(D)
        index_list[i + 1] = fp_index

    X_new.append(X[index_list])

    return np.array(X_new)[0]


"GET THE CORRECT VOXEL SIZE"


def get_voxel_size(source, target_points, vs_0=0.02, delta=0.01, eps=0.001):
    vs = vs_0
    voxel_size = vs_0
    # Downsample
    source_ds = source.voxel_down_sample(vs_0)
    # Get number of points
    n_points = np.asarray(source_ds.points).shape[0]
    # Compute Metric
    metric = np.abs(target_points - n_points)
    while delta >= eps:
        vs = voxel_size + delta
        source_ds = source.voxel_down_sample(vs)
        n_points = np.asarray(source_ds.points).shape[0]
        new_metric = np.abs(target_points - n_points)

        if new_metric < metric:
            voxel_size = vs
            metric = new_metric
            continue

        vs = voxel_size - delta
        if vs <= 0.0001:
            vs = 0.0001
        source_ds = source.voxel_down_sample(vs)
        n_points = np.asarray(source_ds.points).shape[0]
        new_metric = np.abs(target_points - n_points)

        if new_metric < metric:
            voxel_size = vs
            metric = new_metric
            continue

        delta = delta / 2
    print("Voxel size:", voxel_size, "Number of points:", n_points)
    return voxel_size


"ESTIMATE NORMALS AND FPFH FEATURES FOR REGISTRATION ALGORITHMS"


# Input: o3d pointcloud, scalars
def preprocess_point_cloud(pcd, radius_normal, radius_feature, voxel_size):
    pcd_down = copy.deepcopy(pcd)
    pcd_down = pcd_down.voxel_down_sample(voxel_size)

    # Estimate Normals
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # Estimate FPFH Features
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    return pcd_down, pcd_fpfh


"GENERATE INITIAL ROTATION + TRANSLATION MATRIX"


def generate_initialization_matrix(deg=0.2, mu=0, std=0.1):
    # Initialize transformation matrix
    T = np.eye(4)
    # Generate random angles
    theta_1 = np.random.uniform(low=-deg, high=deg)
    theta_2 = np.random.uniform(low=-deg, high=deg)
    theta_3 = np.random.uniform(low=-deg, high=deg)
    # Create the 3 basic rotation matrices
    R1 = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_1), -np.sin(theta_1)],
            [0, np.sin(theta_1), np.cos(theta_1)],
        ]
    )
    R2 = np.array(
        [
            [np.cos(theta_2), 0, np.sin(theta_2)],
            [0, 1, 0],
            [-np.sin(theta_2), 0, np.cos(theta_2)],
        ]
    )
    R3 = np.array(
        [
            [np.cos(theta_3), -np.sin(theta_3), 0],
            [np.sin(theta_3), np.cos(theta_3), 0],
            [0, 0, 1],
        ]
    )
    # Aggregate them to form the final rotation matrix
    R = np.dot(R1, np.dot(R2, R3))
    # Generate random translation vector
    t = mu + np.random.randn(3) * std
    # Update T
    T[:3, :3] = R  # Place rotation matrix in the upper left
    T[:3, 3] = t  # Place the translation vector in the upper right

    return T


"FAST GLOBAL REGISTRATION, TO REFINE LATER ON"


# Inputs: o3d clouds, fpfh features, scalars
def execute_fast_global_registration(
    source, target, source_fpfh, target_fpfh, tuple_scale=0.55, distance_threshold=0.5
):
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        division_factor=1.4,
        tuple_scale=tuple_scale,  # 0.55
        maximum_correspondence_distance=distance_threshold,
        iteration_number=1024,
        decrease_mu=True,
    )

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, option
    )

    return result


"GENERALIZED ICP, TO REFINE LATER ON"


def execute_generalized_icp_registration(
    source, target, max_iterations=100, distance_threshold=0.8
):
    icp_type = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

    result = o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        distance_threshold,
        np.eye(4),
        icp_type,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    return result


"RUN MULTISTART REGISTRATION"


def execute_multistart_registration(
    source,
    target,
    source_fpfh,
    target_fpfh,
    n_attempts=1,
    tuple_scale=0.9,
    distance_threshold=0.8,
    reg_type="fgr",
):
    # Number of iterations for the multistart
    n_attempts = n_attempts
    # Initial metric and transformation
    metric = np.inf
    best_T = np.eye(4)

    for n in range(n_attempts):  # tqdm
        source_copy = copy.deepcopy(source)
        # generate random rotation and translation matrices
        mat_init = generate_initialization_matrix(deg=1.57, mu=0, std=0.1)
        # Apply transformation to the matrix
        source_copy.transform(mat_init)
        # Do fast_global_registration
        if reg_type == "fgr":
            result = execute_fast_global_registration(
                source_copy,
                target,
                source_fpfh,
                target_fpfh,
                tuple_scale,
                distance_threshold=distance_threshold,
            )
        if reg_type == "gicp":
            result = execute_generalized_icp_registration(
                source=source_copy, target=target
            )
        # Check if improvement
        if result.inlier_rmse < metric:
            # Update metric
            metric = result.inlier_rmse
            # print(metric)
            # Find T
            T = np.eye(4)
            T[:3, :3] = np.dot(result.transformation[:3, :3], mat_init[:3, :3])
            T[:3, 3] = (
                np.dot(result.transformation[:3, :3], mat_init[:3, 3]).ravel()
                + result.transformation[:3, 3]
            )
            best_T = T

    return best_T, metric


"RUN COMPASS SEARCH"


def compass_search(
    source,
    target,
    source_fpfh,
    target_fpfh,
    n_attempts,
    delta,
    eps,
    max_iter,
    reg_type="fgr",
):
    # INITIALIZE COUNTER
    it = 0
    # INITIAL SCALE FACTOR
    coeff_star = np.ones((1, 3))
    # DEEPCOPY TARGET CLOUD TO CHANGE IT DURING THE ALGORITHM
    pcd_target = copy.deepcopy(target)
    target_array = np.array(pcd_target.points)
    # DO THE FIRST MULTISTART TO GET THE FIRST METRIC
    T_star, rmse_star = execute_multistart_registration(
        source,
        pcd_target,
        source_fpfh,
        target_fpfh,
        n_attempts=n_attempts,
        distance_threshold=0.8,
        tuple_scale=0.9,
        reg_type=reg_type,
    )
    # INITIALIZE ERRORS LIST
    errors = [rmse_star]
    print("rmse star", rmse_star)
    directions = np.eye(3)

    while delta >= eps and it <= max_iter:
        # UPDATE COUNTER
        it += 1
        print("Iteration number:", it, "Current step:", delta)
        for j in range(3):
            coeff_plus = coeff_star + delta * directions[:, 2 - j]

            # SCALE TARGET CLOUD
            pcd_target.points = o3d.utility.Vector3dVector(target_array * coeff_plus)
            # RETRIEVE FPFH
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
            )
            # RUN MULTISTART
            T_plus, rmse_plus = execute_multistart_registration(
                source,
                pcd_target,
                source_fpfh,
                target_fpfh,
                n_attempts=n_attempts,
                distance_threshold=0.8,
                tuple_scale=0.9,
                reg_type=reg_type,
            )
            if rmse_plus <= rmse_star:
                print("New RMSE:", rmse_plus, "Old RMSE:", rmse_star)
                rmse_star = rmse_plus
                T_star = T_plus
                coeff_star = coeff_plus
                # print('TOP COEFF', coeff_star)
                errors.append(rmse_star)
                break

            # SCALE TARGET CLOUD
            coeff_neg = coeff_star - delta * directions[:, 2 - j]
            pcd_target.points = o3d.utility.Vector3dVector(target_array * coeff_neg)
            # RETRIEVE FPFH
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
            )
            # RUN MULTISTART
            T_neg, rmse_neg = execute_multistart_registration(
                source,
                pcd_target,
                source_fpfh,
                target_fpfh,
                n_attempts=n_attempts,
                distance_threshold=0.8,
                tuple_scale=0.9,
                reg_type=reg_type,
            )

            if rmse_neg <= rmse_star:
                print("New RMSE", rmse_neg, "Old RMSE", rmse_star)
                rmse_star = rmse_neg
                T_star = T_neg
                coeff_star = coeff_neg
                # print('TOP COEFF', coeff_star)
                errors.append(rmse_star)
                break

        if rmse_plus > rmse_star and rmse_neg > rmse_star:
            delta = delta / 2

    return T_star, coeff_star, rmse_star, errors


"REFINE REGISTRATION WITH CLASSIC ICP AFTER THE GLOBAL ONE"


def refine_registration(
    source,
    target,
    initial_transform,
    max_iteration=100,
    distance_threshold=0.1,
    icp_type="PointToPoint",
):
    if icp_type == "PointToPoint":
        icp_type = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )  # for faster convergence
    if icp_type == "PointToPlane":
        icp_type = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    # Define convergence criterion
    convergence_rule = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration
    )
    # Execute ICP
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transform,
        icp_type,
        convergence_rule,
    )

    return result.transformation


def add_noise(cloud, n_points=500):
    max_coordinates = np.max(cloud, axis=0)
    min_coordinates = np.min(cloud, axis=0)

    noise_x = np.random.uniform(
        min_coordinates[0], max_coordinates[0], n_points
    ).reshape((-1, 1))
    noise_y = np.random.uniform(
        min_coordinates[1], max_coordinates[1], n_points
    ).reshape((-1, 1))
    noise_z = np.random.uniform(
        min_coordinates[2], max_coordinates[2], n_points
    ).reshape((-1, 1))

    noise = np.concatenate((noise_x, noise_y, noise_z), axis=1)

    return np.concatenate((cloud, noise), axis=0)


def perturb_cloud(cloud: np.array, deg=0.2, mu=0, std=0.1, n_points=500):
    """Modify input cloud
    :param cloud: initial cloud
    :param deg: degrees of random rotation
    :param mu: mean of random translation
    :param std: standard deviation of random translation
    :param n_points: number of noise points
    :return: cloud modified, Rotation, translation, scale distortion
    """

    T = generate_initialization_matrix(deg=deg, mu=mu, std=std)
    R, t = T[:3, :3], T[:3, 3]

    cloud = np.dot(cloud, R) + t.reshape((1, 3))
    cloud = add_noise(cloud, n_points=n_points)

    scale_distortion = 1 + np.abs(np.random.randn(3))
    cloud *= scale_distortion

    return cloud, R, t, scale_distortion


def evaluate_pipe(R_0, t_0, T_star):
    """Evaluate performances of rotation matrix

    :param R_0: Initial transformation applied to target cloud
    :param R_star: Final transformation obtained
    """

    rotation_diff = np.linalg.norm(R_0 - T_star[:3, :3].T)
    translation_diff = np.linalg.norm(t_0 - T_star[:3, 3].ravel())

    print(f"Rotation error: {rotation_diff}")

    print("Element wise difference:")
    print("Rotation:\n", R_0 - T_star[:3, :3].T, "\n")


def evaluate_distortion(source_cloud, target_cloud, scale_star, T_star, dist_target_ds):
    """

    :param source_cloud: nuvola source con tutti i punti (no sottocampionata), ma normalizzata
    :param target_cloud: target con tutti i punti, senza rumore, perturbata, ricentrata
    :param scale_star:
    :param T_star:
    :param dist_target_ds:
    :return:
    """

    source_rotated = np.dot(source_cloud, T_star[:3, :3]) + T_star[:3, 3].reshape(
        (1, 3)
    )
    source_rescaled = dist_target_ds * (source_rotated / scale_star)

    RMSE = np.linalg.norm(source_rescaled - target_cloud) / np.sqrt(len(source_rotated))

    print("RMSE is: ", RMSE)
