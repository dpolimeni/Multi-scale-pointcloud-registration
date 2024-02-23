import copy
from typing import Tuple
from Preprocessor.preprocessor import Preprocessor
from Optimizer.iOptimizer import IOptimizer
import numpy as np


class Aligner:
    def __init__(self, source_preprocessor: Preprocessor, target_preprocessor: Preprocessor,
                 optimizer: IOptimizer, n_attempts: int = 100, deg=0.2, mu=0, std=0.1, delta=0.1, max_iter=100, eps=1e-6):
        """
        :param source_preprocessor: Source cloud preprocessor
        :param target_preprocessor: Target cloud preprocessor
        :param optimizer: iOptimizer inherited class to run a single optimization
        :param n_attempts: number of multistart attempts
        :param deg: iniitial rotation matrix angle
        :param mu: initial translation vector mean
        :param std: initial translation vector standard deviation
        """
        self.source_preprocessor = source_preprocessor
        self.target_preprocessor = target_preprocessor
        self.optimizer = optimizer
        self.delta = delta
        self.max_iter = max_iter
        self.eps = eps
        self.mu = mu
        self.std = std
        self.deg = deg
        self.n_attempts = n_attempts

    def initialize_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random transformation matrix for a single multi start registration run."""

        # Generate random angles
        theta_1 = np.random.uniform(low=-self.deg, high=self.deg)
        theta_2 = np.random.uniform(low=-self.deg, high=self.deg)
        theta_3 = np.random.uniform(low=-self.deg, high=self.deg)

        # Create the 3 basic rotation matrices
        r_1 = np.array([[1, 0, 0], [0, np.cos(theta_1), -np.sin(theta_1)], [0, np.sin(theta_1), np.cos(theta_1)]])
        r_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2)], [0, 1, 0], [-np.sin(theta_2), 0, np.cos(theta_2)]])
        r_3 = np.array([[np.cos(theta_3), -np.sin(theta_3), 0], [np.sin(theta_3), np.cos(theta_3), 0], [0, 0, 1]])

        # Aggregate them to form the final rotation matrix
        rotation_matrix = np.dot(r_1, np.dot(r_2, r_3))

        # Generate random translation vector
        translation = self.mu + np.random.randn(3) * self.std

        return rotation_matrix, translation

    def multistart_registration(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform multistart registration on the source and target point clouds.
        :param source: Source point cloud
        :param target: Target point cloud

        :return: (best_transformation, metric): (best transformation matrix, best metric)
        """

        # Initial metric and transformation
        metric = np.inf
        best_transformation = np.eye(4)

        for n in range(self.n_attempts):  # tqdm
            source_copy = copy.deepcopy(source)

            # Generate random rotation and translation matrices
            initial_rotation, initial_translation = self.initialize_rotation()
            source_initialized = np.dot(source_copy, initial_rotation) + initial_translation

            # Perform registration
            current_rotation, current_metric = self.optimizer.optimize(source_initialized, target)

            if current_metric < metric:
                # Update metric
                metric = current_metric
                # Define Transformation
                T = np.eye(4)
                T[:3, :3] = np.dot(current_rotation[:3, :3], initial_rotation)
                T[:3, 3] = np.dot(current_rotation[:3, :3], initial_translation).ravel() + initial_translation
                best_transformation = T

        return best_transformation, metric

    def align(self, source: np.ndarray, target: np.ndarray):
        source = self.source_preprocessor.preprocess(source)
        target = self.target_preprocessor.preprocess(target)
        iteration = 0

        optimal_scale_factors = np.ones((1, 3))
        optimal_transformation = np.eye(4)

        optimal_transformation, optimal_metric = self.multistart_registration(source, target)

        # INITIALIZE ERRORS LIST
        errors = [optimal_metric]
        print('rmse star', optimal_metric)
        directions = np.eye(3)

        while self.delta >= self.eps and iteration <= self.max_iter:
            # UPDATE COUNTER
            iteration += 1
            print('Iteration number:', iteration, 'Current step:', self.delta)
            for j in range(3):
                coeff_plus = coeff_star + delta * directions[:, 2 - j]

                # SCALE TARGET CLOUD
                pcd_target.points = o3d.utility.Vector3dVector(target_array * coeff_plus)
                # RETRIEVE FPFH
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_target,
                                                                              o3d.geometry.KDTreeSearchParamHybrid(
                                                                                  radius=0.1, max_nn=100))
                # RUN MULTISTART
                T_plus, rmse_plus = execute_multistart_registration(source, pcd_target, source_fpfh, target_fpfh,
                                                                    n_attempts=n_attempts, distance_threshold=0.8,
                                                                    tuple_scale=0.9, reg_type=reg_type)
                if rmse_plus <= rmse_star:
                    print('New RMSE:', rmse_plus, 'Old RMSE:', rmse_star)
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
                target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_target,
                                                                              o3d.geometry.KDTreeSearchParamHybrid(
                                                                                  radius=0.1, max_nn=100))
                # RUN MULTISTART
                T_neg, rmse_neg = execute_multistart_registration(source, pcd_target, source_fpfh, target_fpfh,
                                                                  n_attempts=n_attempts, distance_threshold=0.8,
                                                                  tuple_scale=0.9, reg_type=reg_type)

                if rmse_neg <= rmse_star:
                    print('New RMSE', rmse_neg, 'Old RMSE', rmse_star)
                    rmse_star = rmse_neg
                    T_star = T_neg
                    coeff_star = coeff_neg
                    # print('TOP COEFF', coeff_star)
                    errors.append(rmse_star)
                    break

            if rmse_plus > rmse_star and rmse_neg > rmse_star:
                delta = delta / 2

        return T_star, coeff_star, rmse_star, errors



