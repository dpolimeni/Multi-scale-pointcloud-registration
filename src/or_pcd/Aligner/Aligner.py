import copy
import multiprocessing
import time
from typing import List
from typing import Tuple

import numpy as np
import open3d as o3d

from or_pcd.Optimizer.iOptimizer import IOptimizer
from or_pcd.Preprocessor.preprocessor import Preprocessor
from or_pcd.Preprocessor.Scalers.BaseScaler import BaseScaler
from or_pcd.Visualizer.Visualizer import (
    visualize_point_clouds,
    draw_registration_result,
)
from or_pcd.utils.constants import (
    __ALIGNER_DEG__,
    __ALIGNER_MU__,
    __ALIGNER_STD__,
    __ALIGNER_MAX_ITER__,
    __ALIGNER_DELTA__,
    __ALIGNER_EPS__,
    __MULTISTART_ATTEMPTS__,
    __REFINER_MAX_ITER__,
    __REFINER_DISTANCE_THRESHOLD__,
)
from or_pcd.utils.create_cloud import create_cloud
from or_pcd.utils.logger_factory import LoggerFactory


class Aligner:
    transfromation: np.ndarray = np.eye(4)
    scale_factors: np.ndarray = np.ones((1, 3))

    def __init__(
        self,
        source_preprocessor: Preprocessor,
        target_preprocessor: Preprocessor,
        optimizer: IOptimizer,
        attempts: int = __MULTISTART_ATTEMPTS__,
        deg: float = __ALIGNER_DEG__,
        mu: float = __ALIGNER_MU__,
        std: float = __ALIGNER_STD__,
        delta: float = __ALIGNER_DELTA__,
        max_iter: int = __ALIGNER_MAX_ITER__,
        eps: float = __ALIGNER_EPS__,
        visualize_intermediate_steps: bool = False,
    ):
        """
        Wrapper function that takes the inner block optimizer, the preprocessors and runs a multi-start optimization on
        the source and target cloud.
        :param source_preprocessor: Source cloud preprocessor
        :param target_preprocessor: Target cloud preprocessor
        :param optimizer: iOptimizer inherited class to run a single optimization
        :param attempts: number of multi-start attempts
        :param deg: initial rotation matrix angle
        :param mu: initial translation vector mean
        :param std: initial translation vector standard deviation
        :param delta: initial compass step
        :param max_iter: maximum number of iterations for the compass search
        :param eps: stopping criteria on delta for the compass search
        """

        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=True
        )

        if attempts <= 0:
            msg = f"attempts cannot be 0 or less. Provided: {attempts}"
            self._LOG.warning(msg)
            self._attempts = __MULTISTART_ATTEMPTS__
        else:
            self._attempts = attempts

        if deg <= 0:
            msg = f"deg cannot be 0 or less. Provided: {deg}"
            self._LOG.warning(msg)
            self._deg = __ALIGNER_DEG__
        else:
            self._deg = deg

        if mu < 0:
            msg = f"mu cannot be 0 or less. Provided: {mu}"
            self._LOG.warning(msg)
            self._mu = __ALIGNER_MU__
        else:
            self._mu = mu

        if std <= 0:
            msg = f"std cannot be 0 or less. Provided: {std}"
            self._LOG.warning(msg)
            self._std = __ALIGNER_STD__
        else:
            self._std = std

        if delta <= 0:
            msg = f"delta cannot be 0 or less. Provided: {delta}"
            self._LOG.warning(msg)
            self._delta = __ALIGNER_DELTA__
        else:
            self._delta = delta

        if max_iter <= 0:
            msg = f"max_iter cannot be 0 or less. Provided: {max_iter}"
            self._LOG.warning(msg)
            self._max_iter = __ALIGNER_MAX_ITER__
        else:
            self._max_iter = max_iter
        print("EPS", eps)
        if eps <= 0:
            msg = f"eps cannot be 0 or less. Provided: {eps}"
            self._LOG.warning(msg)
            self._eps = __ALIGNER_EPS__
        else:
            self._eps = eps

        self._visualize_intermediate_steps = visualize_intermediate_steps
        self._source_preprocessor = source_preprocessor
        self._target_preprocessor = target_preprocessor
        self._optimizer = optimizer

        self._LOG.debug(f"Initialized {self.__class__.__name__} with {self}")

    def initialize_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random transformation matrix for a single multi start registration run."""

        # Generate random angles
        theta_1 = np.random.uniform(low=-self._deg, high=self._deg)
        theta_2 = np.random.uniform(low=-self._deg, high=self._deg)
        theta_3 = np.random.uniform(low=-self._deg, high=self._deg)

        # Create the 3 basic rotation matrices
        r_1 = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta_1), -np.sin(theta_1)],
                [0, np.sin(theta_1), np.cos(theta_1)],
            ]
        )
        r_2 = np.array(
            [
                [np.cos(theta_2), 0, np.sin(theta_2)],
                [0, 1, 0],
                [-np.sin(theta_2), 0, np.cos(theta_2)],
            ]
        )
        r_3 = np.array(
            [
                [np.cos(theta_3), -np.sin(theta_3), 0],
                [np.sin(theta_3), np.cos(theta_3), 0],
                [0, 0, 1],
            ]
        )

        # Aggregate them to form the final rotation matrix
        rotation_matrix = np.dot(r_1, np.dot(r_2, r_3))

        # Generate random translation vector
        translation = self._mu + np.random.randn(3) * self._std

        return rotation_matrix, translation

    def multistart_registration(
        self, source: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Perform multi-start registration on the source and target point clouds.
        :param source: Source point cloud
        :param target: Target point cloud

        :return: (best_transformation, metric): (best transformation matrix, best metric)
        """

        # Initial metric and transformation
        metric = np.inf
        best_transformation = np.eye(4)

        for n in range(self._attempts):  # tqdm
            source_copy = copy.deepcopy(source)

            # Generate random rotation and translation matrices
            initial_rotation, initial_translation = self.initialize_rotation()
            source_initialized = (
                np.dot(source_copy, initial_rotation) + initial_translation
            )

            # Perform registration
            current_transform, current_metric = self._optimizer.optimize(
                source_initialized, target
            )

            if current_metric < metric:
                # Update metric
                metric = current_metric
                # Define Transformation
                T = np.eye(4)
                T[:3, :3] = np.dot(initial_rotation, current_transform[:3, :3])
                T[:3, 3] = (
                    np.dot(initial_translation, current_transform[:3, :3]).ravel()
                    + current_transform[:3, 3]
                )
                best_transformation = T

        return best_transformation, metric  # best_transformation, metric

    def compass_step(
        self,
        source: np.ndarray,
        target: np.ndarray,
        scale_factors: np.ndarray,
        delta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform a single compass step to find the optimal scaling factor for the target cloud.
        :param source: Source point cloud
        :param target: Target point cloud
        :param scale_factors: Current scaling factor
        :param delta: Step size for the scaling factor

        :return: (new_scale_factors, new_rotation, metric): (new scaling factor, obtsined new_rotation ans new metric)
        """
        new_scale_factors = scale_factors + delta
        target_scaled = target * new_scale_factors
        current_rotation, current_metric = self.multistart_registration(
            source, target_scaled
        )
        return new_scale_factors, current_rotation, current_metric

    def align(
        self,
        source: np.ndarray,
        target: np.ndarray,
        refine_registration: bool = True,
        icp_type: str = "PointToPlane",
    ) -> Tuple[np.ndarray, float, np.ndarray, List[float]]:
        """Align the source and target point clouds. using a two block optimization approach.
        :param source: Source point cloud
        :param target: Target point cloud
        :param refine_registration: Flag to enable/disable the refinement step that uses ICP
        :param icp_type: PointToPoint or PointToPlane ICP

        :return: (optimal_transformation, optimal_metric, optimal_scale_factors, errors): (Optimal transformation matrix, RMSE of the inliers, optimal scaling factor, list of errors)
        """
        source = self._source_preprocessor.preprocess(source)
        target = self._target_preprocessor.preprocess(target)
        iteration = 0

        if self._visualize_intermediate_steps:
            visualize_point_clouds([source, target], [(0, 0, 1), (1, 0, 0)])

        optimal_scale_factors = np.ones((1, 3))

        start = time.time()
        optimal_transformation, optimal_metric = self.multistart_registration(
            source, target
        )
        self._LOG.info(f"Multi-start registration time: {time.time() - start}")

        # INITIALIZE ERRORS LIST
        errors = [optimal_metric]
        self._LOG.info(f"Initial RMSE: {optimal_metric}")
        directions = np.eye(3)

        while self._delta >= self._eps and iteration <= self._max_iter:
            # UPDATE ITERATION COUNTER
            iteration += 1
            self._LOG.info(f"Iteration number: {iteration} Current step: {self._delta}")
            for axis in range(3):
                scale_plus = self._delta * directions[:, axis]
                new_scale_factors, new_rotation, new_metric = self.compass_step(
                    source, target, optimal_scale_factors, scale_plus
                )

                if new_metric <= optimal_metric:
                    self._LOG.info(f"New RMSE: {new_metric} Old RMSE: {optimal_metric}")
                    optimal_metric = new_metric
                    optimal_transformation = new_rotation
                    optimal_scale_factors += scale_plus
                    errors.append(new_metric)
                    break

                # SCALE TARGET CLOUD
                scale_neg = -self._delta * directions[:, axis]
                new_scale_factors, new_rotation, new_metric = self.compass_step(
                    source, target, optimal_scale_factors, scale_neg
                )

                if new_metric <= optimal_metric:
                    self._LOG.info(f"New RMSE: {new_metric} Old RMSE: {optimal_metric}")
                    optimal_metric = new_metric
                    optimal_transformation = new_rotation
                    optimal_scale_factors += scale_neg
                    errors.append(new_metric)
                    break
            # TODO (ADD try/except?)
            # UPDATE DELTA
            if new_metric > optimal_metric:
                self._delta = self._delta / 2
                self._LOG.info(f"No improvement, decreasing step size to {self._delta}")

        if refine_registration:
            target = target * optimal_scale_factors
            optimal_transformation, optimal_metric = self.refine_registration(
                source=source, target=target, initial_transform=optimal_transformation, icp_type=icp_type
            )
            errors.append(optimal_metric)

        if self._visualize_intermediate_steps:
            draw_registration_result(
                create_cloud(source),
                create_cloud(target * optimal_scale_factors),
                optimal_transformation,
            )

        self.transfromation = optimal_transformation
        self.scale_factors = optimal_scale_factors

        return optimal_transformation, optimal_metric, optimal_scale_factors, errors

    def refine_registration(
        self,
        source: np.ndarray,
        target: np.ndarray,
        initial_transform: np.ndarray,
        max_iteration: int = __REFINER_MAX_ITER__,
        distance_threshold: float = __REFINER_DISTANCE_THRESHOLD__,
        icp_type: str = "PointToPoint",
    ) -> (np.ndarray, float):
        """Refine the registration using ICP algorithm.
        :param source: Source point cloud
        :param target: Target point cloud
        :param initial_transform: Initial transformation matrix
        :param max_iteration: Maximum number of iterations for the ICP algorithm
        :param distance_threshold: Maximum distance between two correspondences
        :param icp_type: PointToPoint or PointToPlane ICP

        :return: (T_refined, result.inlier_rmse): (Refined transformation matrix, RMSE of the inliers)
        """
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

        source_cloud = create_cloud(points=source)
        target_cloud = create_cloud(points=target)

        # Execute ICP
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            distance_threshold,
            initial_transform,
            icp_type,
            convergence_rule,
        )

        T_refined = result.transformation

        return T_refined, result.inlier_rmse


    def transfrom(self, source: np.ndarray) -> np.ndarray:
        """Transform the source point cloud using the transformation matrix.
        :param source: Source point cloud
        :return: transformed_source: Transformed source point cloud to be aligned with the target
        """
        # Get the mean and distance of the source and target point clouds preprocessor
        for process_block in self._source_preprocessor.preprocessor_blocks:
            if issubclass(process_block.__class__, BaseScaler):
                source_mean = process_block.mean
                source_distance = process_block.scale
        for process_block in self._target_preprocessor.preprocessor_blocks:
            if issubclass(process_block.__class__, BaseScaler):
                target_mean = process_block.mean
                target_distance = process_block.scale

        # Normalize the source point cloud
        source = (source - source_mean) / source_distance

        # Rototranslate the source cloud as done in the align method
        source = np.dot(source, self.transfromation[:3, :3]) + self.transfromation[:3, 3]

        # Apply the inverse scale factors
        source = source / self.scale_factors

        # Denormalize the source point cloud with the target point cloud
        source = source * target_distance + target_mean

        return source

    def __repr__(self):
        return f"""{self.__class__.__name__}
            (source_preprocessor={self._source_preprocessor},
            target_preprocessor={self._target_preprocessor},
            optimizer={self._optimizer},
            attempts={self._attempts},
            deg={self._deg},
            mu={self._mu},
            std={self._std},
            delta={self._delta},
            max_iter={self._max_iter},
            eps={self._eps})
        """
