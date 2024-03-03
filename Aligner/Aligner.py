import copy
from typing import Tuple
from Preprocessor.preprocessor import Preprocessor
from Optimizer.iOptimizer import IOptimizer
import numpy as np
from typing import List
import multiprocessing
import time

from utils.logger_factory import LoggerFactory


class Aligner:
    def __init__(
        self,
        source_preprocessor: Preprocessor,
        target_preprocessor: Preprocessor,
        optimizer: IOptimizer,
        n_attempts: int = 100,
        deg=0.2,
        mu=0,
        std=0.1,
        delta=0.1,
        max_iter=100,
        eps=1e-6,
    ):
        """
        :param source_preprocessor: Source cloud preprocessor
        :param target_preprocessor: Target cloud preprocessor
        :param optimizer: iOptimizer inherited class to run a single optimization
        :param n_attempts: number of multi-start attempts
        :param deg: initial rotation matrix angle
        :param mu: initial translation vector mean
        :param std: initial translation vector standard deviation
        """

        self._LOG = LoggerFactory.get_logger(log_name=self.__class__.__name__, log_on_file=True)

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
        translation = self.mu + np.random.randn(3) * self.std

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

        for n in range(self.n_attempts):  # tqdm
            source_copy = copy.deepcopy(source)

            # Generate random rotation and translation matrices
            initial_rotation, initial_translation = self.initialize_rotation()
            source_initialized = (
                np.dot(source_copy, initial_rotation) + initial_translation
            )

            # Perform registration
            current_rotation, current_metric = self.optimizer.optimize(
                source_initialized, target
            )

            if current_metric < metric:
                # Update metric
                metric = current_metric
                # Define Transformation
                T = np.eye(4)
                T[:3, :3] = np.dot(current_rotation[:3, :3], initial_rotation)
                T[:3, 3] = (
                    np.dot(current_rotation[:3, :3], initial_translation).ravel()
                    + initial_translation
                )
                best_transformation = T

        return best_transformation, metric

    def _worker(self, n: multiprocessing.Queue, source, target, result_queue):
        # print('Starting process', n)
        source_copy = copy.deepcopy(source)

        # Generate random rotation and translation matrices
        initial_rotation, initial_translation = self.initialize_rotation()
        source_initialized = np.dot(source_copy, initial_rotation) + initial_translation

        # Perform registration
        current_rotation, current_metric = self.optimizer.optimize(
            source_initialized, target
        )

        # Prepare result
        result = (
            current_rotation,
            initial_rotation,
            initial_translation,
            current_metric,
        )

        result_queue.put((n, result))
        # print('Process', n, 'finished')

    def parallel_multistart_registration(
        self, source: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Perform multistart registration on the source and target point clouds.
        :param source: Source point cloud
        :param target: Target point cloud

        :return: (best_transformation, metric): (best transformation matrix, best metric)
        """

        # Initial metric and transformation
        metric = np.inf
        best_transformation = np.eye(4)

        processes = []
        result_queue = multiprocessing.Queue()
        num_cores = multiprocessing.cpu_count()
        # print('Number of cores:', num_cores)
        # print('Number of attempts:', self.n_attempts)
        for n in range(self.n_attempts):
            process = multiprocessing.Process(
                target=self._worker, args=(n, source, target, result_queue)
            )
            processes.append(process)
            process.start()
            # TODO improve checks on processes
            if len(processes) == num_cores:
                # print('Joining processes')
                for process in processes:
                    # print('Joining process', process.pid)
                    process.join()
                # print('Processes joined')
                processes = []
        for process in processes:
            process.join()

        # Collect results
        while not result_queue.empty():
            _, (
                current_rotation,
                initial_rotation,
                initial_translation,
                current_metric,
            ) = result_queue.get()
            # print(f'Process {_} finished with RMSE:', current_metric)

            if current_metric < metric:
                metric = current_metric
                T = np.eye(4)
                T[:3, :3] = np.dot(current_rotation[:3, :3], initial_rotation)
                T[:3, 3] = (
                    np.dot(current_rotation[:3, :3], initial_translation).ravel()
                    + initial_translation
                )
                best_transformation = T

        return best_transformation, metric

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
        current_rotation, current_metric = self.multistart_registration(source, target_scaled)
        # current_rotation, current_metric = self.parallel_multistart_registration(
        #     source, target
        # )

        return new_scale_factors, current_rotation, current_metric

    def align(
        self, source: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, float, List[float]]:
        source = self.source_preprocessor.preprocess(source)
        target = self.target_preprocessor.preprocess(target)
        iteration = 0

        optimal_scale_factors = np.ones((1, 3))

        start = time.time()
        optimal_transformation, optimal_metric = self.multistart_registration(
            source, target
        )
        # optimal_transformation, optimal_metric = self.parallel_multistart_registration(source, target)
        print("Multistart registration time:", time.time() - start)

        # INITIALIZE ERRORS LIST
        errors = [optimal_metric]
        print("rmse star", optimal_metric)
        directions = np.eye(3)

        while self.delta >= self.eps and iteration <= self.max_iter:
            # UPDATE ITERATION COUNTER
            iteration += 1
            print("Iteration number:", iteration, "Current step:", self.delta)
            for axis in range(3):
                scale_plus = self.delta * directions[:, axis]
                new_scale_factors, new_rotation, new_metric = self.compass_step(
                    source, target, optimal_scale_factors, scale_plus
                )

                if new_metric <= optimal_metric:
                    print("New RMSE:", new_metric, "Old RMSE:", optimal_metric)
                    optimal_metric = new_metric
                    optimal_transformation = new_rotation
                    optimal_scale_factors = scale_plus
                    errors.append(new_metric)
                    break

                # SCALE TARGET CLOUD
                scale_neg = -self.delta * directions[:, axis]
                new_scale_factors, new_rotation, new_metric = self.compass_step(
                    source, target, optimal_scale_factors, scale_neg
                )

                if new_metric <= optimal_metric:
                    print("New RMSE:", new_metric, "Old RMSE:", optimal_metric)
                    optimal_metric = new_metric
                    optimal_transformation = new_rotation
                    optimal_scale_factors = scale_neg
                    errors.append(new_metric)
                    break
            # TODO (ADD try/except?)
            # UPDATE DELTA
            if new_metric > optimal_metric:
                self.delta = self.delta / 2
                print("Delta updated:", self.delta)

        return optimal_transformation, optimal_metric, errors
