import copy

from Preprocessor.preprocessor import Preprocessor
from Optimizer.iOptimizer import IOptimizer
import numpy as np
class Aligner:
    def __init__(self, source_preprocessor: Preprocessor, target_preprocessor: Preprocessor,
                 optimizer: IOptimizer, n_attempts: int = 100):
        self.source_preprocessor = source_preprocessor
        self.target_preprocessor = target_preprocessor
        self.optimizer = optimizer
        self.n_attempts = n_attempts

    def generate_initialization_matrix(self, deg=0.2, mu=0, std=0.1):
        # Initialize transformation matrix
        T = np.eye(4)
        # Generate random angles
        theta_1 = np.random.uniform(low=-deg, high=deg)
        theta_2 = np.random.uniform(low=-deg, high=deg)
        theta_3 = np.random.uniform(low=-deg, high=deg)
        # Create the 3 basic rotation matrices
        R1 = np.array([[1, 0, 0], [0, np.cos(theta_1), -np.sin(theta_1)], [0, np.sin(theta_1), np.cos(theta_1)]])
        R2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2)], [0, 1, 0], [-np.sin(theta_2), 0, np.cos(theta_2)]])
        R3 = np.array([[np.cos(theta_3), -np.sin(theta_3), 0], [np.sin(theta_3), np.cos(theta_3), 0], [0, 0, 1]])
        # Aggregate them to form the final rotation matrix
        R = np.dot(R1, np.dot(R2, R3))
        # Generate random translation vector
        t = mu + np.random.randn(3) * std
        # Update T
        T[:3, :3] = R  # Place rotation matrix in the upper left
        T[:3, 3] = t  # Place the translation vector in the upper right

        return T

    def multistart_registration(self, source: np.ndarray, target: np.ndarray):
        # Number of iterations for the multistart
        # Initial metric and transformation
        metric = np.inf
        best_T = np.eye(4)

        for n in range(self.n_attempts):  # tqdm
            source_copy = copy.deepcopy(source)
            # generate random rotation and translation matrices
            mat_init = self.generate_initialization_matrix(deg=1.57, mu=0, std=0.1)
            # Apply transformation to the matrix
            source_copy.transform(mat_init)
            # Do fast_global_registration
            if reg_type == 'fgr':
                result = execute_fast_global_registration(source_copy, target, source_fpfh, target_fpfh,
                                                          tuple_scale, distance_threshold=distance_threshold)
            if reg_type == 'gicp':
                result = execute_generalized_icp_registration(source=source_copy, target=target)
            # Check if improvement
            if result.inlier_rmse < metric:
                # Update metric
                metric = result.inlier_rmse
                # print(metric)
                # Find T
                T = np.eye(4)
                T[:3, :3] = np.dot(result.transformation[:3, :3], mat_init[:3, :3])
                T[:3, 3] = np.dot(result.transformation[:3, :3], mat_init[:3, 3]).ravel() + result.transformation[:3, 3]
                best_T = T

        return best_T, metric

    def align(self, source: np.ndarray, target: np.ndarray):
        source = self.source_preprocessor.preprocess(source)
        target = self.target_preprocessor.preprocess(target)
        pass


