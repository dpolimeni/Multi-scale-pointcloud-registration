import copy
from typing import Tuple

import numpy as np
import open3d as o3d
from Optimizer.iOptimizer import IOptimizer

class GeneralizedICP(IOptimizer):
    def __init__(self, max_correspondence_distance: float, max_iterations: int):
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations

    def optimize(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        icp_type = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

        source_point_cloud = o3d.geometry.PointCloud()
        target_point_cloud = o3d.geometry.PointCloud()
        source_point_cloud.points = o3d.utility.Vector3dVector(source)
        target_point_cloud.points = o3d.utility.Vector3dVector(target)

        optimization_result : o3d.pipelines.registration.RegistrationResult = o3d.pipelines.registration.registration_generalized_icp(
            source_point_cloud, target_point_cloud, self.max_correspondence_distance, np.eye(4),
            icp_type, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
        )
        return optimization_result.transformation, optimization_result.inlier_rmse
