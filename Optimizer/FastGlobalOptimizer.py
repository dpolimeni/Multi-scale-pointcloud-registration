import copy
from typing import Tuple

import numpy as np
import open3d as o3d
from Optimizer.iOptimizer import IOptimizer


class FastGlobalOptimizer(IOptimizer):
    def __init__(self, division_factor: float, tuple_scale: float, maximum_correspondence_distance: float,
                 iteration_number: int, decrease_mu: bool, fpfh_radius: float, fpfh_knn: int, fpfh_radius_features: int):
        # TODO (comment the init variables and refer to the article do not even remember some of them)
        self.division_factor = division_factor
        self.tuple_scale = tuple_scale
        self.maximum_correspondence_distance = maximum_correspondence_distance
        self.iteration_number = iteration_number
        self.decrease_mu = decrease_mu
        self.fpfh_radius = fpfh_radius
        self.fpfh_knn = fpfh_knn # TODO differentiate among normal estimate and fpfh extraction
        self.fpfh_radius_features = fpfh_radius_features

    def optimize(self, source: np.ndarray, target: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:

        # TODO create an util function to make fpfh
        source_point_cloud = o3d.geometry.PointCloud()
        target_point_cloud = o3d.geometry.PointCloud()
        source_point_cloud.points = o3d.utility.Vector3dVector(source)
        target_point_cloud.points = o3d.utility.Vector3dVector(target)

        source_copy = copy.deepcopy(source_point_cloud)
        target_copy = copy.deepcopy(target_point_cloud)

        # Estimate Normals
        source_copy.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius, max_nn=self.fpfh_knn)
        )
        target_copy.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius, max_nn=self.fpfh_knn)
        )

        # Estimate FPFH Features
        source_fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(
            source_copy,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius_features, max_nn=self.fpfh_knn)
        )

        target_fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(
            source_copy,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius_features, max_nn=self.fpfh_knn)
        )


        fast_global_option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            division_factor=self.division_factor, tuple_scale=self.tuple_scale,
            maximum_correspondence_distance=self.maximum_correspondence_distance,
            iteration_number=self.iteration_number, decrease_mu=self.decrease_mu
        )

        optimization_result: o3d.pipelines.registration.RegistrationResult = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_point_cloud, target_point_cloud, source_fpfh_features, target_fpfh_features, fast_global_option
        )

        return optimization_result.transformation, optimization_result.inlier_rmse
