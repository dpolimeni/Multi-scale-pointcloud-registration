import copy
from typing import Tuple

import numpy as np
import open3d as o3d

from or_pcd.Optimizer.iOptimizer import IOptimizer
from or_pcd.utils.constants import (
    __DIVISION_FACTOR__,
    __TUPLE_SCALE__,
    __MAXIMUM_CORRESPONDENCE_DISTANCE__,
    __ITERATION_NUMBER__,
    __DECREASE_MU__,
    __FPFH_RADIUS__,
    __FPFH_KNN__,
    __NORMAL_ESTIMATE_RADIUS__,
    __NORMAL_ESTIMATE_KNN__,
)
from or_pcd.utils.logger_factory import LoggerFactory


class FastGlobalOptimizer(IOptimizer):
    def __init__(
        self,
        division_factor: float = __DIVISION_FACTOR__,
        tuple_scale: float = __TUPLE_SCALE__,
        maximum_correspondence_distance: float = __MAXIMUM_CORRESPONDENCE_DISTANCE__,
        iteration_number: int = __ITERATION_NUMBER__,
        decrease_mu: bool = __DECREASE_MU__,
        normal_estimate_radius: float = __NORMAL_ESTIMATE_RADIUS__,
        normal_estimate_knn: int = __NORMAL_ESTIMATE_KNN__,
        fpfh_radius: float = __FPFH_RADIUS__,
        fpfh_knn: int = __FPFH_KNN__,
    ):
        """
        Refer to this link: https://vladlen.info/papers/fast-global-registration.pdf for parameters explanation
        :param division_factor:
        :param tuple_scale:
        :param maximum_correspondence_distance:
        :param iteration_number:
        :param decrease_mu:
        :param fpfh_radius:
        :param fpfh_knn:
        :param fpfh_radius_features:
        """
        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=False
        )
        if division_factor <= 0:
            msg = f"division factor cannot be 0 or less. Provided: {division_factor}"
            self._LOG.error(msg)
            self._division_factor = __DIVISION_FACTOR__
        else:
            self._division_factor = division_factor

        if tuple_scale <= 0:
            msg = f"tuple scale cannot be 0 or less. Provided: {tuple_scale}"
            self._LOG.error(msg)
            self._tuple_scale = __TUPLE_SCALE__
        else:
            self._tuple_scale = tuple_scale

        if maximum_correspondence_distance <= 0:
            msg = f"maximum correspondence distance cannot be 0 or less. Provided: {maximum_correspondence_distance}"
            self._LOG.error(msg)
            self._maximum_correspondence_distance = __MAXIMUM_CORRESPONDENCE_DISTANCE__
        else:
            self._maximum_correspondence_distance = maximum_correspondence_distance

        if iteration_number <= 0:
            msg = f"iteration number cannot be 0 or less. Provided: {iteration_number}"
            self._LOG.error(msg)
            self._iteration_number = __ITERATION_NUMBER__
        else:
            self._iteration_number = iteration_number

        self._decrease_mu = decrease_mu

        if fpfh_radius <= 0:
            msg = f"fpfh radius cannot be 0 or less. Provided: {fpfh_radius}"
            self._LOG.error(msg)
            self._fpfh_radius = __FPFH_RADIUS__
        else:
            self._fpfh_radius = fpfh_radius

        if fpfh_knn <= 0:
            msg = f"fpfh knn cannot be 0 or less. Provided: {fpfh_knn}"
            self._LOG.error(msg)
            self._fpfh_knn = __FPFH_KNN__
        else:
            self._fpfh_knn = fpfh_knn

        if normal_estimate_radius <= 0:
            msg = f"normal estimate radius cannot be 0 or less. Provided: {normal_estimate_radius}"
            self._LOG.error(msg)
            self._normal_estimate_radius = __NORMAL_ESTIMATE_RADIUS__
        else:
            self._normal_estimate_radius = normal_estimate_radius

        if normal_estimate_knn <= 0:
            msg = f"normal estimate knn cannot be 0 or less. Provided: {normal_estimate_knn}"
            self._LOG.error(msg)
            self._normal_estimate_knn = __NORMAL_ESTIMATE_KNN__
        else:
            self._normal_estimate_knn = normal_estimate_knn

        self._LOG.debug(msg=f"initialized optimizer: {self}")

    def get_fpfh_features(
        self,
        source_point_cloud: o3d.geometry.PointCloud,
        target_point_cloud: o3d.geometry.PointCloud,
    ) -> Tuple[o3d.pipelines.registration.Feature, o3d.pipelines.registration.Feature]:
        source_copy = copy.deepcopy(source_point_cloud)
        target_copy = copy.deepcopy(target_point_cloud)

        # Estimate Normals
        source_copy.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._normal_estimate_radius, max_nn=self._normal_estimate_knn
            )
        )
        target_copy.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._normal_estimate_radius, max_nn=self._normal_estimate_knn
            )
        )

        # Estimate FPFH Features
        source_fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(
            source_copy,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._fpfh_radius, max_nn=self._fpfh_knn
            ),
        )

        target_fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(
            source_copy,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._fpfh_radius, max_nn=self._fpfh_knn
            ),
        )

        return source_fpfh_features, target_fpfh_features

    def optimize(
        self, source: np.ndarray, target: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, float]:
        source_point_cloud = o3d.geometry.PointCloud()
        target_point_cloud = o3d.geometry.PointCloud()
        source_point_cloud.points = o3d.utility.Vector3dVector(source)
        target_point_cloud.points = o3d.utility.Vector3dVector(target)

        source_fpfh_features, target_fpfh_features = self.get_fpfh_features(
            source_point_cloud, target_point_cloud
        )

        fast_global_option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            division_factor=self._division_factor,
            tuple_scale=self._tuple_scale,
            maximum_correspondence_distance=self._maximum_correspondence_distance,
            iteration_number=self._iteration_number,
            decrease_mu=self._decrease_mu,
        )

        optimization_result: o3d.pipelines.registration.RegistrationResult = (
            o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_point_cloud,
                target_point_cloud,
                source_fpfh_features,
                target_fpfh_features,
                fast_global_option,
            )
        )

        roto_translation = np.copy(optimization_result.transformation)
        transposed_rotation = roto_translation[:3, :3].T
        roto_translation[:3, :3] = transposed_rotation

        # If the rmse is 0 raise an error as parameters are not correct
        if len(optimization_result.correspondence_set) == 0:
            msg = """No correspondences detected from the optimizer. Parameters are not well set. 
        Probably due to: 
        -   maximum_correspondence_distance set too low
        -   tuple_scale set too high
        """
            self._LOG.error(msg)
            raise Warning(msg)

        return roto_translation, optimization_result.inlier_rmse

    def __repr__(self):
        return f"""{self.__class__.__name__}
            (division_factor={self._division_factor},
            tuple_scale={self._tuple_scale},
            maximum_correspondence_distance={self._maximum_correspondence_distance},
            iteration_number={self._iteration_number},
            decrease_mu={self._decrease_mu},
            normal_estimate_radius={self._normal_estimate_radius},
            normal_estimate_knn={self._normal_estimate_knn},
            fpfh_radius={self._fpfh_radius},
            fpfh_knn={self._fpfh_knn})"
        """
