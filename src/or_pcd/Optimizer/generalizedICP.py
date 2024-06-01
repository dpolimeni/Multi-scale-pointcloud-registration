from typing import Tuple

import numpy as np
import open3d as o3d

from or_pcd.Optimizer.iOptimizer import IOptimizer
from or_pcd.utils.constants import (
    __MAXIMUM_CORRESPONDENCE_DISTANCE__,
    __MAX_ITERATIONS__,
)
from or_pcd.utils.logger_factory import LoggerFactory


class GeneralizedICP(IOptimizer):
    def __init__(
        self,
        max_correspondence_distance: float = __MAXIMUM_CORRESPONDENCE_DISTANCE__,
        max_iterations: int = __MAX_ITERATIONS__,
    ):
        """
        Refer to this link: https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf for parameters explanation
        :param max_correspondence_distance:
        :param max_iterations:
        """

        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=False
        )
        if max_correspondence_distance <= 0:
            msg = f"max correspondence distance cannot be 0 or less. Provided: {max_correspondence_distance}"
            self._LOG.warning(msg)
            self._max_correspondence_distance = __MAXIMUM_CORRESPONDENCE_DISTANCE__
        else:
            self._max_correspondence_distance = max_correspondence_distance

        if max_iterations <= 0:
            msg = f"max iterations cannot be 0 or less. Provided: {max_iterations}"
            self._LOG.warning(msg)
            self._max_iterations = __MAX_ITERATIONS__
        else:
            self._max_iterations = max_iterations

        self._LOG.debug(
            f"Initialized {self.__class__.__name__} with parameters: {self}"
        )

    def optimize(
        self, source: np.ndarray, target: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, float]:
        icp_type = (
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        )

        source_point_cloud = o3d.geometry.PointCloud()
        target_point_cloud = o3d.geometry.PointCloud()
        source_point_cloud.points = o3d.utility.Vector3dVector(source)
        target_point_cloud.points = o3d.utility.Vector3dVector(target)

        optimization_result: o3d.pipelines.registration.RegistrationResult = (
            o3d.pipelines.registration.registration_generalized_icp(
                source_point_cloud,
                target_point_cloud,
                self._max_correspondence_distance,
                np.eye(4),
                icp_type,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self._max_iterations
                ),
            )
        )

        roto_translation = np.copy(optimization_result.transformation)
        transposed_rotation = roto_translation[:3, :3].T
        roto_translation[:3, :3] = transposed_rotation

        # If the rmse is 0 raise an error as parameters are not correct
        if optimization_result.inlier_rmse == 0:
            msg = """Optimization failed with loss = 0. Parameters are not well set. 
        Probably due to maximum_correspondence_distance set too low."""
            self._LOG.error(msg)
            raise ValueError(msg)

        return roto_translation, optimization_result.inlier_rmse

    def __repr__(self):
        return f"""{self.__class__.__name__}
            (max_correspondence_distance={self._max_correspondence_distance},
            max_iterations={self._max_iterations})
        """
