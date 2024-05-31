import numpy as np
import open3d as o3d

from or_pcd.Preprocessor.iProcessBlock import IProcessBlock
from or_pcd.utils.constants import (
    __BASE_VOXEL_SIZE__,
    __MIN_VOXEL_SIZE__,
    __DELTA__,
    __EPS__,
)
from or_pcd.utils.logger_factory import LoggerFactory


class VoxelDownsampler(IProcessBlock):
    """
    Reduces point cloud dimension by creating a voxel of the space. To select the appropriate voxel size, it uses
    a pattern search algorithm that, given the desired number of points, finds the appropriate voxel size
    """

    def __init__(
        self,
        target_points: int,
        base_voxel_size: float = __BASE_VOXEL_SIZE__,
        min_voxel_size: float = __MIN_VOXEL_SIZE__,
        delta: float = __DELTA__,
        eps: float = __EPS__,
    ):
        """
        :param target_points: desired number of points after down sampling
        :param base_voxel_size:
        :param min_voxel_size: minimum voxel size that can be used. It must be greater than 0
        :param delta:
        :param eps:
        """

        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=False
        )

        if target_points <= 0:
            msg = f"target points cannot be 0 or less. Provided: {target_points}"
            self._LOG.error(msg)
            raise ValueError(msg)

        self._target_points = target_points

        if min_voxel_size <= 0:
            msg = f"min voxel size cannot be 0 or less. Provided: {min_voxel_size}. Using default value: {__MIN_VOXEL_SIZE__}"
            self._LOG.warning(msg)
            self._min_voxel_size = __MIN_VOXEL_SIZE__
        else:
            self._min_voxel_size = min_voxel_size

        if base_voxel_size <= 0:
            msg = f"base voxel size cannot be 0 or less. Provided: {base_voxel_size}. Using default value: {__BASE_VOXEL_SIZE__}"
            self._LOG.warning(msg)
            self._base_voxel_size = __BASE_VOXEL_SIZE__
        else:
            self._base_voxel_size = base_voxel_size

        if delta <= 0:
            msg = f"delta cannot be 0 or less. Provided: {delta}. Using default value: {__DELTA__}"
            self._LOG.warning(msg)
            self._delta = __DELTA__
        else:
            self._delta = delta

        if eps <= 0:
            msg = f"eps cannot be 0 or less. Provided: {eps}. Using default value: {__EPS__}"
            self._LOG.warning(msg)
            self._eps = __EPS__
        else:
            self._eps = eps

        self._LOG.debug(msg=f"initialized downsampler: {self}")

    def compass_step(
        self, delta: float, cloud: o3d.geometry.PointCloud, current_voxel_size: float
    ):
        new_voxel_size = current_voxel_size + delta
        if new_voxel_size <= self._min_voxel_size:
            new_voxel_size = self._min_voxel_size
        cloud_ds = cloud.voxel_down_sample(new_voxel_size)
        n_points = np.asarray(cloud_ds.points).shape[0]
        metric = np.abs(self._target_points - n_points)
        return cloud_ds, new_voxel_size, metric

    def process(self, cloud: np.ndarray):
        """Get the voxel size of the cloud needed for target sample size"""
        current_voxel_size = self._base_voxel_size
        source_point_cloud = o3d.geometry.PointCloud()
        source_point_cloud.points = o3d.utility.Vector3dVector(cloud)

        # Downsample first run
        best_cloud = source_point_cloud.voxel_down_sample(current_voxel_size)
        # Get number of points
        n_points = np.asarray(best_cloud.points).shape[0]
        # Compute Metric
        metric = np.abs(self._target_points - n_points)

        while self._delta >= self._eps:
            obtained_cloud, new_voxel_size, obtained_metric = self.compass_step(
                self._delta, source_point_cloud, current_voxel_size
            )

            if obtained_metric < metric:
                current_voxel_size = new_voxel_size
                metric = obtained_metric
                best_cloud = obtained_cloud
                continue

            obtained_cloud, new_voxel_size, obtained_metric = self.compass_step(
                -self._delta, source_point_cloud, current_voxel_size
            )

            if obtained_metric < metric:
                current_voxel_size = new_voxel_size
                metric = obtained_metric
                best_cloud = obtained_cloud
                continue

            self._delta = self._delta / 2

        # Log the voxel size and the number of points
        self._LOG.debug(msg=f"Voxel size: {current_voxel_size}, Metric: {metric}")
        return np.asarray(best_cloud.points)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"target_points={self._target_points}, "
            f"base_voxel_size={self._base_voxel_size}, "
            f"min_voxel_size={self._min_voxel_size}, "
            f"delta={self._delta}, "
            f"eps={self._eps})"
        )
