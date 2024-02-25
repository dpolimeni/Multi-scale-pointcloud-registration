from Preprocessor.iProcessBlock import IProcessBlock
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import open3d as o3d


class VoxelDownsampler(IProcessBlock):
    def __init__(self, target_points: int, base_voxel_size: float = 0.1,
                 min_voxel_size: float = 0.005, delta: float = 0.05, eps: float = 0.001):
        self.target_points = target_points
        self.min_voxel_size = min_voxel_size
        # TODO evaluate if is better create another type of class to estimate the voxel size
        self.base_voxel_size = base_voxel_size
        self.delta = delta
        self.eps = eps

    def compass_step(self, delta: float, cloud: o3d.geometry.PointCloud, current_voxel_size: float):
        new_voxel_size = current_voxel_size + delta
        if new_voxel_size <= self.min_voxel_size:
            new_voxel_size = self.min_voxel_size
        cloud_ds = cloud.voxel_down_sample(new_voxel_size)
        n_points = np.asarray(cloud_ds.points).shape[0]
        metric = np.abs(self.target_points - n_points)
        return cloud_ds, new_voxel_size, metric

    def process(self, cloud: np.ndarray):
        """Get the voxel size of the cloud needed for target sample size"""
        current_voxel_size = self.base_voxel_size
        source_point_cloud = o3d.geometry.PointCloud()
        source_point_cloud.points = o3d.utility.Vector3dVector(cloud)

        # Downsample first run
        obtained_cloud = source_point_cloud.voxel_down_sample(current_voxel_size)
        # Get number of points
        n_points = np.asarray(obtained_cloud.points).shape[0]
        # Compute Metric
        metric = np.abs(self.target_points - n_points)

        while self.delta >= self.eps:
            obtained_cloud, new_voxel_size, obtained_metric = self.compass_step(self.delta,
                                                                                source_point_cloud,
                                                                                current_voxel_size)

            if obtained_metric < metric:
                current_voxel_size = new_voxel_size
                metric = obtained_metric
                continue

            obtained_cloud, new_voxel_size, obtained_metric = self.compass_step(-self.delta,
                                                                                source_point_cloud,
                                                                                current_voxel_size)

            if obtained_metric < metric:
                current_voxel_size = new_voxel_size
                metric = obtained_metric
                continue

            self.delta = self.delta / 2

        return np.asarray(obtained_cloud.points)


