import numpy as np
import open3d as o3d


def create_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Shortcut to create an open3d point cloud from a numpy array
    """

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud
