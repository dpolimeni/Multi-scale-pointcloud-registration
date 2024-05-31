import open3d as o3d
import pkg_resources
import numpy as np

def load_sample_cloud(name: str = "ArmadilloBack_180") -> np.ndarray:
    """
    Load a sample cloud from the or_pcd/data directory

    Args:
        name (str): Name of the cloud to load

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud
    """
    path = name + ".ply"
    data_path = pkg_resources.resource_stream(__name__, path)
    cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(data_path.name)
    # Return the array of the cloud
    return np.asarray(cloud.points)