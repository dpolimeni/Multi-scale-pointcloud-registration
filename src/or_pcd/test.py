import numpy as np
import open3d as o3d

from or_pcd.utils.util_mcs_registration import generate_initialization_matrix

# Create a 5x3 array
source = np.random.randn(5, 3)

source_cloud = o3d.geometry.PointCloud()
source_cloud.points = o3d.utility.Vector3dVector(source)

T = generate_initialization_matrix(deg=1.57, std=0.1)

rotation_matrix = T[:3, :3]
translaton_vector = T[:3, 3]

# Transform with Open3D
source_cloud.transform(T)
trans_o3d = np.asarray(source_cloud.points)

# Transform with normal numpy
trans_normal = np.dot(source, rotation_matrix.T) + translaton_vector

print("Open3D: \n", trans_o3d, "\n\nNormal\n", trans_normal)
