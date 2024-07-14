import numpy as np
from numpy import ndarray
from or_pcd.Preprocessor.iProcessBlock import IProcessBlock
from or_pcd.utils.logger_factory import LoggerFactory
import open3d as o3d

from or_pcd.utils.constants import (
    __NB_NEIGHBOURS__,
    __STD_RATIO__,
)


class SOR(IProcessBlock):
    """
    Statistical outlier removal.

    This class implements the statistical outlier removal (SOR) algorithm, which is used to filter Outliers from point cloud data.

    Args:
        nb_neighbours (int): Number of neighbours used by the SOR algorithm to compute mean and variance.
        std_ratio (float): SOR parameter. By increasing it, more points are kept as inliers.
    """

    def __init__(
        self, nb_neighbours: int = __NB_NEIGHBOURS__, std_ratio: float = __STD_RATIO__
    ) -> None:
        """
        Initialize the SOR object with the specified parameters.

        Args:
            nb_neighbours (int): Number of neighbours used by the SOR algorithm to compute mean and variance.
            std_ratio (float): SOR parameter. By increasing it, more points are kept as inliers.
        """

        super().__init__()

        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=True
        )

        self._nb_neighbours = nb_neighbours
        """
        Number of neighbours used by SOR algorithm (to compute mean and variance)
        """

        self._std_ratio = std_ratio
        """
        SOR parameter. By increasing it, more points are kept (as inliers)
        """

    def process(self, cloud: ndarray) -> ndarray:
        """
        Filters Outliers from the input point cloud using the SOR algorithm and returns only the inliers.
        Since the implementation provided by open3d is used, a temporary pointcluoud is created with
        the provided array

        Args:
            cloud (ndarray): Input point cloud data.

        Returns:
            ndarray: Filtered point cloud containing only the inliers.
        """

        # Create a temporary point cloud to use open3d method
        _temp = o3d.geometry.PointCloud()
        _temp.points = o3d.utility.Vector3dVector(np.copy(a=cloud))

        # Remove Outliers
        inliers, ind = _temp.remove_statistical_outlier(
            nb_neighbors=self._nb_neighbours, std_ratio=self._std_ratio
        )
        inliers = np.asarray(inliers.points)

        self._LOG.debug(
            msg=f"Cloud before SOR had {int(cloud.shape[0])} points. After SOR has {inliers.shape[0]} points!"
        )

        # Return only inliers
        return inliers

    @property
    def nb_neighbours(self) -> int:
        """
        Getter method for the number of neighbours parameter.

        Returns:
            int: Number of neighbours used by the SOR algorithm.
        """
        return self._nb_neighbours

    @property
    def std_ratio(self) -> float:
        """
        Getter method for the standard deviation ratio parameter.

        Returns:
            float: Standard deviation ratio used by the SOR algorithm.
        """
        return self._std_ratio

    def __repr__(self):
        return f"{self.__class__.__name__}(nb_neighbours={self._nb_neighbours}, std_ratio={self._std_ratio})"
