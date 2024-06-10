import numpy as np
from scipy.spatial.distance import cdist

from or_pcd.Preprocessor.Scalers.BaseScaler import BaseScaler
from or_pcd.utils.logger_factory import LoggerFactory


class RadiusScaler(BaseScaler):
    """
    Scales point cloud by centering it by its mean and dividing by the radius of the cloud
    """

    def __init__(self):
        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=False
        )

    def process(self, cloud: np.ndarray) -> np.ndarray:
        """
        IprocessBlock implementation that saves mean and scale of the cloud
        :param cloud: point-cloud to process
        :return: processed cloud
        """
        self._LOG.debug("Normalizing clouds")
        center = np.mean(cloud, axis=0, keepdims=True)
        radius = np.max(cdist(center, cloud))
        self.mean = center
        self.scale = radius

        return (cloud - center) / radius
