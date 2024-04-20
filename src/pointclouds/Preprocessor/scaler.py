import numpy as np
from scipy.spatial.distance import cdist

from pointclouds.Preprocessor.iProcessBlock import IProcessBlock
from pointclouds.utils.logger_factory import LoggerFactory


class Scaler(IProcessBlock):
    """
    Scales point cloud by centering it by its mean and dividing by the radius of the cloud
    """

    def __init__(self):
        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=False
        )

    def process(self, cloud: np.ndarray) -> np.ndarray:
        self._LOG.debug("Normalizing clouds")
        center = np.mean(cloud, axis=0, keepdims=True)
        radius = np.max(cdist(center, cloud))

        return (cloud - center) / radius
