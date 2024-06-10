import numpy as np
from scipy.spatial.distance import cdist

from or_pcd.Preprocessor.iProcessBlock import IProcessBlock
from or_pcd.utils.logger_factory import LoggerFactory


class BaseScaler(IProcessBlock):
    """
    Scales point cloud by centering it by its mean and dividing by the radius of the cloud
    """
    mean: float = 0.0
    scale: float = 1.0

    def __init__(self):
        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=False
        )