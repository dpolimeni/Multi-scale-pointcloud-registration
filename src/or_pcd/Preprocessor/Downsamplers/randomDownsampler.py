import numpy as np

from or_pcd.Preprocessor.iProcessBlock import IProcessBlock
from or_pcd.utils.constants import __SAMPLE_SIZE__
from or_pcd.utils.logger_factory import LoggerFactory


class RandomDownsampler(IProcessBlock):
    def __init__(self, sample_size: int = __SAMPLE_SIZE__, replace: bool = False):
        """
        :param sample_size: the number of points to keep from the cloud
        :param replace: choose to sample with or without replacement
        """
        self._LOG = LoggerFactory.get_logger(
            log_name=self.__class__.__name__, log_on_file=False
        )

        if sample_size <= 0:
            msg = f"sample size cannot be 0 or less. Provided: {sample_size}. Using default value: {__SAMPLE_SIZE__}"
            self._LOG.warning(msg)
            self._sample_size = __SAMPLE_SIZE__
        else:
            self._sample_size = sample_size

        self._replace = replace

        self._LOG.debug(f"Initialized {self.__class__.__name__} with {self}")

    def process(self, cloud: np.ndarray) -> np.ndarray:
        """
        :param cloud: cloud to down-sample
        :return: randomly down-sampled cloud
        """
        # Downsample data
        return cloud[
            np.random.choice(cloud.shape[0], self._sample_size, replace=self._replace)
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}(sample_size={self._sample_size}, replace={self._replace})"
