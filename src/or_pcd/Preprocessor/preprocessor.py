from typing import List

import numpy as np
from or_pcd.Preprocessor.iProcessBlock import IProcessBlock
from or_pcd.Preprocessor.Scalers.BaseScaler import BaseScaler
from or_pcd.Preprocessor.Scalers import RadiusScaler

from or_pcd.utils.logger_factory import LoggerFactory


class Preprocessor:
    def __init__(self, preprocessor_blocks: List[IProcessBlock]):
        """Preprocessor class to preprocess the input cloud using the preprocessor blocks
        Scalers block should be always applied
        """

        self._LOG = LoggerFactory.get_logger(self.__class__.__name__, log_on_file=False)
        # Check if the Scalers class is among the preprocessor blocks
        if not any(issubclass(block.__class__, BaseScaler) for block in preprocessor_blocks):
            self._LOG.warning(
                """Scalers block is not present in the preprocessor blocks --> Adding Scalers block to the preprocessor blocks"""
            )
            preprocessor_blocks.insert(0, RadiusScaler())
        self.preprocessor_blocks = preprocessor_blocks

    def preprocess(self, cloud: np.ndarray) -> np.ndarray:
        """Preprocess the input cloud using the preprocessor blocks. sequentially"""
        # Preprocess data
        for block in self.preprocessor_blocks:
            cloud = block.process(cloud)
        return cloud
