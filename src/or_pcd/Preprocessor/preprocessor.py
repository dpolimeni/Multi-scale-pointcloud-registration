from typing import List

import numpy as np
from or_pcd.Preprocessor.iProcessBlock import IProcessBlock
from or_pcd.Preprocessor.scaler import Scaler
from or_pcd.utils.logger_factory import LoggerFactory


class Preprocessor:
    def __init__(self, preprocessor_blocks: List[IProcessBlock]):
        """Preprocessor class to preprocess the input cloud using the preprocessor blocks
        Scaler block should be always applied
        """

        self._LOG = LoggerFactory.get_logger(self.__class__.__name__, log_on_file=False)
        # Check if the Scaler class is among the preprocessor blocks
        if not any(isinstance(block, Scaler) for block in preprocessor_blocks):
            self._LOG.warning(
                """Scaler block is not present in the preprocessor blocks --> Adding Scaler block to the preprocessor blocks"""
            )
            preprocessor_blocks.insert(-1, Scaler())
        self.preprocessor_blocks = preprocessor_blocks

    def preprocess(self, cloud: np.ndarray) -> np.ndarray:
        """Preprocess the input cloud using the preprocessor blocks. sequentially"""
        # Preprocess data
        for block in self.preprocessor_blocks:
            cloud = block.process(cloud)
        return cloud
