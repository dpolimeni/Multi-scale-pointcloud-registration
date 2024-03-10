from typing import List

import numpy as np

from src.Preprocessor.iProcessBlock import IProcessBlock


class Preprocessor:
    def __init__(self, preprocessor_blocks: List[IProcessBlock]):
        self.preprocessor_blocks = preprocessor_blocks

    def preprocess(self, cloud: np.ndarray) -> np.ndarray:
        """Preprocess the input cloud using the preprocessor blocks. sequentially"""
        # Preprocess data
        for block in self.preprocessor_blocks:
            cloud = block.process(cloud)
        return cloud
