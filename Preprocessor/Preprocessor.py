from  typing import List
from Preprocessor.iProcessBlock import IProcessBlock
import numpy as np
class Preprocessor:
    def __init__(self, preprocessor_blocks: List[IProcessBlock]):
        self.preprocessor_blocks = preprocessor_blocks

    def preprocess(self, cloud: np.ndarray) -> np.ndarray:
        # Preprocess data
        for block in self.preprocessor_blocks:
            cloud = block.process(cloud)
        return cloud