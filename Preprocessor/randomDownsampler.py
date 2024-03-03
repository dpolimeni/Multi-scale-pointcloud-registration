import numpy as np

from Preprocessor.iProcessBlock import IProcessBlock


class RandomDownsampler(IProcessBlock):
    def __init__(self, sample_size: int, replace: bool = False):
        self.sample_size = sample_size
        self.replace = replace

    def process(self, cloud: np.ndarray) -> np.ndarray:
        # Downsample data
        return cloud[
            np.random.choice(cloud.shape[0], self.sample_size, replace=self.replace)
        ]
