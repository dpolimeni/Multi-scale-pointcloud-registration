from abc import ABC, abstractmethod

import numpy as np


class IProcessBlock(ABC):
    """Abstract class for a single inner block of preprocessing pipeline."""

    @abstractmethod
    def process(self, cloud: np.ndarray) -> np.ndarray:
        """Processing function of the input cloud array."""
        pass
