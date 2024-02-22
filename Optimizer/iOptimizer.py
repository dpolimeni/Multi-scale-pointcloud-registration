from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class IOptimizer(ABC):

    @abstractmethod
    def optimize(self, source: np.ndarray, target: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        """Method to run the inner block optimizer.

        Args:
            source (numpy array): source point-cloud array
            target (numpy array): target point-cloud array
            **kwargs : Any keywords arguments necessary to run the optimizer

        Returns:
            Tuple[np.ndarray, float]: A tuple with the Roto-translation matrix and inlier RMSE of the solution
        """
        pass
