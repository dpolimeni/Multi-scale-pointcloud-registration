import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from or_pcd.Preprocessor.iProcessBlock import IProcessBlock
from or_pcd.utils.constants import __SAMPLE_SIZE__
from or_pcd.utils.logger_factory import LoggerFactory


class FarthestDownsampler(IProcessBlock):
    def __init__(self, sample_size: int = __SAMPLE_SIZE__):
        """
        :param sample_size: size of the down-sampled cloud
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

    def process(self, cloud: np.ndarray):
        """
        :param cloud: cloud to down-sample
        :return: cloud downsampled with farthest point algorithm refer to pointnet ++ paper for explanation
        link: https://arxiv.org/abs/1706.02413
        """
        # Create a list of indexes of the points that will be kept
        index_list = [0] * self._sample_size
        # The first point is chosen randomly
        index_list[0] = np.random.randint(low=0, high=cloud.shape[0])

        # Create a distance matrix with all points far away
        distances = np.ones((1, cloud.shape[0])) * 1e6

        # Iterate over the sample size and choose the farthest point
        for i in tqdm(range(self._sample_size - 1)):
            # Get the current point
            point = cloud[index_list[i]].reshape((1, cloud.shape[1]))

            # Calculate the distance between the current point and all the others
            point_distances = cdist(point, cloud)
            # Get the minimum distance between the current distances and the new ones
            distances = np.min((point_distances, distances), axis=0)

            # Get the farthest point among all distances
            fp_index = np.argmax(distances)
            index_list[i + 1] = fp_index

        return cloud[index_list]

    def __repr__(self):
        return f"{self.__class__.__name__}(sample_size={self._sample_size})"
