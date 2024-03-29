from Preprocessor.iProcessBlock import IProcessBlock
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


class FarthesDownsampler(IProcessBlock):
    def __init__(self, sample_size: int):
        self.sample_size = sample_size

    def process(self, cloud: np.ndarray):
        # Create a list of indexes of the points that will be kept
        index_list = [0] * self.sample_size
        # The first point is chosen randomly
        index_list[0] = np.random.randint(low=0, high=cloud.shape[0])

        # Create a distance matrix with all points far away
        distances = np.ones((1, cloud.shape[0])) * 1e6

        # Iterate over the sample size and choose the farthest point
        for i in tqdm(range(self.sample_size - 1)):
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