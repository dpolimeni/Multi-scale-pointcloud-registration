import os
import time
from datetime import datetime
from pathlib import Path
import math

# TIMESTAMP (COMPUTED ONCE FOR LOGGERS INITIALIZATION)
__TIME_NOW__ = time.time()

# TIMESTAMP EXPRESSED IN HUMAN READABLE FORMAT
__DATETIME_NOW__ = formatted_time = datetime.fromtimestamp(__TIME_NOW__).strftime("%Y_%m_%d_%H_%M_%S")

# PROJECT ROOT
__ROOT__ = str(Path(os.getcwd()).parent)

# LOG DEFAULT FOLDER
__LOG_FOLDER__ = os.path.join(__ROOT__, "logs", __DATETIME_NOW__)

# LOG DEFAULT FORMAT
__DEFAULT_LOG_FORMAT__ = "%(asctime)s | %(name)s | %(levelname)s : %(message)s"

# ------------------------------------------------------------------------------------------------------- #
# OUTLIER REMOVAL
# ------------------------------------------------------------------------------------------------------- #
# SOR
__NB_NEIGHBOURS__ = 64
"""
Number of neighbours used by SOR algorithm (to compute mean and variance)
"""
__STD_RATIO__ = 2
"""
SOR parameter. By increasing it, more points are kept (as inliers)
"""
# ------------------------------------------------------------------------------------------------------- #

# DOWN SAMPLERS
# RANDOM DOWN SAMPLER
__RANDOM_SAMPLE_SIZE__ = 15000

# FARTHEST DOWN SAMPLER
__FARTHEST_SAMPLE_SIZE__ = 10000

# VOXEL DOWN SAMPLER
__VOXEL_SAMPLE_SIZE__ = 2000
__BASE_VOXEL_SIZE__ = 0.01
__MIN_VOXEL_SIZE__ = 0.0001
__DELTA__ = 0.01
__EPS__ = 0.0005

# DOWNSAMPLER DEFAULT SAMPLE SIZE
__SAMPLE_SIZE__ = 4096

# FAST GLOBAL OPTIMIZER
__DIVISION_FACTOR__ = 1.4
__TUPLE_SCALE__ = 0.9
__ITERATION_NUMBER__ = 100
__DECREASE_MU__ = True
__NORMAL_ESTIMATE_RADIUS__ = 0.1
__NORMAL_ESTIMATE_KNN__ = 20
__FPFH_RADIUS__ = 0.1
__FPFH_KNN__ = 20

# GENERALIZED ICP
__MAX_ITERATIONS__ = 100

# OPTIMIZER GENERAL PARAMS
__MAXIMUM_CORRESPONDENCE_DISTANCE__ = 0.5


# ALIGNER
__MULTISTART_ATTEMPTS__ = 30
__ALIGNER_DEG__ = math.pi / 2
__ALIGNER_MU__ = 0.0
__ALIGNER_STD__ = 0.1
__ALIGNER_MAX_ITER__ = 100
__ALIGNER_DELTA__ = 0.2
__ALIGNER_EPS__ = 0.05

# REFINER
__REFINER_MAX_ITER__ = 200
__REFINER_DISTANCE_THRESHOLD__ = 0.5
