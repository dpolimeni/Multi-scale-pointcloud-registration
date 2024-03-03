import os
import time
from datetime import datetime
from pathlib import Path

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
