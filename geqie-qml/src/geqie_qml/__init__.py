import logging
import os

from .layer import VQCLayer, MatrixDataset
from .precompute import compute_and_save_circuits

LOGGER_FORMAT = "%(levelname)s %(asctime)s --- %(message)s (%(filename)s:%(lineno)d)"

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level), format=LOGGER_FORMAT)
