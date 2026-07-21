import logging
import os

from .layer import VQCLayer, MatrixDataset
from .layers import UnitaryInputLayer, SamplerAnsatzLayer

from .dataset.zip_unitary_dataset import load_precomputed_zip_matrices, ZipUnitaryDataset
from .precompute import compute_and_save_circuits


LOGGER_FORMAT = "%(levelname)s %(asctime)s --- %(message)s (%(filename)s:%(lineno)d)"


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level), format=LOGGER_FORMAT)
