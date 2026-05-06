import logging
import os

from .layer import VQCLayer, MatrixDataset
from .precompute import compute_and_save_circuits
from .qec import QECCode, BitFlipCode, PhaseFlipCode, ShorCode, SurfaceCode
from .noise_suppression import (
    PAULI_TWIRL_CNOT,
    PAULI_TWIRL_CZ,
    PAULI_TWIRL_ECR,
    twirl_two_qubit_gate,
    twirl_circuit,
    DD_SEQUENCES,
    make_dd_pass_manager,
    apply_dd,
)
LOGGER_FORMAT = "%(levelname)s %(asctime)s --- %(message)s (%(filename)s:%(lineno)d)"

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level), format=LOGGER_FORMAT)
