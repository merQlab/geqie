import numpy as np
from qiskit.quantum_info import Statevector, Operator

from .. import ensure_2d
from .init import init as init_function
from .data import data as _data_function
from .map import map as _map_function
from .retrieve import retrieve as retrieve_function


def data_function(u: int, v: int, R: int, image: np.ndarray) -> Statevector:
    filtered_image = ensure_2d(image)
    return _data_function(u, v, R, filtered_image)


def map_function(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    filtered_image = ensure_2d(image)
    return _map_function(u, v, R, filtered_image)
