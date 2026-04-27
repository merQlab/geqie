from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector, Operator

from .. import ensure_grayscale
from .init import init as init_function
from .data import data as _data_function
from .map import map as _map_function
from .retrieve import retrieve as retrieve_function


def data_function(u: int, v: int, R: int, image: np.ndarray, grayscale_strategy: str | None = None, **encoding_args: Any) -> Statevector:
    filtered_image = ensure_grayscale(image, grayscale_strategy)
    return _data_function(u, v, R, filtered_image, **encoding_args)


def map_function(u: int, v: int, R: int, image: np.ndarray, grayscale_strategy: str | None = None, **encoding_args: Any) -> Operator:
    filtered_image = ensure_grayscale(image, grayscale_strategy)
    return _map_function(u, v, R, filtered_image, **encoding_args)
