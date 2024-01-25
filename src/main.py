import importlib
import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import scipy as sp

from qiskit import Aer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.result import Result


ENCODINGS_PATH = Path(__file__).parent / "encodings"


def import_module(name: str, file_path: str):
    path = ENCODINGS_PATH / file_path
    spec = importlib.util.spec_from_file_location(name, path.absolute())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_all_encodings() -> List[str]:
    return sorted(dir.stem for dir in ENCODINGS_PATH.glob("*"))


def encode(data_function: Callable, map_function: Callable, image: np.ndarray) -> QuantumCircuit:
    arrays = []
    data_vectors = []
    map_vectors = []
    for u, v in itertools.product(range(image.shape[0]), range(image.shape[1])):
        data_vector = data_function(u, v, image)
        map_vector = map_function(u, v, image)
        product = data_vector ^ map_vector
        arrays.append(product.data.reshape(product.dims()))
        # debug
        data_vectors.append(data_vector)
        map_vectors.append(map_vector)

    G = sp.linalg.block_diag(*arrays)
    U = np.linalg.qr(G, mode="complete").Q
    
    op = Operator(U)
    n_qubits = op.num_qubits

    circuit = QuantumCircuit(n_qubits)
    [circuit.x(q) for q in range(n_qubits)]
    circuit.append(op, range(n_qubits))
    circuit.measure_all()

    print(circuit.draw())

    return circuit


def simulate(circuit: QuantumCircuit, n_shots: int) -> Result:
    simulator = Aer.get_backend('aer_simulator')
    circuit = transpile(circuit, simulator)
    # print(circuit.draw())

    result = simulator.run(circuit, shots=n_shots, memory=True).result()

    return result