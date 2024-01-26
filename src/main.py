import importlib
import importlib.util
import itertools
import sys
from pathlib import Path
from tabulate import tabulate
from typing import Callable, Dict, List

import numpy as np
import scipy as sp

from qiskit import Aer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, Kraus
from qiskit.result import Result
from qiskit_aer.noise import QuantumError


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


def encode(data_function: Callable, map_function: Callable, image: np.ndarray, ctx: Dict = {}) -> QuantumCircuit:
    R = np.ceil(np.log2(np.max(image.shape))).astype(int)

    products = []
    data_vectors = []
    map_vectors = []
    for u, v in itertools.product(range(image.shape[0]), range(image.shape[1])):
        data_vector = data_function(u, v, R, image)
        map_vector = map_function(u, v, R, image)
        product = data_vector ^ map_vector
        products.append(product)

        if ctx.get("verbose"):
            print(f"{u=}, {v=}")
            print(f"{data_vector=}")
            print(f"{map_vector=}")
            print(f"{product=}")
            data_vectors.append(data_vector)
            map_vectors.append(map_vector)
            print("===========")

    G = sp.linalg.block_diag(*np.sum(products, axis=0))
    U, S, Vh = np.linalg.svd(G)
    if ctx.get("verbose"):
        print(f"G=\n{tabulate(G)}")
        print(f"U=\n{tabulate(U)}")
        # print(f"S=\n{tabulate(S)}")
        print(f"Vh=\n{tabulate(Vh)}")

    G_op = Operator(G)
    U_op = Operator(U)
    S_op = Statevector(S).to_operator()
    Vh_op = Operator(Vh)
    
    n_qubits = U_op.num_qubits

    circuit = QuantumCircuit(n_qubits)
    [circuit.h(q) for q in range(n_qubits)]
    circuit.append(G_op, range(n_qubits))
    # circuit.append(U_op, range(n_qubits))
    # circuit.append(S_op, range(n_qubits))
    # circuit.append(Vh_op, range(n_qubits))
    circuit.measure_all()

    print(circuit.draw())

    return circuit


def simulate(circuit: QuantumCircuit, n_shots: int) -> Result:
    simulator = Aer.get_backend('aer_simulator')
    circuit = transpile(circuit, simulator)
    # print(circuit.draw())

    result = simulator.run(circuit, shots=n_shots, memory=True).result()

    return result
