import itertools
from tabulate import tabulate
from typing import Callable, Dict

import numpy as np

from qiskit import Aer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.extensions import Initialize
from qiskit.quantum_info import Operator, Statevector
from qiskit.result import Result

import src.utils as utils


def encode(
    init_function: Callable[[int], Statevector], 
    data_function: Callable[[int, int, int, np.ndarray], Statevector], 
    map_function: Callable[[int, int, int, np.ndarray], Operator], 
    image: np.ndarray,
    ctx: Dict = {}
) -> QuantumCircuit:
    R = np.ceil(np.log2(np.max(image.shape))).astype(int)

    products, data_vectors, map_operators = [], [], []
    for u, v in itertools.product(range(image.shape[0]), range(image.shape[1])):
        data_vector = data_function(u, v, R, image)
        map_operator = map_function(u, v, R, image)
        product = data_vector.to_operator() ^ map_operator
        products.append(product)
        data_vectors.append(data_vector)
        map_operators.append(map_operator)

        if ctx.get("verbose", 0) > 1:
            print(f"{u=}, {v=}")
            print(f"{data_vector=}")
            print(f"{map_operator=}")
            print(f"{product=}")
            print("===========")

    G = np.sum(products, axis=0)
    U, _ = np.linalg.qr(G)
    if ctx.get("verbose"):
        print(f"G=\n{utils.tabulate_complex(G)}")
        print(f"U=\n{utils.tabulate_complex(U)}")
    
    U_op = Operator(U)
    n_qubits = U_op.num_qubits
    init_state = init_function(n_qubits)
    if ctx.get("verbose"):
        print(f"{init_state=}")

    circuit = QuantumCircuit(n_qubits)
    if not np.all(init_state.data == 1):
        circuit.append(Initialize(init_state, normalize=True), range(n_qubits))
    circuit.append(U_op, range(n_qubits))
    circuit.measure_all()

    print(circuit.draw())

    return circuit


def simulate(circuit: QuantumCircuit, n_shots: int, return_qiskit_result: bool = False) -> Dict[str, int] | Result:
    simulator = Aer.get_backend('aer_simulator')
    result = simulator.run(circuit, shots=n_shots, memory=True).result()
    if return_qiskit_result:
        return result

    counts = result.get_counts(circuit)
    counts_padded = {f"{n:0{circuit.num_qubits}b}": 0 for n in range(2**circuit.num_qubits)}
    counts_padded = {**counts_padded, **counts}
    return counts_padded
