import itertools
from typing import Any, Callable, Dict

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.result import Result
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.quantum_info import SuperOp, Kraus

from geqie.utils.print import tabulate_complex


def encode(
    init_function: Callable[[int], Statevector],
    # arbitrary coordinate indices + R + image
    data_function: Callable[..., Statevector],
    map_function: Callable[..., Operator],
    image: np.ndarray,
    verbosity_level: int = 0,
    **_: Dict[Any, Any],
) -> QuantumCircuit:
    shape = image.shape

    # treat the last axis as channels if it's small (1, 3, 4)
    if image.ndim >= 3 and shape[-1] in (1, 3, 4):
        spatial_shape = shape[:-1]   # e.g. RGB - extracting only (u, v) from (u, v, 3)
    else:
        spatial_shape = shape        # pure spatial: (u, v), (u, v, w) etc.

    R = int(np.ceil(np.log2(max(spatial_shape))))

    products, data_vectors, map_operators = [], [], []
    for coords in np.ndindex(*spatial_shape):
        # coords = (u, v) for 2D, (d, u, v) for 3D, etc.
        data_vector = data_function(*coords, R=R, image=image)
        map_operator = map_function(*coords, R=R, image=image)
        product = data_vector.to_operator() ^ map_operator

        products.append(product)
        data_vectors.append(data_vector)
        map_operators.append(map_operator)

        if verbosity_level > 2:
            print(f"{coords=}")
            print(f"{data_vector=}")
            print(f"{map_operator=}")
            print(f"{product=}")
            print("===========")

    G = np.sum(products, axis=0)
    U, _ = np.linalg.qr(G)
    if verbosity_level > 1:
        print(f"G=\n{tabulate_complex(G)}")
        print(f"U=\n{tabulate_complex(U)}")
    
    U_op = Operator(U)
    n_qubits = U_op.num_qubits
    init_state = init_function(n_qubits)
    if verbosity_level > 1:
        print(f"{init_state=}")

    circuit = QuantumCircuit(n_qubits)
    circuit.initialize(init_state, range(n_qubits), normalize=True)
    circuit.append(U_op, range(n_qubits))
    circuit.measure_all()

    if verbosity_level > 0:
        print(circuit.draw())

    return circuit


def simulate(
    circuit: QuantumCircuit, 
    n_shots: int, 
    return_qiskit_result: bool = False,
    return_padded_counts: bool = False,
    device: str = "CPU",
    method: str = "automatic",
    noise_model: NoiseModel | None = None,
    **_: Dict[Any, Any],
) -> Dict[str, int] | Result:	
    
    simulator = AerSimulator(device=device, method=method)

    result = simulator.run(circuit, shots=n_shots, memory=True, noise_model=noise_model).result()
    if return_qiskit_result:
        return result

    counts = result.get_counts(circuit)

    if return_padded_counts:
        counts_padded = {f"{n:0{circuit.num_qubits}b}": 0 for n in range(2**circuit.num_qubits)}
        return {**counts_padded, **counts}
    else:
        return counts
