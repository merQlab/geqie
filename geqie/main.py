from typing import Any, Callable, Dict

import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import Operator, Statevector

import geqie.ibm_qp as ibm_qp
from geqie.logging.logger import setup_logger
from geqie.logging.tabulate import tabulate_complex


def encode(
    init_function: Callable[[int], Statevector],
    # arbitrary coordinate indices + R + image
    data_function: Callable[..., Statevector],
    map_function: Callable[..., Operator],
    image: np.ndarray,
    image_dimensionality: int = 2,
    verbosity_level: int | None = None,
    **_: Dict[Any, Any],
) -> QuantumCircuit:
    logger = setup_logger(verbosity_level, reset=True)

    shape = image.shape[:image_dimensionality]

    R = int(np.ceil(np.log2(max(shape))))

    products, data_vectors, map_operators = [], [], []

    for coords in np.ndindex(*shape):
        data_vector = data_function(*coords, R=R, image=image)
        map_operator = map_function(*coords, R=R, image=image)
        product = data_vector.to_operator() ^ map_operator

        products.append(product)
        data_vectors.append(data_vector)
        map_operators.append(map_operator)

        logger.state(f"{coords=}")
        logger.state(f"{data_vector=}")
        logger.state(f"{map_operator=}")
        logger.state(f"{product=}")
        logger.state("===========")

    G = np.sum(products, axis=0)
    U, _ = np.linalg.qr(G)
    logger.math(f"G=\n{tabulate_complex(G)}")
    logger.math(f"U=\n{tabulate_complex(U)}")

    U_op = Operator(U)
    n_qubits = U_op.num_qubits
    init_state = init_function(n_qubits)
    logger.state(f"{init_state=}")

    circuit = QuantumCircuit(n_qubits)
    circuit.initialize(init_state, range(n_qubits), normalize=True)
    circuit.append(U_op, range(n_qubits))
    circuit.measure_all()

    logger.debug(circuit.draw())

    return circuit


def simulate(
    circuit: QuantumCircuit, 
    n_shots: int, 
    return_qiskit_result: bool = False,
    return_padded_counts: bool = False,
    device: str = "CPU",
    method: str = "automatic",
    noise_model: NoiseModel | None = None,
    verbosity_level: int | None = None,
    **_: Dict[Any, Any],
) -> Result | Dict[str, int]:
    logger = setup_logger(verbosity_level, reset=True)

    simulator = AerSimulator(device=device, method=method)
    
    logger.debug("Simulating circuit...")
    result = simulator.run(circuit, shots=n_shots, memory=True, noise_model=noise_model).result()
    logger.debug("Simulation completed.")
    if return_qiskit_result:
        return result

    counts = result.get_counts(circuit)

    if return_padded_counts:
        logger.debug("Padding counts...")
        counts_padded = {f"{n:0{circuit.num_qubits}b}": 0 for n in range(2**circuit.num_qubits)}
        return {**counts_padded, **counts}
    else:
        return counts


def execute(
    circuit: QuantumCircuit, 
    n_shots: int,
    circuit_param_values: Dict[str, Any] = {},
    return_qiskit_result: bool = False,
    return_padded_counts: bool = False,
    dry_run: bool = False,
    api_token: str = "",
    instance_crn: str = "",
    credentials_name: str = "",
    backend: str = "",
    channel: str = "ibm_quantum_platform",
    ibm_qp_runtime_args: Dict[str, Any] = {},
    transpiler_args: Dict[str, Any] = {},
    verbosity_level: int | None = None,
    **_: Dict[Any, Any],
) -> Result | Dict[str, int] | None:
    logger = setup_logger(verbosity_level, reset=True)

    logger.info("Setting up IBM Quantum backend...")
    ibm_qp_backend = ibm_qp.get_ibm_quantum_backend(
        api_token=api_token,
        instance_crn=instance_crn,
        credentials_name=credentials_name,
        backend=backend,
        channel=channel,
        iqp_runtime_args=ibm_qp_runtime_args,
        min_num_qubits=circuit.num_qubits,
    )

    pass_manager = generate_preset_pass_manager(
        backend=ibm_qp_backend, 
        translation_method="translator",
        **transpiler_args,
    )
    logger.info("Circuit transpilation...")
    transpiled_circuit = pass_manager.run(circuit)
    logger.info("Circuit transpilation. Done.")
    logger.trace(transpiled_circuit.draw())

    sampler = ibm_qp.get_sampler(ibm_qp_backend)

    if dry_run:
        logger.info("Dry run mode. Exiting before job submission.")
        return None

    logger.info(f"Submitting job to backend '{ibm_qp_backend.name}'...")
    job = sampler.run([(transpiled_circuit, circuit_param_values)], shots=n_shots)
    logger.debug(f"{job.job_id()=}")

    result = job.result()
    logger.info("Job completed.")
    logger.debug(f"{job.metrics()=}")

    if return_qiskit_result:
        return result

    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    if return_padded_counts:
        counts_padded = {f"{n:0{circuit.num_qubits}b}": 0 for n in range(2**circuit.num_qubits)}
        return {**counts_padded, **counts}
    else:
        return counts
