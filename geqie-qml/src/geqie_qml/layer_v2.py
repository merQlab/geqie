import numpy as np
import qiskit
from qiskit.circuit import Gate, ParameterVector


class GEQIELayer(qiskit.QuantumCircuit):
    def __init__(self, num_qubits: int):
        """
        A quantum circuit consisting of a single parameterized unitary gate over all qubits.

        Each of the `(2**num_qubits)**2` matrix entries is represented as one complex `ParameterExpression`. The underlying `(2**num_qubits)**2` real Parameters are the `input_params` of a `SamplerQNN`: their values are the parts of a precomputed unitary matrix supplied by the data loader at inference time.

        Usage with `SamplerQNN`::

            pqc = qiskit.QuantumCircuit(num_qubits)
            geqie_layer = GEQIELayer(num_qubits)
            ansatz = ...  # some parameterized circuit with trainable weights
            qnn = SamplerQNN(
                circuit=pqc.compose(
                    geqie_layer, 
                    ansatz,
                    inplace=True,
                ),
                input_params=geqie_layer.input_params,
                weight_params=[],  # trainable weights to be provided by ansatz circuit
                ...
            )
        """
        super().__init__(num_qubits)

        gate = ParameterizedUnitaryGate(num_qubits)
        self.append(gate, self.qubits)


class ParameterizedUnitaryGate(Gate):
    """
    A unitary gate whose matrix is determined by ``(2**num_qubits)**2``
    complex `ParameterExpression` objects — one per matrix entry in
    row-major order.

    ``__array__`` is called by the simulator only after all parameters are
    bound to concrete values via ``assign_parameters``.
    """

    def __init__(self, num_qubits: int):
        """Initialize the gate with the specified number of qubits."""
        n_params = (2**num_qubits) ** 2
        self.params = ParameterVector("u", length=n_params)
        super().__init__("unitary", num_qubits, self.params, "GEQIE")

    def __array__(self, dtype=None, copy=None):
        dim = 2 ** self.num_qubits
        matrix = np.array([complex(p) for p in self.params]).reshape(dim, dim)

        if dtype is not None:
            matrix = matrix.astype(dtype)
        return matrix

    def inverse(self):
        raise NotImplementedError("Inverse of ParameterizedUnitaryGate requires bound parameters.")
