import numpy as np
import pytest
import torch
from qiskit.quantum_info import SparsePauliOp, random_statevector

from geqie_qml import PauliZExpectationLayer


def test_matches_qiskit_z_expectations_including_qubit_order():
    n_qubits = 4
    states = [random_statevector(2 ** n_qubits, seed=seed) for seed in range(3)]
    probs = torch.tensor(np.stack([state.probabilities() for state in states]))

    actual = PauliZExpectationLayer(n_qubits)(probs)

    expected = np.array([
        [state.expectation_value(SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], n_qubits)).real for i in range(n_qubits)]
        for state in states
    ])
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)


def test_accepts_leading_dimensions_and_rejects_wrong_size():
    layer = PauliZExpectationLayer(3)
    assert layer(torch.full((2, 5, 8), 1 / 8)).shape == (2, 5, 3)
    with pytest.raises(ValueError):
        layer(torch.ones(2, 4))
