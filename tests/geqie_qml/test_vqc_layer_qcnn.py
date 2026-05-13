from pathlib import Path
import sys

import numpy as np
import pytest


GEQIE_QML_SRC = Path(__file__).resolve().parents[2] / "geqie-qml" / "src"
sys.path.insert(0, str(GEQIE_QML_SRC))

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_machine_learning")
torch = pytest.importorskip("torch")

from geqie_qml.ansatze import QCNN_layer
from geqie_qml.layer import VQCLayer
import geqie_qml.layer as layer_module


def test_qcnn_layer_uses_requested_output_qubits():
    circuit = QCNN_layer(input_qubits=4, output_qubits=3)

    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 3


def test_vqc_layer_executes_qcnn_layer_with_configured_output_qubits(monkeypatch):
    ansatz_calls = []
    observed_circuits = []

    class FakeSampler:
        def __init__(self, default_shots):
            self.default_shots = default_shots

    class FakeGradient:
        def __init__(self, sampler):
            self.sampler = sampler

    class FakeSamplerQNN:
        def __init__(self, circuit, input_params, weight_params, sampler, gradient):
            observed_circuits.append(circuit)

        def forward(self, input_data, weights):
            return np.ones((1, 8), dtype=float) / 8

    monkeypatch.setattr(layer_module, "Sampler", FakeSampler)
    monkeypatch.setattr(layer_module, "SPSASamplerGradient", FakeGradient)
    monkeypatch.setattr(layer_module, "SamplerQNN", FakeSamplerQNN)

    def qcnn_layer_spy(input_qubits, *args, **kwargs):
        ansatz_calls.append((input_qubits, args, kwargs))
        return QCNN_layer(input_qubits, *args, **kwargs)

    num_qubits = 4
    output_qubits = 3
    layer = VQCLayer(
        num_qubits=num_qubits,
        num_layers=1,
        shots=16,
        ansatz_factory=qcnn_layer_spy,
        output_qubits=output_qubits,
    )

    identity_matrix = torch.eye(2**num_qubits, dtype=torch.complex128).unsqueeze(0)
    result = layer(identity_matrix)

    assert result.shape == (1, 2**output_qubits)
    assert ansatz_calls[-1] == (
        num_qubits,
        (),
        {"num_layers": 1, "output_qubits": output_qubits},
    )
    assert observed_circuits
    assert observed_circuits[0].num_clbits == output_qubits
