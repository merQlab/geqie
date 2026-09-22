"""Numerical and training checks for the quantum baselines' exact simulator."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Operator, Statevector, random_unitary

from experiments.QNN_integration.experimental_pipelines.baseline.quantum_without_geqie import (
    cnn_angle_vqc_dense_senokosov2024 as angle_pipeline,
    cnn_zz_vqc_dense as cnn_pipeline,
    pca_zz_vqc_dense as pca_pipeline,
)
from experiments.QNN_integration.experimental_pipelines.baseline.quantum_without_geqie.statevector_encodings import (
    RXFeatureMapStatevector,
    ZZFeatureMapStatevector,
)
from experiments.QNN_integration.experimental_pipelines.common import DataBlock, DataSet, DatasetSplit
from geqie_qml import SamplerAnsatzLayer
from geqie_qml.ansatze import default_vqc_ansatz


def rx_feature_map(num_qubits):
    circuit = QuantumCircuit(num_qubits)
    for index, parameter in enumerate(ParameterVector("x", num_qubits)):
        circuit.rx(parameter, index)
    return circuit


ENCODINGS = (
    (ZZFeatureMapStatevector, lambda n: zz_feature_map(n, reps=1)),
    (RXFeatureMapStatevector, rx_feature_map),
)
PIPELINES = (pca_pipeline, cnn_pipeline, angle_pipeline)


class BaselineSamplerNumericalTests(unittest.TestCase):
    def test_encoded_states_match_qiskit_including_qubit_order(self):
        rng = np.random.default_rng(42)
        for encoder_class, circuit_factory in ENCODINGS:
            for num_qubits in (2, 3, 12):
                with self.subTest(encoding=encoder_class.__name__, qubits=num_qubits):
                    features = np.stack((
                        rng.uniform(-4, 4, num_qubits),
                        np.zeros(num_qubits),
                        np.full(num_qubits, np.pi),
                    ))
                    circuit = circuit_factory(num_qubits)
                    actual = encoder_class(num_qubits)(torch.tensor(features)).numpy()
                    expected = np.stack([
                        Statevector.from_instruction(circuit.assign_parameters(row)).data
                        for row in features
                    ])
                    np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_probabilities_and_input_and_weight_gradients_match_qiskit(self):
        rng = np.random.default_rng(17)
        for encoder_class, circuit_factory in ENCODINGS:
            with self.subTest(encoding=encoder_class.__name__):
                encoding = encoder_class(3)
                feature_map = circuit_factory(3)
                ansatz = default_vqc_ansatz(3, 2)
                values = rng.uniform(-1, 1, (2, 3))
                weights = rng.uniform(-1, 1, ansatz.num_parameters)
                layer = SamplerAnsatzLayer(3, ansatz).double()
                with torch.no_grad():
                    layer.weights.copy_(torch.tensor(weights))
                inputs = torch.tensor(values, requires_grad=True)
                coefficients = rng.normal(size=(2, 8))

                def reference(features, parameters):
                    bound_ansatz = ansatz.assign_parameters(parameters)
                    return np.stack([
                        Statevector.from_instruction(feature_map.assign_parameters(row))
                        .evolve(bound_ansatz).probabilities()
                        for row in features
                    ])

                probabilities = layer(encoding(inputs))
                np.testing.assert_allclose(probabilities.detach(), reference(values, weights), atol=1e-12)
                np.testing.assert_allclose(probabilities.detach().sum(dim=1), 1, atol=1e-12)
                (probabilities * torch.tensor(coefficients)).sum().backward()

                epsilon = 1e-6
                for target, actual_gradient in ((values, inputs.grad), (weights, layer.weights.grad)):
                    numerical = np.zeros_like(target)
                    for index in np.ndindex(target.shape):
                        plus, minus = target.copy(), target.copy()
                        plus[index] += epsilon
                        minus[index] -= epsilon
                        if target is values:
                            difference = reference(plus, weights) - reference(minus, weights)
                        else:
                            difference = reference(values, plus) - reference(values, minus)
                        numerical[index] = np.sum(difference * coefficients) / (2 * epsilon)
                    np.testing.assert_allclose(actual_gradient, numerical, atol=1e-7)
                    self.assertGreater(np.linalg.norm(numerical), 1e-3)

    def test_input_adjoint_handles_complex_fixed_gates_and_real_or_complex_states(self):
        circuit = default_vqc_ansatz(3, 2)
        circuit.h(1)
        circuit.s(0)
        circuit.unitary(random_unitary(4, seed=19).data, [2, 0])
        layer = SamplerAnsatzLayer(3, circuit, seed=5).double()
        matrix = torch.tensor(Operator(circuit.assign_parameters(layer.weights.detach().numpy())).data)
        for dtype in (torch.float64, torch.complex128):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                inputs = torch.randn(2, 8, dtype=dtype, requires_grad=True)
                expected_inputs = inputs.detach().clone().requires_grad_()
                coefficients = torch.randn(2, 8, dtype=torch.float64)
                actual = layer(inputs)
                expected = (expected_inputs.to(matrix.dtype) @ matrix.T).abs().square()
                (actual * coefficients).sum().backward()
                (expected * coefficients).sum().backward()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(inputs.grad, expected_inputs.grad)


class BaselineSamplerTrainingTests(unittest.TestCase):
    @staticmethod
    def data_block(channels):
        rng = np.random.default_rng(41)
        shape = (8, 8) if channels == 1 else (8, 8, channels)
        splits = [
            DatasetSplit(rng.integers(0, 256, (count, *shape), dtype=np.uint8), np.arange(count) % 2)
            for count in (6, 2, 2)
        ]
        return DataBlock(*splits)

    def test_all_pipelines_train_quantum_and_classical_layers(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for pipeline in PIPELINES:
            for channels in (1, 3):
                with self.subTest(pipeline=pipeline.__name__, channels=channels):
                    torch.manual_seed(42)
                    original_train = pipeline.train_model
                    initial = {}
                    head_input_sums = []

                    def capture_training(**kwargs):
                        model = kwargs["model"]
                        self.assertIsInstance(model.qnn, SamplerAnsatzLayer)
                        initial.update({name: p.detach().clone() for name, p in model.named_parameters()})
                        hook = model.head.register_forward_pre_hook(
                            lambda _, args: head_input_sums.append(args[0].detach().sum(dim=1))
                        )
                        try:
                            return original_train(**kwargs)
                        finally:
                            hook.remove()

                    progress = []
                    with patch.object(pipeline, "train_model", side_effect=capture_training), \
                         patch("qiskit.primitives.StatevectorSampler.run", side_effect=AssertionError("Legacy sampler used")), \
                         patch("qiskit_machine_learning.neural_networks.SamplerQNN.__init__", side_effect=AssertionError("Legacy QNN used")):
                        result = pipeline.train_one_subset(
                            self.data_block(channels), num_classes=2, num_qubits=3, num_layers=2,
                            epochs=2, batch_size=2, progress_callback=progress.append,
                        )
                    self.assertEqual(len(result["history"]["train_loss"]), 2)
                    self.assertEqual(result["confusion_matrix"].shape, (2, 2))
                    self.assertEqual(progress[-1]["status"], "complete")
                    self.assertTrue(np.isfinite(result["test_metrics"]["loss"]))
                    parameters = dict(result["model"].named_parameters())
                    for name in ("qnn.weights", "head.weight", "cnn.0.weight", "feature_head.1.weight"):
                        if name in initial:
                            self.assertFalse(torch.equal(initial[name], parameters[name]), msg=name)
                    for sums in head_input_sums:
                        torch.testing.assert_close(sums, torch.ones_like(sums))

    def test_runners_report_exact_sampler_configuration(self):
        dataset = DataSet([self.data_block(1)], {"name": "synthetic"})
        for pipeline in PIPELINES:
            with self.subTest(pipeline=pipeline.__name__), \
                 patch.object(pipeline, "run_subsets", side_effect=lambda **kwargs: kwargs):
                configuration = pipeline.run(
                    dataset=dataset, training_setup_extra={"experiment_note": "test", "shots": 1024},
                )
                setup = configuration["training_setup_extra"]
                self.assertEqual(setup["experiment_note"], "test")
                self.assertEqual(setup["quantum_layer"], "SamplerAnsatzLayer")
                self.assertEqual(setup["gradient_method"], "parameter_shift")
                self.assertIsNone(setup["shots"])
                self.assertFalse(setup["scale_output"])
                self.assertIn("SamplerAnsatzLayer(default VQC)", configuration["model_architecture"])


if __name__ == "__main__":
    unittest.main()
