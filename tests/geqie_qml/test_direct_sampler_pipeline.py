"""Numerical and integration checks for the direct-GEQIE sampler migration."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from qiskit.quantum_info import Statevector, random_unitary

from experiments.QNN_integration.experimental_pipelines import common
from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models import direct_geqie
from experiments.QNN_integration.datasets.dataset_structure import DataBlock, DataSet, DatasetSplit
from geqie_qml import VQCLayer
from geqie_qml.ansatze import default_vqc_ansatz, real_amplitudes_ansatz


class DirectSamplerPipelineTests(unittest.TestCase):
    def test_probabilities_and_gradients_match_qiskit(self):
        matrices = np.stack([random_unitary(8, seed=seed).data for seed in (17, 23)])
        for factory in (default_vqc_ansatz, real_amplitudes_ansatz):
            with self.subTest(ansatz=factory.__name__):
                torch.manual_seed(42)
                model = common.GEQIEFirstClassifier(
                    num_qubits=3, num_layers=2, num_classes=2,
                    ansatz_factory=factory, use_sampler_ansatz=True,
                )
                inputs = torch.tensor(matrices, dtype=torch.complex128)
                probabilities = model.vqc(inputs)
                circuit = factory(3, num_layers=2)
                weights = model.vqc[1].weights.detach().numpy().astype(float)

                def reference(values):
                    bound = circuit.assign_parameters(dict(zip(circuit.parameters, values)))
                    return np.stack([
                        Statevector(matrix[:, 0]).evolve(bound).probabilities()
                        for matrix in matrices
                    ])

                np.testing.assert_allclose(probabilities.detach(), reference(weights), atol=1e-7)
                np.testing.assert_allclose(probabilities.detach().sum(dim=1), 1, atol=1e-7)
                coefficients = torch.linspace(-1, 1, 16).reshape(2, 8)
                (probabilities * coefficients).sum().backward()
                numerical = []
                epsilon = 1e-5
                for index in range(len(weights)):
                    plus, minus = weights.copy(), weights.copy()
                    plus[index] += epsilon
                    minus[index] -= epsilon
                    numerical.append(np.sum(
                        (reference(plus) - reference(minus)) * coefficients.numpy()
                    ) / (2 * epsilon))
                np.testing.assert_allclose(model.vqc[1].weights.grad, numerical, atol=1e-6)
                self.assertGreater(np.linalg.norm(numerical), 1e-3)

    def test_direct_variant_trains_from_zip_without_legacy_layer(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "subset_1.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for split in ("train", "val", "test"):
                    for index in range(4):
                        buffer = io.BytesIO()
                        np.savez(buffer, matrix=random_unitary(8, seed=index).data, label=index % 2)
                        archive.writestr(f"{split}/matrix_{index}_label_{index % 2}.npz", buffer.getvalue())

            initial = {}
            train_model = common.train_model

            def capture_training(**kwargs):
                model = kwargs["model"]
                initial["quantum"] = model.vqc[1].weights.detach().clone()
                initial["head"] = model.head.weight.detach().clone()
                return train_model(**kwargs)

            progress = []
            with patch("geqie_qml.VQCLayer", side_effect=AssertionError("Legacy layer used")), \
                 patch.object(common, "train_model", side_effect=capture_training) as training_spy:
                result = direct_geqie.train_one_subset(
                    0, zip_path=str(archive_path), model_id="direct_vqc_dense",
                    num_qubits=3, num_layers=1, num_classes=2,
                    epochs=2, batch_size=2, quantum_workers=32,
                    progress_callback=progress.append,
                )
            # Release the mock's DataLoader references so Windows can close the ZIP.
            training_spy.reset_mock()

            self.assertEqual(len(result["history"]["train_loss"]), 2)
            self.assertEqual(result["confusion_matrix"].shape, (2, 2))
            self.assertEqual(len(result["y_pred"]), 4)
            self.assertEqual(progress[-1]["status"], "complete")
            self.assertFalse(torch.equal(initial["quantum"], result["model"].vqc[1].weights))
            self.assertFalse(torch.equal(initial["head"], result["model"].head.weight))
            self.assertTrue(np.isfinite(result["test_metrics"]["loss"]))

    def test_adaptive_variants_keep_legacy_layer_and_readout(self):
        for model_id, variant in direct_geqie.MODEL_VARIANTS.items():
            if model_id == "direct_vqc_dense":
                continue
            with self.subTest(model=model_id):
                with patch.object(direct_geqie, "train_geqie_first_subset", side_effect=lambda **kw: kw):
                    configuration = direct_geqie.train_one_subset(
                        0, zip_path="unused.zip", model_id=model_id,
                        num_qubits=5, quantum_workers=7,
                    )
                self.assertFalse(configuration["use_sampler_ansatz"])
                self.assertEqual(configuration["quantum_workers"], 7)
                model = common.GEQIEFirstClassifier(
                    num_qubits=5, num_layers=1, num_classes=2,
                    ansatz_factory=configuration["ansatz_factory"],
                    output_qubits=configuration["output_qubits"],
                    interpret=configuration["interpret"],
                    use_sampler_ansatz=configuration["use_sampler_ansatz"],
                )
                self.assertIsInstance(model.vqc, VQCLayer)
                self.assertTrue(model.vqc.scale_output)
                self.assertEqual(model.head.in_features, 16 if variant.get("interpret") else 32)

    def test_direct_runner_records_new_backend_for_each_encoding(self):
        for encoding in ("frqi", "neqr", "mcqi"):
            with self.subTest(encoding=encoding):
                shape = (2, 2, 2, 3) if encoding == "mcqi" else (2, 2, 2)
                split = DatasetSplit(np.zeros(shape, dtype=np.uint8), np.array([0, 1]))
                dataset = DataSet([DataBlock(split, split, split)], {"name": "synthetic"})
                with patch.object(direct_geqie, "run_subsets", side_effect=lambda **kw: kw):
                    configuration = direct_geqie.run_direct_geqie(
                        encoding_id=encoding, model_id="direct_vqc_dense", dataset=dataset,
                    )
                setup = configuration["training_setup_extra"]
                self.assertEqual(setup["quantum_layer"], "SamplerAnsatzLayer")
                self.assertEqual(setup["gradient_method"], "parameter_shift")
                self.assertIsNone(setup["shots"])
                self.assertFalse(setup["scale_output"])
                self.assertEqual(setup["quantum_workers"], 1)
                self.assertIn("SamplerAnsatzLayer(default VQC)", configuration["model_architecture"])


if __name__ == "__main__":
    unittest.main()
