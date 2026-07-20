"""Quantum baseline without GEQIE: PCA -> ZZFeatureMap -> VQC -> dense."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
	for candidate in Path(__file__).resolve().parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			sys.path.insert(0, str(candidate))
			break

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from experiments.QNN_integration.experimental_pipelines.common import (
	DataBlock,
	load_dataset,
	pca_image_loaders,
	run_subsets,
	train_model,
)

from geqie_qml.ansatze import default_vqc_ansatz
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import SPSASamplerGradient
from qiskit_machine_learning.neural_networks import SamplerQNN


class PCAZZVQCDenseClassifier(nn.Module):
	def __init__(
		self,
		num_qubits: int = 12,
		num_layers: int = 1,
		num_classes: int = 10,
		shots: int = 1024,
	) -> None:
		super().__init__()
		self.num_qubits = num_qubits
		feature_map = zz_feature_map(
			feature_dimension=num_qubits,
			reps=1,
		)
		ansatz = default_vqc_ansatz(
			num_qubits,
			num_layers,
		)
		circuit = QuantumCircuit(num_qubits)
		circuit.compose(feature_map, inplace=True)
		circuit.compose(ansatz, inplace=True)
		sampler = StatevectorSampler(default_shots=shots)
		gradient = SPSASamplerGradient(sampler=sampler)
		qnn = SamplerQNN(
			circuit=circuit,
			input_params=list(feature_map.parameters),
			weight_params=list(ansatz.parameters),
			sampler=sampler,
			gradient=gradient,
		)
		initial_weights = np.random.uniform(
			-np.pi,
			np.pi,
			len(ansatz.parameters),
		)
		self.qnn = TorchConnector(
			qnn,
			initial_weights=initial_weights,
		)
		self.head = nn.Linear(
			2 ** num_qubits,
			num_classes,
		)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.qnn(x)
		x = x * (2 ** self.num_qubits)
		x = self.head(x)
		x = self.log_softmax(x)
		return x


def train_one_subset(
	data_block: DataBlock,
	*,
	num_classes=10,
	num_qubits=12,
	num_layers=1,
	epochs=50,
	batch_size=16,
	device="cpu",
	verbose=False,
	report_context=None,
	progress_callback=None,
):
	model = PCAZZVQCDenseClassifier(
		num_qubits,
		num_layers,
		num_classes,
	)
	train_loader, val_loader, test_loader = pca_image_loaders(
		data_block,
		batch_size,
		num_qubits,
	)
	optimizer = Adam([
		{"params": model.qnn.parameters(), "lr": 1e-3},
		{"params": model.head.parameters(), "lr": 1e-2},
	])
	return train_model(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		test_loader=test_loader,
		optimizer=optimizer,
		num_classes=num_classes,
		epochs=epochs,
		device=device,
		verbose=verbose,
		report_context=report_context,
		progress_callback=progress_callback,
	)


def run(
	dataset=None,
	*,
	dataset_id="mnist_digits",
	**overrides,
):
	run_options = {
		"num_classes": 10,
		"num_qubits": 12,
		"num_layers": 1,
		"epochs": 50,
		"batch_size": 16,
		"device": "cpu",
		"verbose": False,
	}
	run_options.update(overrides)
	return run_subsets(
		dataset=dataset or load_dataset(dataset_id),
		trainer=train_one_subset,
		dataset_id=dataset_id,
		experiment_group="baseline",
		model_family="quantum_without_geqie",
		encoding_id="zz_feature_map",
		model_id="pca_vqc_dense",
		pipeline_name="PCA + ZZ feature map + VQC + dense",
		classifier_name="PCA + ZZFeatureMap + QNN + Dense",
		model_architecture="16x16 -> Flatten(256) -> PCA(qubits) -> ZZFeatureMap -> VQC -> QNN -> Dense",
		**run_options,
	)


if __name__ == "__main__":
	run()
