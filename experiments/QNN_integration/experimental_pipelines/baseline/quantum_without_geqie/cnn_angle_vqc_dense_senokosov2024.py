"""Quantum baseline without GEQIE inspired by Senokosov et al. (2024)."""

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
	image_loaders,
	load_dataset,
	run_subsets,
	train_model,
)

from geqie_qml.ansatze import default_vqc_ansatz
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import SPSASamplerGradient
from qiskit_machine_learning.neural_networks import SamplerQNN


def angle_embedding(
	num_qubits: int,
) -> QuantumCircuit:
	circuit = QuantumCircuit(num_qubits)
	parameters = ParameterVector("input", num_qubits)
	for index, parameter in enumerate(parameters):
		circuit.rx(parameter, index)
	return circuit


class CNNAngleVQCDenseSenokosov2024(nn.Module):
	def __init__(
		self,
		num_qubits: int = 9,
		num_layers: int = 1,
		num_classes: int = 10,
		shots: int = 1024,
	) -> None:
		super().__init__()
		self.num_qubits = num_qubits
		self.cnn = nn.Sequential(
			nn.Conv2d(1, 16, 5, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, 5, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.feature_head = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 4 * 4, num_qubits),
			nn.BatchNorm1d(num_qubits),
			nn.ReLU(),
		)
		embedding = angle_embedding(num_qubits)
		ansatz = default_vqc_ansatz(num_qubits, num_layers)
		circuit = QuantumCircuit(num_qubits)
		circuit.compose(embedding, inplace=True)
		circuit.compose(ansatz, inplace=True)
		sampler = StatevectorSampler(default_shots=shots)
		gradient = SPSASamplerGradient(sampler=sampler)
		qnn = SamplerQNN(
			circuit=circuit,
			input_params=list(embedding.parameters),
			weight_params=list(ansatz.parameters),
			sampler=sampler,
			gradient=gradient,
		)
		initial_weights = np.random.uniform(-np.pi, np.pi, len(ansatz.parameters))
		self.qnn = TorchConnector(qnn, initial_weights=initial_weights)
		self.head = nn.Linear(2 ** num_qubits, num_classes)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim == 3:
			x = x.unsqueeze(1)
		x = x.float() / 255.0
		x = self.cnn(x)
		x = self.feature_head(x)
		x = self.qnn(x)
		x = x * (2 ** self.num_qubits)
		x = self.head(x)
		x = self.log_softmax(x)
		return x


def train_one_subset(
	data_block: DataBlock,
	*,
	num_classes=10,
	num_qubits=9,
	num_layers=1,
	epochs=30,
	batch_size=16,
	device="cpu",
	verbose=False,
	report_context=None,
	progress_callback=None,
):
	model = CNNAngleVQCDenseSenokosov2024(num_qubits, num_layers, num_classes)
	train_loader, val_loader, test_loader = image_loaders(
		data_block,
		batch_size,
		normalize=False,
		add_channel=False,
	)
	optimizer = Adam([
		{"params": model.cnn.parameters(), "lr": 1e-3},
		{"params": model.feature_head.parameters(), "lr": 1e-3},
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
		"num_qubits": 9,
		"num_layers": 1,
		"epochs": 30,
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
		encoding_id="angle_embedding",
		model_id="cnn_vqc_dense_senokosov2024",
		pipeline_name="CNN + angle embedding + VQC + dense (Senokosov 2024)",
		classifier_name="CNN + angle embedding + QNN + Dense",
		model_architecture="16x16 -> CNN -> Dense(512, qubits) -> Rx angle embedding -> VQC -> QNN -> Dense",
		**run_options,
	)


if __name__ == "__main__":
	run()
