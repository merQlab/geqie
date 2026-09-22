"""Quantum baseline without GEQIE: PCA -> ZZFeatureMap -> VQC -> dense."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
	for candidate in Path(__file__).resolve().parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			sys.path.insert(0, str(candidate))
			break

import torch
import torch.nn as nn
from torch.optim import Adam

from experiments.QNN_integration.experimental_pipelines.common import (
	DataBlock,
	dataset_image_shape,
	describe_image_shape,
	load_dataset,
	pca_image_loaders,
	run_subsets,
	train_model,
)

from experiments.QNN_integration.experimental_pipelines.baseline.quantum_without_geqie.statevector_encodings import (
	ZZFeatureMapStatevector,
)
from geqie_qml import SamplerAnsatzLayer
from geqie_qml.ansatze import default_vqc_ansatz


class PCAZZVQCDenseClassifier(nn.Module):
	def __init__(
		self,
		num_qubits: int = 12,
		num_layers: int = 1,
		num_classes: int = 10,
	) -> None:
		super().__init__()
		self.num_qubits = num_qubits
		self.encoding = ZZFeatureMapStatevector(num_qubits)
		ansatz = default_vqc_ansatz(
			num_qubits,
			num_layers,
		)
		self.qnn = SamplerAnsatzLayer(
			num_qubits,
			ansatz,
			weight_init=torch.empty(ansatz.num_parameters).uniform_(-torch.pi, torch.pi),
		)
		self.head = nn.Linear(
			2 ** num_qubits,
			num_classes,
		)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.encoding(x)
		x = self.qnn(x)
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
	dataset = dataset or load_dataset(dataset_id)
	image_shape = dataset_image_shape(dataset)
	input_features = image_shape[0] * image_shape[1] * image_shape[2]
	return run_subsets(
		dataset=dataset,
		trainer=train_one_subset,
		dataset_id=dataset_id,
		experiment_group="baseline",
		model_family="quantum_without_geqie",
		encoding_id="zz_feature_map",
		model_id="pca_vqc_dense",
		pipeline_name="PCA + ZZ feature map + VQC + dense",
		classifier_name="PCA + ZZFeatureMap + SamplerAnsatzLayer + Dense",
		model_architecture=(
			f"{describe_image_shape(image_shape)} -> Flatten({input_features}) -> PCA(qubits) "
			"-> ZZFeatureMap statevector -> SamplerAnsatzLayer(default VQC) -> Dense"
		),
		training_setup_extra={
			**(run_options.pop("training_setup_extra", None) or {}),
			"quantum_layer": "SamplerAnsatzLayer",
			"shots": None,
			"gradient_method": "parameter_shift",
			"scale_output": False,
		},
		**run_options,
	)


if __name__ == "__main__":
	run()
