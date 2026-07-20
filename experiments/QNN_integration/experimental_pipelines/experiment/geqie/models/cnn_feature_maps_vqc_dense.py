"""Shared CNN feature maps -> GEQIE(FRQI) -> VQC -> dense implementation."""

from __future__ import annotations

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

from geqie_qml.input_output_layer import VQCLayerForCNNFeatureMaps


def build_feature_extractor(
	depth: int = 2,
) -> tuple[nn.Sequential, int, tuple[int, int]]:
	if depth not in (1, 2, 3):
		raise ValueError("convolution_depth must be 1, 2, or 3.")

	height = width = 16
	layers: list[nn.Module] = []
	for level in range(depth):
		in_channels = 1 if level == 0 else 2 ** (level + 2)
		out_channels = 2 ** (level + 3)
		layers.extend((
			nn.Conv2d(in_channels, out_channels, 3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.MaxPool2d(2),
		))
		height //= 2
		width //= 2

	return nn.Sequential(*layers), out_channels, (height, width)


class CNNFeatureMapsGEQIEVQCDenseClassifier(nn.Module):
	def __init__(
		self,
		*,
		num_classes: int,
		num_qubits: int,
		num_layers: int,
		batch_size: int,
		convolution_depth: int = 2,
	) -> None:
		super().__init__()
		self.cnn, feature_maps, feature_size = build_feature_extractor(convolution_depth)
		pixels = feature_size[0] * feature_size[1]
		expected_qubits = int(np.log2(pixels)) + 1
		if pixels & (pixels - 1) or num_qubits != expected_qubits:
			raise ValueError(
				f"FRQI needs {expected_qubits} qubits for feature maps of shape {feature_size}; got {num_qubits}."
			)
		self.quantum = VQCLayerForCNNFeatureMaps(
			num_qubits=num_qubits,
			num_layers=num_layers,
			shots=1024,
			batch_size=batch_size,
			feature_maps=feature_maps,
			output_qubits=num_qubits,
		)
		self.head = nn.Linear(
			feature_maps * (2 ** num_qubits),
			num_classes,
		)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.cnn(x)
		x = self.quantum(x)
		x = x.flatten(start_dim=1)
		x = self.head(x)
		x = self.log_softmax(x)
		return x


def train_one_subset(
	data_block: DataBlock,
	*,
	num_classes=10,
	num_qubits=5,
	num_layers=1,
	epochs=50,
	batch_size=16,
	device="cpu",
	verbose=True,
	report_context=None,
	convolution_depth=2,
	lr=1e-3,
	**_,
):
	model = CNNFeatureMapsGEQIEVQCDenseClassifier(
		num_classes=num_classes,
		num_qubits=num_qubits,
		num_layers=num_layers,
		batch_size=batch_size,
		convolution_depth=convolution_depth,
	)
	train_loader, val_loader, test_loader = image_loaders(
		data_block,
		batch_size,
		normalize=True,
		add_channel=True,
	)
	optimizer = Adam([
		{"params": model.cnn.parameters(), "lr": lr},
		{"params": model.quantum.parameters(), "lr": 1e-2},
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
	)


def run_cnn_feature_maps_vqc_dense(
	dataset=None,
	*,
	dataset_id="mnist_digits",
	convolution_depth=2,
	lr=1e-3,
	**overrides,
):
	run_options = {
		"num_classes": 10,
		"num_qubits": 5,
		"num_layers": 1,
		"epochs": 50,
		"batch_size": 16,
		"device": "cpu",
		"verbose": True,
		"training_setup_extra": {
			"cnn_lr": lr,
			"convolution_depth": convolution_depth,
			"encoding_method": "frqi",
		},
		"subset_kwargs_factory": lambda _index, _: {
			"convolution_depth": convolution_depth,
			"lr": lr,
		},
	}
	run_options.update(overrides)
	return run_subsets(
		dataset=dataset or load_dataset(dataset_id),
		trainer=train_one_subset,
		dataset_id=dataset_id,
		experiment_group="experiment",
		model_family="geqie",
		encoding_id="frqi",
		model_id="cnn_feature_maps_vqc_dense",
		pipeline_name="CNN feature maps + GEQIE/FRQI + VQC + dense",
		classifier_name="CNN + GEQIE(FRQI feature maps) + QNN + Dense",
		model_architecture="16x16 -> CNN feature maps (16x4x4) -> GEQIE(FRQI) -> VQC -> Dense",
		**run_options,
	)
