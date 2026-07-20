"""Shared CNN feature maps -> GEQIE encoding -> VQC -> dense implementation."""

from __future__ import annotations

import importlib
from typing import Any

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


def normalize_encoding_id(encoding_id: str) -> str:
	"""Return the module name used by the GEQIE encoding registry."""
	if not isinstance(encoding_id, str):
		raise TypeError(f"encoding_id must be a string; got {type(encoding_id).__name__}.")
	encoding_id = encoding_id.strip().lower()
	if not encoding_id:
		raise ValueError("encoding_id must not be empty.")
	return encoding_id


def infer_encoded_feature_map_qubits(
	encoding_id: str,
	feature_size: tuple[int, int],
	encoding_params: dict[str, Any] | None = None,
) -> int:
	"""Infer the encoded circuit width using the selected GEQIE module."""
	encoding_id = normalize_encoding_id(encoding_id)
	if len(feature_size) != 2 or any(size <= 0 for size in feature_size):
		raise ValueError(f"feature_size must contain two positive dimensions; got {feature_size!r}.")

	try:
		encoding_module = importlib.import_module(f"geqie.encodings.{encoding_id}")
	except ModuleNotFoundError as error:
		if error.name == f"geqie.encodings.{encoding_id}":
			raise ValueError(f"Unknown GEQIE encoding_id: {encoding_id!r}.") from error
		raise

	params = dict(encoding_params or {})
	probe_image = np.zeros(feature_size, dtype=np.float32)
	coordinate_qubits = int(np.ceil(np.log2(max(feature_size))))
	data_state = encoding_module.data_function(
		0,
		0,
		R=coordinate_qubits,
		image=probe_image,
		**params,
	)
	map_operator = encoding_module.map_function(
		0,
		0,
		R=coordinate_qubits,
		image=probe_image,
		**params,
	)
	num_qubits = data_state.num_qubits + map_operator.num_qubits

	# The initial state is part of the encoding contract as well.  Calling it
	# here catches inconsistent custom encodings before a training process starts.
	initial_state = encoding_module.init_function(num_qubits, **params)
	if initial_state.num_qubits != num_qubits:
		raise ValueError(
		f"GEQIE/{encoding_id.upper()} creates an initial state with "
		f"{initial_state.num_qubits} qubits, but its data and map functions require {num_qubits}."
	)
	return num_qubits


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
		num_qubits: int | None,
		num_layers: int,
		batch_size: int,
		convolution_depth: int = 2,
		encoding_id: str = "frqi",
		encoding_params: dict[str, Any] | None = None,
	) -> None:
		super().__init__()
		self.encoding_id = normalize_encoding_id(encoding_id)
		self.encoding_params = dict(encoding_params or {})
		self.cnn, feature_maps, feature_size = build_feature_extractor(convolution_depth)
		expected_qubits = infer_encoded_feature_map_qubits(
			self.encoding_id,
			feature_size,
			self.encoding_params,
		)
		if num_qubits is None:
			num_qubits = expected_qubits
		elif num_qubits != expected_qubits:
			raise ValueError(
				f"GEQIE/{self.encoding_id.upper()} needs {expected_qubits} qubits for feature maps "
				f"of shape {feature_size} with encoding_params={self.encoding_params!r}; got {num_qubits}."
			)
		self.quantum = VQCLayerForCNNFeatureMaps(
			num_qubits=num_qubits,
			num_layers=num_layers,
			shots=1024,
			batch_size=batch_size,
			feature_maps=feature_maps,
			output_qubits=num_qubits,
			geqie_encoding=self.encoding_id,
			encoding_params=self.encoding_params,
		)
		self.head = nn.Linear(
			feature_maps * (2 ** num_qubits),
			num_classes,
		)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.cnn(x)
		x = self.quantum(
			x,
			geqie_encoding=self.encoding_id,
			encoding_params=self.encoding_params,
		)
		x = x.flatten(start_dim=1)
		x = self.head(x)
		x = self.log_softmax(x)
		return x


def train_one_subset(
	data_block: DataBlock,
	*,
	num_classes=10,
	num_qubits=None,
	num_layers=1,
	epochs=50,
	batch_size=16,
	device="cpu",
	verbose=True,
	report_context=None,
	progress_callback=None,
	convolution_depth=2,
	lr=1e-3,
	encoding_id="frqi",
	encoding_params=None,
	**_,
):
	model = CNNFeatureMapsGEQIEVQCDenseClassifier(
		num_classes=num_classes,
		num_qubits=num_qubits,
		num_layers=num_layers,
		batch_size=batch_size,
		convolution_depth=convolution_depth,
		encoding_id=encoding_id,
		encoding_params=encoding_params,
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
		progress_callback=progress_callback,
	)


def run_cnn_feature_maps_vqc_dense(
	dataset=None,
	*,
	dataset_id="mnist_digits",
	convolution_depth=2,
	lr=1e-3,
	encoding_id=None,
	encoding_params=None,
	num_qubits=None,
	**overrides,
):
	encoding_id = normalize_encoding_id(encoding_id)
	encoding_params = dict(encoding_params or {})
	_, feature_maps, feature_size = build_feature_extractor(convolution_depth)
	expected_qubits = infer_encoded_feature_map_qubits(
		encoding_id,
		feature_size,
		encoding_params,
	)
	if num_qubits is None:
		num_qubits = expected_qubits
	elif num_qubits != expected_qubits:
		raise ValueError(
			f"GEQIE/{encoding_id.upper()} needs {expected_qubits} qubits for feature maps "
			f"of shape {feature_size} with encoding_params={encoding_params!r}; got {num_qubits}."
		)
	encoding_label = encoding_id.upper()

	run_options = {
		"num_classes": 10,
		"num_qubits": num_qubits,
		"num_layers": 1,
		"epochs": 50,
		"batch_size": 16,
		"device": "cpu",
		"verbose": True,
		"max_workers": 16,
		"training_setup_extra": {
			"cnn_lr": lr,
			"convolution_depth": convolution_depth,
			"encoding_method": encoding_id,
			"encoding_params": encoding_params,
		},
		"subset_kwargs_factory": lambda _index, _: {
			"convolution_depth": convolution_depth,
			"lr": lr,
			"encoding_id": encoding_id,
			"encoding_params": encoding_params,
		},
	}
	run_options.update(overrides)
	return run_subsets(
		dataset=dataset or load_dataset(dataset_id),
		trainer=train_one_subset,
		dataset_id=dataset_id,
		experiment_group="experiment",
		model_family="geqie",
		encoding_id=encoding_id,
		model_id="cnn_feature_maps_vqc_dense",
		pipeline_name=f"CNN feature maps + GEQIE/{encoding_label} + VQC + dense",
		classifier_name=f"CNN + GEQIE({encoding_label} feature maps) + QNN + Dense",
		model_architecture=(
			f"16x16 -> CNN feature maps ({feature_maps}x{feature_size[0]}x{feature_size[1]}) "
			f"-> GEQIE({encoding_label}) -> VQC -> Dense"
		),
		**run_options,
	)
