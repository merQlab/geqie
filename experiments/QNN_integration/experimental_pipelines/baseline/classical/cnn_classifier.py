"""Classical baseline: image -> CNN -> dense classifier."""

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
	data_block_image_shape,
	dataset_image_shape,
	describe_image_shape,
	image_loaders,
	load_dataset,
	run_subsets,
	train_model,
)


class CNNClassifier(nn.Module):
	def __init__(
		self,
		num_classes: int = 10,
		input_shape: tuple[int, int, int] = (1, 32, 32),
	) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(input_shape[0], 16, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		with torch.no_grad():
			feature_count = self.features(torch.zeros(1, *input_shape)).numel()
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(feature_count, 64),
			nn.ReLU(),
			nn.Linear(64, num_classes),
			nn.LogSoftmax(dim=-1),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x


def train_one_subset(
	data_block: DataBlock,
	*,
	num_classes=10,
	num_qubits=0,
	num_layers=0,
	epochs=50,
	batch_size=64,
	device="cpu",
	verbose=True,
	report_context=None,
	progress_callback=None,
):
	image_shape = data_block_image_shape(data_block)
	model = CNNClassifier(num_classes, input_shape=image_shape)
	train_loader, val_loader, test_loader = image_loaders(
		data_block,
		batch_size,
		normalize=True,
		add_channel=True,
	)
	optimizer = Adam(
		model.parameters(),
		lr=1e-3,
	)
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
	dataset_id="cifar_rgb",
	**overrides,
):
	run_options = {
		"num_classes": 10,
		"num_qubits": 0,
		"num_layers": 0,
		"epochs": 50,
		"batch_size": 64,
		"device": "cuda" if torch.cuda.is_available() else "cpu",
		"verbose": True,
	}
	run_options.update(overrides)
	dataset = dataset or load_dataset(dataset_id)
	image_shape = dataset_image_shape(dataset)
	return run_subsets(
		dataset=dataset,
		trainer=train_one_subset,
		dataset_id=dataset_id,
		experiment_group="baseline",
		model_family="classical",
		encoding_id="raw_pixels",
		model_id="cnn_classifier",
		pipeline_name="CNN classifier",
		classifier_name="CNN + Dense",
		model_architecture=(
			f"{describe_image_shape(image_shape)} -> Conv({image_shape[0]},16) -> Pool -> "
			"Conv(16,32) -> Pool -> Flatten -> Linear(64) -> Linear(classes)"
		),
		**run_options,
	)


if __name__ == "__main__":
	run()
