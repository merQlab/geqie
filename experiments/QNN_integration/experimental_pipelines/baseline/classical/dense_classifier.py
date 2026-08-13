"""Classical baseline: image -> dense classifier."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
	for candidate in Path(__file__).resolve().parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			sys.path.insert(0, str(candidate))
			break

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


class DenseClassifier(nn.Module):
	def __init__(
		self,
		num_classes: int = 10,
		dim_input: tuple[int, int, int] = (1, 32, 32),
	) -> None:
		super().__init__()
		self.flatten = nn.Flatten()
		self.classifier = nn.Linear(
			dim_input[0] * dim_input[1] * dim_input[2],
			num_classes,
		)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		x = self.flatten(x)
		x = self.classifier(x)
		x = self.softmax(x)
		return x


def train_one_subset(
	data_block: DataBlock,
	*,
	num_classes=10,
	num_qubits=0,
	num_layers=0,
	epochs=50,
	batch_size=16,
	device="cpu",
	verbose=True,
	report_context=None,
	progress_callback=None,
):
	image_shape = data_block_image_shape(data_block)
	model = DenseClassifier(num_classes, dim_input=image_shape)
	train_loader, val_loader, test_loader = image_loaders(
		data_block,
		batch_size,
		normalize=True,
		add_channel=False,
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
	dataset_id="cifar_bw",
	**overrides,
):
	run_options = {
		"num_classes": 10,
		"num_qubits": 0,
		"num_layers": 0,
		"epochs": 50,
		"batch_size": 16,
		"device": "cpu",
		"verbose": True,
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
		model_family="classical",
		encoding_id="raw_pixels",
		model_id="dense_classifier",
		pipeline_name="Dense classifier",
		classifier_name="Dense classifier",
		model_architecture=(
			f"{describe_image_shape(image_shape)} -> Flatten({input_features}) -> "
			f"Linear({input_features}, num_classes) -> Softmax"
		),
		**run_options,
	)


if __name__ == "__main__":
	run()
