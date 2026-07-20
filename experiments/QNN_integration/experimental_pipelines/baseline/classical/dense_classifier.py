"""Classical baseline: 16x16 image -> dense classifier."""

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
	image_loaders,
	load_dataset,
	run_subsets,
	train_model,
)


class DenseClassifier(nn.Module):
	def __init__(
		self,
		num_classes: int = 10,
	) -> None:
		super().__init__()
		self.flatten = nn.Flatten()
		self.classifier = nn.Linear(
			16 * 16,
			num_classes,
		)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x):
		x = self.flatten(x)
		x = self.classifier(x)
		x = self.log_softmax(x)
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
	model = DenseClassifier(num_classes)
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
	dataset_id="mnist_digits",
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
	return run_subsets(
		dataset=dataset or load_dataset(dataset_id),
		trainer=train_one_subset,
		dataset_id=dataset_id,
		experiment_group="baseline",
		model_family="classical",
		encoding_id="raw_pixels",
		model_id="dense_classifier",
		pipeline_name="Dense classifier",
		classifier_name="Dense classifier",
		model_architecture="16x16 grayscale -> Flatten(256) -> Linear(256, num_classes) -> LogSoftmax",
		**run_options,
	)


if __name__ == "__main__":
	run()
