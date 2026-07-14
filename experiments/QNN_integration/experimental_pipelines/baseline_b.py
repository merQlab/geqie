"""Baseline B: 16x16 image -> classical CNN -> Dense classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam

try:
	from .common import DataBlock, image_loaders, load_mnist_digits_dataset, run_subsets, train_model
except ImportError:
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parent))

	from common import DataBlock, image_loaders, load_mnist_digits_dataset, run_subsets, train_model


class SimpleCNNBaseline(nn.Module):
	def __init__(
		self,
		num_classes: int = 10,
	) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 16, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 4 * 4, 64),
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
):
	model = SimpleCNNBaseline(num_classes)
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
	)


def run(
	dataset=None,
	**overrides,
):
	return run_subsets(
		dataset=dataset or load_mnist_digits_dataset(),
		trainer=train_one_subset,
		pipeline_name="Baseline B",
		classifier_name="CNN + Dense",
		model_architecture="16x16 grayscale -> Conv(1,16) -> Pool -> Conv(16,32) -> Pool -> Linear(512,64) -> Linear(64, classes)",
		num_classes=10,
		num_qubits=0,
		num_layers=0,
		epochs=50,
		batch_size=64,
		device="cuda" if torch.cuda.is_available() else "cpu",
		verbose=True,
		**overrides,
	)


if __name__ == "__main__":
	run()
