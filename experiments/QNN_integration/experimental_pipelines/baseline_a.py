"""Baseline A: 16x16 image -> Dense classifier."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Adam

try:
	from .common import DataBlock, image_loaders, load_mnist_digits_dataset, run_subsets, train_model
except ImportError:  # direct execution: python baseline_a.py
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).resolve().parent))

	from common import DataBlock, image_loaders, load_mnist_digits_dataset, run_subsets, train_model


class DenseBaseline(nn.Module):
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
):
	model = DenseBaseline(num_classes)
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
	)


def run(
	dataset=None,
	**overrides,
):
	return run_subsets(
		dataset=dataset or load_mnist_digits_dataset(),
		trainer=train_one_subset,
		pipeline_name="Baseline A",
		classifier_name="Dense layers only",
		model_architecture="16x16 grayscale -> Flatten(256) -> Linear(256, num_classes) -> LogSoftmax",
		num_classes=10,
		num_qubits=0,
		num_layers=0,
		epochs=50,
		batch_size=16,
		device="cpu",
		verbose=True,
		**overrides,
	)


if __name__ == "__main__":
	run()
