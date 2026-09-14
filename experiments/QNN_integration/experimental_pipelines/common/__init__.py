"""Shared, side-effect-free support for the QNN experimental pipelines.

Every pipeline module owns its model definition and its ``main`` function.  This
module intentionally contains only reusable infrastructure: locating the
repository, loading data, preparing loaders, training, reporting, and GEQIE
precomputation.
"""

from __future__ import annotations

import copy
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Mapping

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def find_repository_root(start: Path | None = None) -> Path:
	"""Return the repository root without relying on the current directory."""
	path = (start or Path(__file__)).resolve()
	for candidate in (path, *path.parents):
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			return candidate
	raise RuntimeError("Could not locate the repository root from experimental_pipelines.")


ROOT = find_repository_root()
QNN_INTEGRATION_DIR = ROOT / "experiments" / "QNN_integration"
DATASETS_DIR = QNN_INTEGRATION_DIR / "datasets"

DATASET_FILES = {
	"mnist_digits": "MNIST_Digits_5_subsets_train_val_test_16x16.joblib",
	"mnist_fashion": "MNIST_Fashion_5_subsets_train_val_test_16x16.joblib",
	"cifar_bw": "CIFAR-10/CIFAR-BW_5_subsets_train_val_test_32x32.joblib",
	"cifar_rgb": "CIFAR-10/CIFAR-RGB_5_subsets_train_val_test_32x32.joblib",
}
DATASET_ALIASES = {
	"mnist": "mnist_digits",
	"mnist_digit": "mnist_digits",
	"fashion_mnist": "mnist_fashion",
	"fashion": "mnist_fashion",
	"cifar": "cifar_bw",
	"cifar_10": "cifar_bw",
	"cifar_10_bw": "cifar_bw",
	"cifar_10_rgb": "cifar_rgb",
}


def ensure_repository_imports() -> None:
	"""Make direct ``python pipeline.py`` execution work from any directory."""
	for path in (ROOT, ROOT / "geqie" / "src", ROOT / "geqie-qml" / "src"):
		text = str(path)
		if text not in sys.path:
			sys.path.insert(0, text)


ensure_repository_imports()

from experiments.QNN_integration.datasets.dataset_structure import DataBlock, DataSet, DatasetSplit  # noqa: E402
from experiments.QNN_integration.experiment_results import (  # noqa: E402
	print_epoch_table_footer,
	print_epoch_table_header,
	print_epoch_table_row,
	print_metrics_report,
	with_torchinfo_summary,
)
from experiments.QNN_integration.subset_multiprocessing import train_subsets_with_process_pool  # noqa: E402


def normalize_dataset_id(dataset_id: str) -> str:
	"""Return the canonical filesystem-safe identifier of a configured dataset."""
	canonical = str(dataset_id).strip().lower().replace("-", "_").replace(" ", "_")
	return DATASET_ALIASES.get(canonical, canonical)


def load_dataset(dataset_id: str = "mnist_digits") -> DataSet:
	"""Load a configured protocol dataset by its canonical identifier."""
	canonical = normalize_dataset_id(dataset_id)
	try:
		filename = DATASET_FILES[canonical]
	except KeyError as error:
		available = ", ".join(sorted(DATASET_FILES))
		raise ValueError(f"Unknown dataset_id {dataset_id!r}. Available datasets: {available}.") from error

	# The existing joblib artifacts were created by notebooks in which these
	# dataclasses lived in ``__main__``. Registering the aliases keeps the files
	# loadable from standalone Python pipeline modules as well.
	import __main__

	for class_ in (DatasetSplit, DataBlock, DataSet):
		if not hasattr(__main__, class_.__name__):
			setattr(__main__, class_.__name__, class_)
	return joblib.load(DATASETS_DIR / filename)


def load_mnist_digits_dataset() -> DataSet:
	"""Backward-compatible alias for the canonical dataset registry."""
	return load_dataset("mnist_digits")


def infer_image_shape(images: np.ndarray) -> tuple[int, int, int]:
	"""Return an image batch shape as ``(channels, height, width)``.

	Protocol datasets store grayscale images as ``NHW`` and RGB images as
	``NHWC``.  ``NCHW`` is accepted as well, which keeps callers independent of
	the storage layout used by a future dataset notebook.
	"""
	values = np.asarray(images)
	if values.ndim == 3:
		_, height, width = values.shape
		return 1, int(height), int(width)
	if values.ndim != 4:
		raise ValueError(
			"Expected an image batch with shape NHW, NHWC, or NCHW; "
			f"got {values.shape}."
		)

	sample_shape = tuple(int(size) for size in values.shape[1:])
	if sample_shape[-1] in (1, 3, 4):
		height, width, channels = sample_shape
		return channels, height, width
	if sample_shape[0] in (1, 3, 4):
		channels, height, width = sample_shape
		return channels, height, width
	raise ValueError(
		"Could not identify the channel axis in image batch "
		f"with shape {values.shape}."
	)


def data_block_image_shape(data_block: DataBlock) -> tuple[int, int, int]:
	"""Infer and validate one common image shape for all splits."""
	shapes = {
		split_name: infer_image_shape(getattr(data_block, split_name).X)
		for split_name in ("train", "val", "test")
	}
	if len(set(shapes.values())) != 1:
		raise ValueError(f"Dataset splits use inconsistent image shapes: {shapes}.")
	return shapes["train"]


def dataset_image_shape(dataset: DataSet) -> tuple[int, int, int]:
	"""Infer and validate one common image shape for every dataset subset."""
	if not dataset.subsets:
		raise ValueError("Dataset does not contain any subsets.")
	shapes = [data_block_image_shape(block) for block in dataset.subsets]
	if len(set(shapes)) != 1:
		raise ValueError(f"Dataset subsets use inconsistent image shapes: {shapes}.")
	return shapes[0]


def describe_image_shape(image_shape: tuple[int, int, int]) -> str:
	"""Return a compact human-readable ``HxW``/``HxWxC`` shape label."""
	channels, height, width = image_shape
	return f"{height}x{width}" if channels == 1 else f"{height}x{width}x{channels}"


def _channels_first_images(images: np.ndarray) -> torch.Tensor:
	"""Convert a protocol image batch to the NCHW layout expected by PyTorch."""
	values = torch.as_tensor(np.asarray(images), dtype=torch.float32)
	channels, _, _ = infer_image_shape(images)
	if values.ndim == 3:
		return values.unsqueeze(1)
	if values.shape[1] == channels:
		return values
	return values.permute(0, 3, 1, 2).contiguous()


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
	}


def history_template() -> dict[str, list[float]]:
	return {key: [] for key in (
		"train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1",
		"val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1",
	)}


def image_loaders(
	data_block: DataBlock,
	batch_size: int,
	*,
	normalize: bool,
	add_channel: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""Create loaders from image splits while keeping preprocessing explicit."""
	def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
		values = (
			_channels_first_images(x)
			if add_channel
			else torch.as_tensor(np.asarray(x), dtype=torch.float32)
		)
		if normalize:
			values = values / 255.0
		labels = torch.tensor(np.asarray(y), dtype=torch.long)
		return DataLoader(TensorDataset(values, labels), batch_size=batch_size, shuffle=shuffle)

	return (
		make_loader(data_block.train.X, data_block.train.y, True),
		make_loader(data_block.val.X, data_block.val.y, False),
		make_loader(data_block.test.X, data_block.test.y, False),
	)


def pca_image_loaders(
	data_block: DataBlock,
	batch_size: int,
	n_components: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""Fit PCA only on a training split, then transform all three splits."""
	pca = PCA(n_components=n_components)
	train = np.asarray(data_block.train.X, dtype=np.float32).reshape(len(data_block.train.X), -1) / 255.0
	pca.fit(train)

	def transform(x: np.ndarray) -> torch.Tensor:
		flattened = np.asarray(x, dtype=np.float32).reshape(len(x), -1) / 255.0
		return torch.tensor(pca.transform(flattened), dtype=torch.float32)

	def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
		features = transform(x)
		labels = torch.tensor(
			np.asarray(y),
			dtype=torch.long,
		)
		dataset = TensorDataset(
			features,
			labels,
		)
		return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
		)

	return (
		make_loader(data_block.train.X, data_block.train.y, True),
		make_loader(data_block.val.X, data_block.val.y, False),
		make_loader(data_block.test.X, data_block.test.y, False),
	)


def matrix_loaders(
	circuits_dir: Path,
	batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""Load GEQIE-precomputed matrices stored in train/val/test directories."""
	from geqie_qml import MatrixDataset

	def make_loader(split: str, shuffle: bool) -> DataLoader:
		files = sorted((circuits_dir / split).glob("*.npz"))
		if not files:
			raise FileNotFoundError(f"No precomputed matrices found in {circuits_dir / split}.")
		return DataLoader(MatrixDataset([str(file) for file in files]), batch_size=batch_size, shuffle=shuffle)

	return make_loader("train", True), make_loader("val", False), make_loader("test", False)


def zip_matrix_loaders(
	zip_path: Path,
	batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""Load train/validation/test GEQIE matrices lazily from one ZIP archive."""
	from geqie_qml import load_precomputed_zip_matrices

	if not zip_path.is_file():
		raise FileNotFoundError(f"Precomputed GEQIE ZIP archive does not exist: {zip_path}")

	train_dataset, val_dataset, test_dataset = load_precomputed_zip_matrices(
		str(zip_path),
		split_names=["train", "val", "test"],
	)
	for split_name, dataset in (
		("train", train_dataset),
		("val", val_dataset),
		("test", test_dataset),
	):
		if len(dataset) == 0:
			raise ValueError(
				f"ZIP archive '{zip_path}' does not contain usable '{split_name}/' matrices."
			)

	return (
		DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
		DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
		DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
	)


ProgressCallback = Callable[[Mapping[str, Any]], None]


def evaluate_model(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: str,
	*,
	on_batch_complete: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
	model.eval()
	losses: list[float] = []
	predictions: list[int] = []
	targets: list[int] = []
	batch_count = len(loader)
	with torch.no_grad():
		for batch_idx, (x_batch, y_batch) in enumerate(loader, start=1):
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)
			output = model(x_batch)
			losses.append(criterion(output, y_batch).item())
			predictions.extend(torch.argmax(output, dim=1).cpu().tolist())
			targets.extend(y_batch.cpu().tolist())
			if on_batch_complete is not None:
				on_batch_complete(batch_idx, batch_count)
	metrics = classification_metrics(targets, predictions)
	return {"loss": float(np.mean(losses)) if losses else 0.0, **metrics, "y_true": targets, "y_pred": predictions}


def train_model(
	*,
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	test_loader: DataLoader,
	optimizer: Adam,
	num_classes: int,
	epochs: int,
	device: str,
	verbose: bool,
	report_context: dict[str, Any] | None,
	early_stopping: bool = True,
	patience: int = 5,
	min_delta: float = 1e-4,
	training_context: ContextManager[Any] | None = None,
	progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
	"""Train one subset and return the common result structure used by reports."""
	model = model.to(device)
	report_context = with_torchinfo_summary(report_context, model)
	criterion = nn.NLLLoss()
	history = history_template()
	best_loss = np.inf
	best_state: dict[str, Any] | None = None
	stale_epochs = 0
	context = training_context or nullcontext()
	train_batch_count = len(train_loader)
	val_batch_count = len(val_loader)
	test_batch_count = len(test_loader)
	epoch_batch_count = train_batch_count + val_batch_count
	total_steps = epochs * epoch_batch_count + test_batch_count
	completed_steps = 0
	completed_epochs = 0
	stopped_early = False

	def report_progress(
		phase: str,
		*,
		epoch: int | None = None,
		batch: int | None = None,
		phase_total: int | None = None,
		status: str = "running",
	) -> None:
		if progress_callback is None:
			return
		progress_callback({
			"phase": phase,
			"epoch": epoch,
			"epochs": epochs,
			"batch": batch,
			"phase_total": phase_total,
			"completed": completed_steps,
			"total": total_steps,
			"status": status,
			"early_stopping": stopped_early,
		})

	report_progress("starting", status="starting")

	with context:
		for epoch in range(epochs):
			model.train()
			losses: list[float] = []
			predictions: list[int] = []
			targets: list[int] = []
			for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
				x_batch, y_batch = x_batch.to(device), y_batch.to(device)
				optimizer.zero_grad()
				output = model(x_batch)
				loss = criterion(output, y_batch)
				loss.backward()
				optimizer.step()
				losses.append(loss.item())
				predictions.extend(torch.argmax(output, dim=1).detach().cpu().tolist())
				targets.extend(y_batch.detach().cpu().tolist())
				completed_steps += 1
				report_progress(
					"train",
					epoch=epoch + 1,
					batch=batch_idx,
					phase_total=train_batch_count,
				)

			train_metrics = classification_metrics(targets, predictions)
			train_loss = float(np.mean(losses)) if losses else 0.0

			def report_validation_batch(batch_idx: int, batch_count: int) -> None:
				nonlocal completed_steps
				completed_steps += 1
				report_progress(
					"validation",
					epoch=epoch + 1,
					batch=batch_idx,
					phase_total=batch_count,
				)

			validation = evaluate_model(
				model,
				val_loader,
				criterion,
				device,
				on_batch_complete=report_validation_batch,
			)
			completed_epochs = epoch + 1
			for name, value in (("loss", train_loss), *train_metrics.items()):
				history[f"train_{name}"].append(value)
			for name in ("loss", "accuracy", "precision", "recall", "f1"):
				history[f"val_{name}"].append(validation[name])

			if validation["loss"] < best_loss - min_delta:
				best_loss, best_state, stale_epochs = validation["loss"], copy.deepcopy(model.state_dict()), 0
			else:
				stale_epochs += 1

			if verbose:
				if epoch == 0:
					if report_context is not None:
						from experiments.QNN_integration.experiment_results import print_experiment_report
						print_experiment_report(**report_context)
					print_epoch_table_header()
				print_epoch_table_row(epoch + 1, epochs, train_loss, train_metrics, validation["loss"], validation)

			if early_stopping and stale_epochs >= patience:
				stopped_early = True
				total_steps = completed_steps + test_batch_count
				report_progress(
					"early stopping",
					epoch=epoch + 1,
					status="early_stopping",
				)
				if verbose:
					print(f"Early stopping at epoch {epoch + 1}.")
				break

		if verbose and history["train_loss"]:
			print_epoch_table_footer()
		if best_state is not None:
			model.load_state_dict(best_state)

		def report_test_batch(batch_idx: int, batch_count: int) -> None:
			nonlocal completed_steps
			completed_steps += 1
			report_progress(
				"test",
				epoch=completed_epochs,
				batch=batch_idx,
				phase_total=batch_count,
			)

		test = evaluate_model(
			model,
			test_loader,
			criterion,
			device,
			on_batch_complete=report_test_batch,
		)
		report_progress("complete", epoch=completed_epochs, status="complete")

	matrix = confusion_matrix(test["y_true"], test["y_pred"], labels=list(range(num_classes)))
	if verbose:
		print_metrics_report(
			title="TEST RESULTS",
			metrics={name: test[name] for name in ("loss", "accuracy", "precision", "recall", "f1")},
			matrix=matrix,
		)
	return {
		"model": model,
		"history": history,
		"report_context": report_context,
		"test_metrics": {name: test[name] for name in ("loss", "accuracy", "precision", "recall", "f1")},
		"confusion_matrix": matrix,
		"y_true": test["y_true"],
		"y_pred": test["y_pred"],
	}


def run_subsets(
	*,
	dataset: DataSet,
	trainer: Callable[..., dict[str, Any]],
	dataset_id: str,
	experiment_group: str,
	model_family: str,
	encoding_id: str,
	model_id: str,
	pipeline_name: str,
	classifier_name: str,
	model_architecture: str,
	num_classes: int,
	num_qubits: int,
	num_layers: int,
	epochs: int,
	batch_size: int,
	device: str,
	verbose: bool,
	save_results: bool = True,
	results_base_dir: str | Path | None = None,
	training_setup_extra: dict[str, Any] | None = None,
	subset_kwargs_factory: Callable[[int, DataBlock], dict[str, Any]] | None = None,
	max_workers: int | None = None,
	show_progress_bars: bool | None = None,
) -> dict[str, Any]:
	dataset_id = normalize_dataset_id(dataset_id)
	result_identity = {
		"dataset_id": dataset_id,
		"experiment_group": experiment_group,
		"model_family": model_family,
		"encoding_id": encoding_id,
		"model_id": model_id,
	}
	return train_subsets_with_process_pool(
		dataset=dataset,
		trainer=trainer,
		**result_identity,
		pipeline_name=pipeline_name,
		classifier_name=classifier_name,
		model_architecture=model_architecture,
		num_classes=num_classes,
		num_qubits=num_qubits,
		num_layers=num_layers,
		epochs=epochs,
		batch_size=batch_size,
		device=device,
		verbose=verbose,
		save_results=save_results,
		results_base_dir=results_base_dir,
		training_setup_extra={
			**result_identity,
			**(training_setup_extra or {}),
		},
		subset_trainer_kwargs_factory=subset_kwargs_factory,
		max_workers=max_workers,
		show_progress_bars=show_progress_bars,
	)


def precompute_geqie_dataset(
	dataset: DataSet,
	*,
	circuits_root: Path,
	encoding_method: str,
	number_of_workers: int = 1,
	encoding_params: dict[str, Any] | None = None,
) -> None:
	"""Precompute and package each subset as ``subset_N.zip``.

	The ZIP root contains direct ``train/``, ``val/``, and ``test/`` folders,
	which is the layout consumed by :func:`geqie_qml.load_precomputed_zip_matrices`.
	"""
	from geqie_qml import compute_and_save_circuits

	for index, block in enumerate(dataset.subsets, start=1):
		subset_dir = circuits_root / f"subset_{index}"
		for split_name in ("train", "val", "test"):
			split = getattr(block, split_name)
			compute_and_save_circuits(
				data=split.X,
				labels=split.y,
				save_dir=str(subset_dir / split_name),
				geqie_encoding=encoding_method,
				number_of_workers=number_of_workers,
				encoding_params=encoding_params or {},
			)

		zip_path = circuits_root / f"subset_{index}.zip"
		with zipfile.ZipFile(
			zip_path,
			mode="w",
			compression=zipfile.ZIP_DEFLATED,
		) as archive:
			for matrix_file in subset_dir.glob("*/*.npz"):
				archive.write(
					matrix_file,
					arcname=matrix_file.relative_to(subset_dir).as_posix(),
				)


class GEQIEFirstClassifier(nn.Module):
	"""GEQIE matrix input -> VQC -> dense classification head."""
	def __init__(
		self,
		*,
		num_qubits: int,
		num_layers: int,
		num_classes: int,
		ansatz_factory: Callable[..., Any],
		output_qubits: int | None = None,
		interpret: Callable[[int], int] | None = None,
		shots: int = 1024,
	) -> None:
		super().__init__()
		from geqie_qml import VQCLayer

		self.vqc = VQCLayer(
			num_qubits=num_qubits,
			num_layers=num_layers,
			shots=shots,
			ansatz_factory=ansatz_factory,
			output_qubits=output_qubits,
			interpret=interpret,
		)
		self.head = nn.Linear(
			self.vqc.output_size,
			num_classes,
		)
		self.log_softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.vqc(x)
		x = self.head(x)
		x = self.log_softmax(x)
		return x


def train_geqie_first_subset(
	*,
	circuits_dir: Path | None = None,
	zip_path: Path | None = None,
	num_classes: int,
	num_qubits: int,
	num_layers: int,
	epochs: int,
	batch_size: int,
	device: str,
	verbose: bool,
	report_context: dict[str, Any] | None,
	ansatz_factory: Callable[..., Any],
	output_qubits: int | None = None,
	interpret: Callable[[int], int] | None = None,
	quantum_workers: int = 1,
	progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
	"""Train a GEQIE-first model from a matrix directory or a ZIP archive."""
	if (circuits_dir is None) == (zip_path is None):
		raise ValueError("Provide exactly one of circuits_dir or zip_path.")
	model = GEQIEFirstClassifier(
		num_qubits=num_qubits,
		num_layers=num_layers,
		num_classes=num_classes,
		ansatz_factory=ansatz_factory,
		output_qubits=output_qubits,
		interpret=interpret,
	)
	if zip_path is not None:
		loaders = zip_matrix_loaders(zip_path, batch_size)
	else:
		loaders = matrix_loaders(circuits_dir, batch_size)
	optimizer = Adam([{"params": model.vqc.parameters(), "lr": 1e-3}, {"params": model.head.parameters(), "lr": 1e-2}])
	return train_model(
		model=model,
		train_loader=loaders[0],
		val_loader=loaders[1],
		test_loader=loaders[2],
		optimizer=optimizer,
		num_classes=num_classes,
		epochs=epochs,
		device=device,
		verbose=verbose,
		report_context=report_context,
		training_context=model.vqc.parallel_context(num_workers=quantum_workers),
		progress_callback=progress_callback,
	)
