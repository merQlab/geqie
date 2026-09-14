"""Shared implementation of models that receive GEQIE-encoded images directly."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from experiments.QNN_integration.experimental_pipelines.common import (
	load_dataset,
	normalize_dataset_id,
	precompute_geqie_dataset,
	run_subsets,
	train_geqie_first_subset,
)
from geqie_qml.ansatze import (
	build_adaptive_qcnn_ansatz,
	build_adaptive_qcnn_with_QNN_compression_layer,
	default_vqc_ansatz,
)
from geqie_qml import QCNNOutputInterpret


MODEL_VARIANTS = {
	"direct_vqc_dense": {
		"ansatz_factory": default_vqc_ansatz,
		"output_qubits": None,
		"num_layers": 5,
		"pipeline_name": "Direct GEQIE + VQC + dense",
		"classifier_name": "GEQIE + VQC + dense",
		"architecture": "GEQIE matrices -> VQCLayer(default VQC) -> Dense -> LogSoftmax",
	},
	"adaptive_qnn_no_qnn_inspired_dense": {
		"ansatz_factory": build_adaptive_qcnn_ansatz,
		"output_qubits": 4,
		"num_layers": 5,
		"pipeline_name": "Adaptive QNN inspired by No-QNN + dense",
		"classifier_name": "GEQIE + adaptive QNN inspired by No-QNN + dense",
		"architecture": "GEQIE matrices -> adaptive QNN inspired by No-QNN -> Dense -> LogSoftmax",
	},
	"adaptive_qnn_no_qnn_inspired_dense_interpret": {
		"ansatz_factory": build_adaptive_qcnn_ansatz,
		"output_qubits": 4,
		"num_layers": 5,
		"interpret": QCNNOutputInterpret(4),
		"pipeline_name": "Adaptive QNN inspired by No-QNN + interpret + dense",
		"classifier_name": "GEQIE + adaptive QNN inspired by No-QNN + interpret + dense",
		"architecture": "GEQIE matrices -> adaptive QNN inspired by No-QNN -> interpret -> Dense -> LogSoftmax",
	},
	"adaptive_qnn_no_qnn_inspired_qnn_compression_dense": {
		"ansatz_factory": build_adaptive_qcnn_with_QNN_compression_layer,
		"output_qubits": 4,
		"num_layers": 1,
		"pipeline_name": "Adaptive QNN inspired by No-QNN with QNN compression + dense",
		"classifier_name": "GEQIE + adaptive QNN inspired by No-QNN + QNN compression + dense",
		"architecture": (
			"GEQIE matrices -> adaptive QNN inspired by No-QNN with trainable QNN compression "
			"-> Dense -> LogSoftmax"
		),
	},
}


SERVER_CIRCUITS_ROOT = Path("/mnt/data02/mkordasz/circuits")
SERVER_DATASET_DIRECTORIES = {
	"mnist_digits": "MNIST_Digits",
	"mnist_fashion": "MNIST_Fashion",
	"cifar_bw": "CIFAR-BW",
	"cifar_rgb": "CIFAR-RGB",
}


def default_server_zip_root(dataset_id: str, encoding_id: str) -> Path:
	"""Return the server-side archive directory for a dataset and encoding."""
	dataset_id = normalize_dataset_id(dataset_id)
	encoding_id = str(encoding_id).strip().upper()
	if not encoding_id:
		raise ValueError("encoding_id must not be empty.")
	dataset_directory = SERVER_DATASET_DIRECTORIES.get(dataset_id, dataset_id)
	return SERVER_CIRCUITS_ROOT / encoding_id / dataset_directory


def infer_direct_geqie_qubits(
	encoding_id: str,
	image: np.ndarray,
	encoding_params: dict[str, Any] | None = None,
) -> int:
	"""Infer the encoded matrix width from an actual dataset image."""
	encoding_id = str(encoding_id).strip().lower()
	if not encoding_id:
		raise ValueError("encoding_id must not be empty.")
	try:
		encoding_module = importlib.import_module(f"geqie.encodings.{encoding_id}")
	except ModuleNotFoundError as error:
		if error.name == f"geqie.encodings.{encoding_id}":
			raise ValueError(f"Unsupported GEQIE encoding_id: {encoding_id!r}.") from error
		raise

	image = np.asarray(image)
	if image.ndim not in (2, 3):
		raise ValueError(f"Expected one HW or HWC image; got shape {image.shape}.")
	params = dict(encoding_params or {})
	coordinate_qubits = int(np.ceil(np.log2(max(image.shape[:2]))))
	data_state = encoding_module.data_function(
		0,
		0,
		R=coordinate_qubits,
		image=image,
		**params,
	)
	map_operator = encoding_module.map_function(
		0,
		0,
		R=coordinate_qubits,
		image=image,
		**params,
	)
	num_qubits = data_state.num_qubits + map_operator.num_qubits
	initial_state = encoding_module.init_function(num_qubits, **params)
	if initial_state.num_qubits != num_qubits:
		raise ValueError(
			f"GEQIE/{encoding_id.upper()} creates an initial state with "
			f"{initial_state.num_qubits} qubits, but the image requires {num_qubits}."
		)
	return num_qubits


def train_one_subset(
	subset_idx: int,
	*,
	zip_path: str,
	model_id: str,
	num_classes=10,
	num_qubits=9,
	num_layers=5,
	epochs=50,
	batch_size=16,
	device="cpu",
	verbose=False,
	report_context=None,
	progress_callback=None,
	quantum_workers=1,
	**_,
):
	"""Train one subset for any direct-GEQIE model variant."""
	try:
		variant = MODEL_VARIANTS[model_id]
	except KeyError as error:
		raise ValueError(f"Unknown direct GEQIE model_id: {model_id!r}.") from error
	return train_geqie_first_subset(
		zip_path=Path(zip_path),
		num_classes=num_classes,
		num_qubits=num_qubits,
		num_layers=num_layers,
		epochs=epochs,
		batch_size=batch_size,
		device=device,
		verbose=verbose,
		report_context=report_context,
		progress_callback=progress_callback,
		ansatz_factory=variant["ansatz_factory"],
		output_qubits=variant["output_qubits"],
		interpret=variant.get("interpret"),
		quantum_workers=quantum_workers,
	)


def run_direct_geqie(
	*,
	encoding_id: str,
	model_id: str,
	dataset=None,
	dataset_id: str = "mnist_digits",
	create_circuits: bool = False,
	zip_root: str | Path | None = None,
	encoding_params: dict[str, Any] | None = None,
	precompute_workers: int = 1,
	quantum_workers: int = 32,
	num_qubits: int | None = None,
	num_layers: int | None = None,
	**overrides,
):
	"""Run a named direct-GEQIE architecture with a selected image encoding."""
	encoding_id = str(encoding_id).strip().lower()
	dataset_id = normalize_dataset_id(dataset_id)
	try:
		variant = MODEL_VARIANTS[model_id]
	except KeyError as error:
		raise ValueError(f"Unknown direct GEQIE model_id: {model_id!r}.") from error

	dataset = dataset or load_dataset(dataset_id)
	zip_root = (
		Path(zip_root)
		if zip_root is not None
		else default_server_zip_root(dataset_id, encoding_id)
	)
	encoding_params = dict(encoding_params or {})
	expected_qubits = infer_direct_geqie_qubits(
		encoding_id,
		dataset.subsets[0].train.X[0],
		encoding_params,
	)
	if num_qubits is None:
		num_qubits = expected_qubits
	elif num_qubits != expected_qubits:
		raise ValueError(
			f"GEQIE/{encoding_id.upper()} needs {expected_qubits} qubits for images "
			f"of shape {dataset.subsets[0].train.X.shape[1:]}; got {num_qubits}."
		)
	if create_circuits:
		precompute_geqie_dataset(
			dataset,
			circuits_root=zip_root,
			encoding_method=encoding_id,
			number_of_workers=precompute_workers,
			encoding_params=encoding_params,
		)

	run_options = {
		"num_classes": 10,
		"num_qubits": num_qubits,
		"num_layers": num_layers or variant["num_layers"],
		"epochs": 50,
		"batch_size": 16,
		"device": "cpu",
		"verbose": False,
		"show_progress_bars": True,
		"training_setup_extra": {
			"encoding_method": encoding_id,
			"encoding_params": encoding_params,
			"precompute_workers": precompute_workers,
			"quantum_workers": quantum_workers,
		},
		"subset_kwargs_factory": lambda index, _: {
			"zip_path": str(zip_root / f"subset_{index + 1}.zip"),
			"model_id": model_id,
			"quantum_workers": quantum_workers,
		},
	}
	run_options.update(overrides)
	return run_subsets(
		dataset=dataset,
		trainer=train_one_subset,
		dataset_id=dataset_id,
		experiment_group="experiment",
		model_family="geqie",
		encoding_id=encoding_id,
		model_id=model_id,
		pipeline_name=variant["pipeline_name"],
		classifier_name=variant["classifier_name"],
		model_architecture=f"GEQIE/{encoding_id.upper()} {variant['architecture']}",
		**run_options,
	)
