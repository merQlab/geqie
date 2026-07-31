"""Shared implementation of models that receive GEQIE-encoded images directly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.QNN_integration.experimental_pipelines.common import (
	DATASETS_DIR,
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

ENCODING_QUBITS = {
	"frqi": 9,
	"neqr": 12,
}


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
		quantum_workers=quantum_workers,
	)


def run_direct_geqie(
	*,
	encoding_id: str,
	model_id: str,
	dataset=None,
	dataset_id: str = "mnist_digits",
	create_circuits: bool = False,
	zip_root: Path | None = None,
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
	if encoding_id not in ENCODING_QUBITS:
		raise ValueError(f"Unsupported GEQIE encoding_id: {encoding_id!r}.")
	try:
		variant = MODEL_VARIANTS[model_id]
	except KeyError as error:
		raise ValueError(f"Unknown direct GEQIE model_id: {model_id!r}.") from error

	dataset = dataset or load_dataset(dataset_id)
	zip_root = zip_root or DATASETS_DIR / ".precomputed_zips" / dataset_id / encoding_id
	encoding_params = dict(encoding_params or {})
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
		"num_qubits": num_qubits or ENCODING_QUBITS[encoding_id],
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
