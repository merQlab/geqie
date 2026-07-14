"""Experiment F: GEQIE(NEQR, 4-bit) -> adaptive QCNN -> Dense."""

from __future__ import annotations

from pathlib import Path

try:
	from .common import DATASETS_DIR, load_mnist_digits_dataset, precompute_geqie_dataset, run_subsets, train_geqie_first_subset
except ImportError:
	import sys

	sys.path.insert(0, str(Path(__file__).resolve().parent))

	from common import DATASETS_DIR, load_mnist_digits_dataset, precompute_geqie_dataset, run_subsets, train_geqie_first_subset

from geqie_qml.ansatze import build_adaptive_qcnn_ansatz


def train_one_subset(
	subset_idx: int,
	*,
	circuits_dir: str,
	num_classes=10,
	num_qubits=12,
	num_layers=5,
	epochs=50,
	batch_size=16,
	device="cpu",
	verbose=False,
	report_context=None,
	quantum_workers=1,
	**_,
):
	return train_geqie_first_subset(
		circuits_dir=Path(circuits_dir),
		num_classes=num_classes,
		num_qubits=num_qubits,
		num_layers=num_layers,
		epochs=epochs,
		batch_size=batch_size,
		device=device,
		verbose=verbose,
		report_context=report_context,
		ansatz_factory=build_adaptive_qcnn_ansatz,
		output_qubits=4,
		quantum_workers=quantum_workers,
	)


def run(
	dataset=None,
	*,
	create_circuits=True,
	circuits_root: Path | None = None,
	precompute_workers=1,
	quantum_workers=1,
	**overrides,
):
	"""Run F; NEQR precomputation defaults to one worker to remain RAM-safe."""
	dataset = dataset or load_mnist_digits_dataset()
	circuits_root = circuits_root or DATASETS_DIR / "NEQR" / "circuits"
	if create_circuits:
		precompute_geqie_dataset(
			dataset,
			circuits_root=circuits_root,
			encoding_method="neqr",
			number_of_workers=precompute_workers,
			encoding_params={"bitrate": 4},
		)
	return run_subsets(
		dataset=dataset,
		trainer=train_one_subset,
		pipeline_name="Experiment F",
		classifier_name="NEQR (4-bit) + adaptive QCNN + Dense",
		model_architecture="GEQIE/NEQR matrices -> adaptive QCNN -> Dense -> LogSoftmax",
		num_classes=10,
		num_qubits=12,
		num_layers=5,
		epochs=50,
		batch_size=16,
		device="cpu",
		verbose=False,
		subset_kwargs_factory=lambda index, _: {
			"circuits_dir": str(circuits_root / f"subset_{index + 1}"),
			"quantum_workers": quantum_workers,
		},
		**overrides,
	)


if __name__ == "__main__":
	run()
