"""Generate publication-ready figures for the experimental GEQIE models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from generate_baseline_architectures import (
	BlockPresentation,
	DiagramSpec,
	HERE,
	TraceModel,
	_sample_tensor,
	_standard_presentations,
	generate,
)


OUTPUT_DIR = HERE / "output" / "experiment" / "geqie"


def _direct_spec(encoding_id: str, model_id: str) -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.common import GEQIEFirstClassifier
	from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.direct_geqie import (
		ENCODING_QUBITS,
		MODEL_VARIANTS,
	)

	variant = MODEL_VARIANTS[model_id]
	num_qubits = ENCODING_QUBITS[encoding_id]
	num_layers = variant["num_layers"]
	model = GEQIEFirstClassifier(
		num_qubits=num_qubits,
		num_layers=num_layers,
		num_classes=10,
		ansatz_factory=variant["ansatz_factory"],
		output_qubits=variant["output_qubits"],
	).eval()

	encoding_proxy = nn.Linear(16 * 16, num_qubits, bias=False)
	encoding_separator = nn.ReLU()
	quantum_proxy = nn.Linear(num_qubits, model.head.in_features, bias=False)
	quantum_separator = nn.ReLU()
	trace_model = TraceModel(
		nn.Flatten(),
		encoding_proxy,
		encoding_separator,
		quantum_proxy,
		quantum_separator,
		model.head,
		model.log_softmax,
	).eval()
	presentations = _standard_presentations(trace_model)
	encoding_label = encoding_id.upper()
	presentations[encoding_proxy] = BlockPresentation(
		caption=(
			rf"\shortstack{{GEQIE/{encoding_label} encoding\\"
			rf"$16\!\times\!16 \rightarrow q={num_qubits}$}}"
		),
		kind="preprocess",
		width=2*14,
	)

	if model_id == "direct_vqc_dense":
		quantum_caption = (
			rf"\shortstack{{Variational quantum circuit\\"
			rf"$q={num_qubits}$, $L={num_layers}$\\"
			rf"${num_qubits} \rightarrow {model.head.in_features}$}}"
		)
	elif model_id == "adaptive_qnn_no_qnn_inspired_dense":
		quantum_caption = (
			r"\shortstack{Adaptive QCNN\\No-QNN-inspired convolution + pooling\\"
			rf"$q_{{in}}={num_qubits}$, $q_{{out}}={variant['output_qubits']}$}}"
		)
	else:
		quantum_caption = (
			r"\shortstack{Adaptive QCNN + QNN compression\\"
			r"No-QNN-inspired convolution + pooling\\"
			rf"$q_{{in}}={num_qubits}$, $q_{{out}}={variant['output_qubits']}$}}"
		)

	presentations[quantum_proxy] = BlockPresentation(
		caption=quantum_caption,
		kind="quantum",
		output_label=str(model.head.in_features),
		width=18,
		extra_offset=6,
	)
	classifier_offset = {
		"direct_vqc_dense": 5,
		"adaptive_qnn_no_qnn_inspired_dense": 17,
		"adaptive_qnn_no_qnn_inspired_qnn_compression_dense": 20,
	}[model_id]
	presentations[model.head] = BlockPresentation(
		caption=(
			rf"\shortstack{{Linear\\${model.head.in_features} \rightarrow 10$; LogSoftmax}}"
		),
		kind="classifier",
		output_label="10",
		banded=True,
		extra_offset=classifier_offset,
	)

	source = Path(
		f"experimental_pipelines/experiment/geqie/{encoding_id}/{model_id}.py"
	)
	return DiagramSpec(
		key=f"{encoding_id}_{model_id}",
		title=f"GEQIE/{encoding_label}: {variant['pipeline_name']}",
		source=source,
		model=model,
		trace_model=trace_model,
		input_tensor=_sample_tensor(),
		output_prefix="experiment_geqie",
		presentations=presentations,
	)


def _cnn_feature_maps_spec(encoding_id: str) -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.cnn_feature_maps_vqc_dense import (
		CNNFeatureMapsGEQIEVQCDenseClassifier,
	)

	encoding_params = {"bitrate": 4} if encoding_id == "neqr" else {}
	model = CNNFeatureMapsGEQIEVQCDenseClassifier(
		num_classes=10,
		num_qubits=None,
		num_layers=1,
		batch_size=16,
		convolution_depth=2,
		encoding_id=encoding_id,
		encoding_params=encoding_params,
	).eval()
	num_qubits = model.quantum.num_qubits
	feature_maps = model.quantum.feature_maps
	quantum_proxy = nn.Linear(feature_maps * 4 * 4, model.head.in_features, bias=False)
	quantum_separator = nn.ReLU()
	trace_model = TraceModel(
		*list(model.cnn.children()),
		nn.Flatten(),
		quantum_proxy,
		quantum_separator,
		model.head,
		model.log_softmax,
	).eval()
	presentations = _standard_presentations(trace_model)
	presentations[model.cnn[0]] = BlockPresentation(caption="", kind="conv")
	presentations[model.cnn[4]] = BlockPresentation(
		caption="", kind="conv", extra_offset=8
	)
	for activation in (model.cnn[2], model.cnn[6]):
		presentations[activation] = BlockPresentation(caption="", kind="activation")
	presentations[model.cnn[3]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + BN + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$1\!\times\!16\!\times\!16 \rightarrow 8\!\times\!8\!\times\!8$}"
		),
		kind="pool",
	)
	presentations[model.cnn[7]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + BN + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$8\!\times\!8\!\times\!8 \rightarrow 16\!\times\!4\!\times\!4$}"
		),
		kind="pool",
	)
	encoding_label = encoding_id.upper()
	per_map_outputs = 2**num_qubits
	presentations[quantum_proxy] = BlockPresentation(
		caption=(
			rf"\shortstack{{GEQIE/{encoding_label} + VQC per feature map\\"
			rf"$16$ maps; $q={num_qubits}$, $L=1$\\"
			rf"Flatten: $16\!\times\!{per_map_outputs} \rightarrow {model.head.in_features}$}}"
		),
		kind="quantum",
		output_label=str(model.head.in_features),
		width=20,
		extra_offset=8,
	)
	presentations[model.head] = BlockPresentation(
		caption=(
			rf"\shortstack{{Linear\\${model.head.in_features} \rightarrow 10$; LogSoftmax}}"
		),
		kind="classifier",
		output_label="10",
		banded=True,
		extra_offset=16,
	)
	return DiagramSpec(
		key=f"{encoding_id}_cnn_feature_maps_vqc_dense",
		title=f"CNN feature maps + GEQIE/{encoding_label} + VQC + dense",
		source=Path(
			f"experimental_pipelines/experiment/geqie/{encoding_id}/cnn_feature_maps_vqc_dense.py"
		),
		model=model,
		trace_model=trace_model,
		input_tensor=_sample_tensor(),
		output_prefix="experiment_geqie",
		presentations=presentations,
	)


BUILDERS: dict[str, Callable[[], DiagramSpec]] = {
	"frqi_direct_vqc_dense": lambda: _direct_spec("frqi", "direct_vqc_dense"),
	"frqi_adaptive_qnn_no_qnn_inspired_dense": lambda: _direct_spec(
		"frqi", "adaptive_qnn_no_qnn_inspired_dense"
	),
	"frqi_adaptive_qnn_no_qnn_inspired_qnn_compression_dense": lambda: _direct_spec(
		"frqi", "adaptive_qnn_no_qnn_inspired_qnn_compression_dense"
	),
	"frqi_cnn_feature_maps_vqc_dense": lambda: _cnn_feature_maps_spec("frqi"),
	"neqr_direct_vqc_dense": lambda: _direct_spec("neqr", "direct_vqc_dense"),
	"neqr_adaptive_qnn_no_qnn_inspired_dense": lambda: _direct_spec(
		"neqr", "adaptive_qnn_no_qnn_inspired_dense"
	),
	"neqr_cnn_feature_maps_vqc_dense": lambda: _cnn_feature_maps_spec("neqr"),
}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", choices=("all", *BUILDERS), default="all")
	parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
	parser.add_argument("--dpi", type=int, default=300)
	return parser.parse_args()


def main() -> int:
	args = _parse_args()
	if args.dpi < 300:
		raise ValueError("Publication output must use at least 300 DPI.")
	torch.manual_seed(0)
	selected = BUILDERS if args.model == "all" else {args.model: BUILDERS[args.model]}
	entries = []
	for key, builder in selected.items():
		print(f"Generating {key} ...", flush=True)
		entries.append(generate(builder(), args.output_dir.resolve(), args.dpi))

	manifest_path = args.output_dir.resolve() / "manifest.json"
	existing = {}
	if manifest_path.is_file() and args.model != "all":
		existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
		existing = {entry["key"]: entry for entry in existing_data.get("architectures", [])}
	for entry in entries:
		existing[entry["key"]] = entry
	if args.model == "all":
		existing = {entry["key"]: entry for entry in entries}
	manifest = {
		"generator": Path(__file__).name,
		"pytorch_version": torch.__version__,
		"pytorch2tikz_version": __import__("pytorch2tikz").__version__,
		"architectures": [existing[key] for key in sorted(existing)],
	}
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
	print(f"Generated {len(entries)} architecture(s); see output/experiment/geqie.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
