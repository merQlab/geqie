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
# pytorch2tikz must trace concrete tensor sizes even though the publication
# diagrams describe the encoding-independent architecture symbolically.  These
# values are drawing-only representatives and are never exposed as an encoding.
TRACE_DIRECT_QUBITS = 9
TRACE_FEATURE_MAP_QUBITS = 5


def _direct_spec(model_id: str) -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.common import GEQIEFirstClassifier
	from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.direct_geqie import (
		MODEL_VARIANTS,
	)

	variant = MODEL_VARIANTS[model_id]
	num_layers = variant["num_layers"]
	model = GEQIEFirstClassifier(
		num_qubits=TRACE_DIRECT_QUBITS,
		num_layers=num_layers,
		num_classes=10,
		ansatz_factory=variant["ansatz_factory"],
		output_qubits=variant["output_qubits"],
	).eval()

	encoding_proxy = nn.Linear(16 * 16, TRACE_DIRECT_QUBITS, bias=False)
	encoding_separator = nn.ReLU()
	quantum_proxy = nn.Linear(TRACE_DIRECT_QUBITS, model.head.in_features, bias=False)
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
	presentations[encoding_proxy] = BlockPresentation(
		caption=r"\shortstack{GEQIE encoding\\$16\!\times\!16 \rightarrow q$}",
		kind="preprocess",
		width=2*14,
	)

	if model_id == "direct_vqc_dense":
		quantum_caption = (
			r"\shortstack{Variational quantum circuit\\"
			rf"$q$ qubits, $L={num_layers}$\\"
			r"$q \rightarrow 2^q$}"
		)
	elif model_id == "adaptive_qnn_no_qnn_inspired_dense":
		quantum_caption = (
			r"\shortstack{Adaptive QCNN\\No-QNN-inspired convolution + pooling\\"
			rf"$q_{{in}}=q$, $q_{{out}}={variant['output_qubits']}$}}"
		)
	else:
			quantum_caption = (
			r"\shortstack{Adaptive QCNN + QNN compression\\"
			r"No-QNN-inspired convolution + pooling\\"
			rf"$q_{{in}}=q$, $q_{{out}}={variant['output_qubits']}$}}"
		)

	presentations[quantum_proxy] = BlockPresentation(
		caption=quantum_caption,
		kind="quantum",
		output_label=r"$2^q$",
		width=18,
		extra_offset=6,
	)
	classifier_offset = {
		"direct_vqc_dense": 5,
		"adaptive_qnn_no_qnn_inspired_dense": 17,
		"adaptive_qnn_no_qnn_inspired_qnn_compression_dense": 20,
	}[model_id]
	presentations[model.head] = BlockPresentation(
		caption=r"\shortstack{Linear\\$2^q \rightarrow 10$; LogSoftmax}",
		kind="classifier",
		output_label="10",
		banded=True,
		extra_offset=classifier_offset,
	)

	return DiagramSpec(
		key=model_id,
		title=variant["pipeline_name"],
		source=Path("experimental_pipelines/experiment/geqie/models/direct_geqie.py"),
		model=model,
		trace_model=trace_model,
		input_tensor=_sample_tensor(),
		output_prefix="experiment_geqie",
		presentations=presentations,
	)


def _cnn_feature_maps_spec() -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.cnn_feature_maps_vqc_dense import (
		build_feature_extractor,
	)

	cnn, feature_maps, feature_size = build_feature_extractor(depth=2)
	trace_output_features = feature_maps * (2**TRACE_FEATURE_MAP_QUBITS)
	quantum_proxy = nn.Linear(
		feature_maps * feature_size[0] * feature_size[1],
		trace_output_features,
		bias=False,
	)
	quantum_separator = nn.ReLU()
	head = nn.Linear(trace_output_features, 10)
	log_softmax = nn.LogSoftmax(dim=-1)
	trace_model = TraceModel(
		*list(cnn.children()),
		nn.Flatten(),
		quantum_proxy,
		quantum_separator,
		head,
		log_softmax,
	).eval()
	presentations = _standard_presentations(trace_model)
	presentations[cnn[0]] = BlockPresentation(caption="", kind="conv")
	presentations[cnn[4]] = BlockPresentation(
		caption="", kind="conv", extra_offset=8
	)
	for activation in (cnn[2], cnn[6]):
		presentations[activation] = BlockPresentation(caption="", kind="activation")
	presentations[cnn[3]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + BN + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$1\!\times\!16\!\times\!16 \rightarrow 8\!\times\!8\!\times\!8$}"
		),
		kind="pool",
	)
	presentations[cnn[7]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + BN + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$8\!\times\!8\!\times\!8 \rightarrow 16\!\times\!4\!\times\!4$}"
		),
		kind="pool",
	)
	presentations[quantum_proxy] = BlockPresentation(
		caption=(
			r"\shortstack{GEQIE + VQC per feature map\\"
			r"$16$ maps; $q$ qubits, $L=1$\\"
			r"Flatten: $16\!\times\!2^q \rightarrow 16\!\cdot\!2^q$}"
		),
		kind="quantum",
		output_label=r"$16\cdot2^q$",
		width=20,
		extra_offset=8,
	)
	presentations[head] = BlockPresentation(
		caption=r"\shortstack{Linear\\$16\!\cdot\!2^q \rightarrow 10$; LogSoftmax}",
		kind="classifier",
		output_label="10",
		banded=True,
		extra_offset=16,
	)
	return DiagramSpec(
		key="cnn_feature_maps_vqc_dense",
		title="CNN feature maps + GEQIE + VQC + dense",
		source=Path("experimental_pipelines/experiment/geqie/models/cnn_feature_maps_vqc_dense.py"),
		model=trace_model,
		trace_model=trace_model,
		input_tensor=_sample_tensor(),
		output_prefix="experiment_geqie",
		presentations=presentations,
	)


BUILDERS: dict[str, Callable[[], DiagramSpec]] = {
	"direct_vqc_dense": lambda: _direct_spec("direct_vqc_dense"),
	"adaptive_qnn_no_qnn_inspired_dense": lambda: _direct_spec(
		"adaptive_qnn_no_qnn_inspired_dense"
	),
	"adaptive_qnn_no_qnn_inspired_qnn_compression_dense": lambda: _direct_spec(
		"adaptive_qnn_no_qnn_inspired_qnn_compression_dense"
	),
	"cnn_feature_maps_vqc_dense": _cnn_feature_maps_spec,
}


def _remove_stale_architectures(output_dir: Path) -> None:
	"""Remove outputs from architecture keys superseded by the current set."""
	active_stems = {f"experiment_geqie_{key}" for key in BUILDERS}
	for path in output_dir.glob("experiment_geqie_*"):
		is_active = any(
			path.name == f"{stem}{suffix}"
			or path.name.startswith(f"{stem}_input_")
			for stem in active_stems
			for suffix in (".tex", ".pdf", ".png", ".aux", ".log")
		)
		if path.is_file() and not is_active:
			path.unlink()


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
	if args.model == "all":
		args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
		_remove_stale_architectures(args.output_dir.resolve())
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
