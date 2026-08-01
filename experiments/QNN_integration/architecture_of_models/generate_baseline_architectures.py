"""Generate publication-ready architecture figures for all baseline models.

The script uses pytorch2tikz to trace the standard PyTorch modules.  Operations
which live outside PyTorch's module graph (PCA and Qiskit's TorchConnector) are
represented by explicit proxy blocks with the correct input and output sizes.
They are only used while drawing; the model implementations are not changed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch2tikz import Architecture
from pytorch2tikz.constants import PICTYPE
from pytorch2tikz.utils import hex_to_tex_color


HERE = Path(__file__).resolve().parent


def _find_project_root() -> Path:
	for candidate in HERE.parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").is_dir():
			return candidate
	raise RuntimeError("Could not locate the GEQIE project root.")


PROJECT_ROOT = _find_project_root()
for import_root in (
	PROJECT_ROOT,
	PROJECT_ROOT / "geqie-qml" / "src",
	PROJECT_ROOT / "geqie" / "src",
):
	if str(import_root) not in sys.path:
		sys.path.insert(0, str(import_root))


OUTPUT_DIR = HERE / "output" / "baseline"
SAMPLE_IMAGE = PROJECT_ROOT / "assets" / "test_images" / "grayscale" / "mnist" / "5_16x16.png"

COLORS = {
	"CONV": "#4C78A8",
	"ACTIVATION": "#F58518",
	"DROPOUT": "#E45756",
	"POOL": "#E45756",
	"VEC_INPUT": "#72B7B2",
	"LINEAR": "#B279A2",
	"EMBEDDING": "#59A14F",
	"NORM": "#9D755D",
	"LSTM": "#79706E",
	"EDGE": "#3D405B",
}

SPECIAL_COLORS = {
	"preprocess": "#59A14F",
	"projection": "#76B7B2",
	"quantum": "#6F4C9B",
	"classifier": "#B279A2",
}


@dataclass(frozen=True)
class BlockPresentation:
	caption: str
	kind: str
	output_label: str | None = None
	banded: bool = False
	width: float | None = None
	extra_offset: float = 0.0


@dataclass
class DiagramSpec:
	key: str
	title: str
	source: Path
	model: nn.Module
	trace_model: nn.Module
	input_tensor: torch.Tensor
	output_prefix: str = "baseline"
	presentations: dict[nn.Module, BlockPresentation] = field(default_factory=dict)


class TraceModel(nn.Module):
	"""Drawing-only wrapper that keeps the root out of pytorch2tikz hooks."""

	def __init__(self, *layers: nn.Module) -> None:
		super().__init__()
		self.layers = nn.Sequential(*layers)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.layers(inputs)


def _sample_tensor() -> torch.Tensor:
	if not SAMPLE_IMAGE.is_file():
		raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMAGE}")
	with Image.open(SAMPLE_IMAGE) as image:
		image = image.convert("L").resize((16, 16))
		pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, 1, 16, 16)
	return pixels / 255.0


def _linear_caption(module: nn.Linear, *, activation: str | None = None) -> str:
	second_line = rf"${module.in_features} \rightarrow {module.out_features}$"
	if activation:
		second_line += rf" + {activation}"
	return rf"\shortstack{{Linear\\{second_line}}}"


def _standard_presentations(model: nn.Module) -> dict[nn.Module, BlockPresentation]:
	presentations: dict[nn.Module, BlockPresentation] = {}
	for module in model.modules():
		if isinstance(module, nn.Conv2d):
			kernel_h, kernel_w = module.kernel_size
			presentations[module] = BlockPresentation(
				caption=(
					rf"\shortstack{{Conv2d + ReLU\\"
					rf"${module.in_channels} \rightarrow {module.out_channels}$, "
					rf"${kernel_h}\!\times\!{kernel_w}$}}"
				),
				kind="conv",
			)
		elif isinstance(module, nn.MaxPool2d):
			kernel = module.kernel_size
			if isinstance(kernel, tuple):
				kernel_label = rf"{kernel[0]}\!\times\!{kernel[1]}"
			else:
				kernel_label = rf"{kernel}\!\times\!{kernel}"
			presentations[module] = BlockPresentation(
				caption=rf"\shortstack{{MaxPool2d\\${kernel_label}$}}",
				kind="pool",
			)
		elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
			presentations[module] = BlockPresentation(
				caption="",
				kind="norm",
			)
		elif isinstance(module, nn.Linear):
			presentations[module] = BlockPresentation(
				caption=_linear_caption(module),
				kind="classifier",
				output_label=str(module.out_features),
				banded=True,
			)
	return presentations


def _dense_spec() -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.baseline.classical.dense_classifier import (
		DenseClassifier,
	)

	model = DenseClassifier(num_classes=10).eval()
	presentations = _standard_presentations(model)
	presentations[model.classifier] = BlockPresentation(
		caption=r"\shortstack{Flatten + Linear\\$256 \rightarrow 10$; LogSoftmax}",
		kind="classifier",
		output_label="10 classes",
		banded=True,
	)
	return DiagramSpec(
		key="classical_dense_classifier",
		title="Dense classifier",
		source=Path("experimental_pipelines/baseline/classical/dense_classifier.py"),
		model=model,
		trace_model=model,
		input_tensor=_sample_tensor(),
		presentations=presentations,
	)


def _cnn_spec() -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.baseline.classical.cnn_classifier import (
		CNNClassifier,
	)

	model = CNNClassifier(num_classes=10).eval()
	presentations = _standard_presentations(model)
	presentations[model.features[0]] = BlockPresentation(caption="", kind="conv")
	presentations[model.features[3]] = BlockPresentation(
		caption="", kind="conv", extra_offset=4
	)
	presentations[model.features[2]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$1\!\times\!16\!\times\!16 \rightarrow 16\!\times\!8\!\times\!8$}"
		),
		kind="pool",
	)
	presentations[model.features[5]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$16\!\times\!8\!\times\!8 \rightarrow 32\!\times\!4\!\times\!4$}"
		),
		kind="pool",
	)
	first_linear = model.classifier[1]
	last_linear = model.classifier[3]
	presentations[first_linear] = BlockPresentation(
		caption=r"\shortstack{Flatten + Linear\\$512 \rightarrow 64$; ReLU}",
		kind="projection",
		output_label="64",
		banded=True,
		extra_offset=4,
	)
	presentations[last_linear] = BlockPresentation(
		caption=r"\shortstack{Linear\\$64 \rightarrow 10$; LogSoftmax}",
		kind="classifier",
		output_label="10 classes",
		banded=True,
	)
	return DiagramSpec(
		key="classical_cnn_classifier",
		title="CNN classifier",
		source=Path("experimental_pipelines/baseline/classical/cnn_classifier.py"),
		model=model,
		trace_model=model,
		input_tensor=_sample_tensor(),
		presentations=presentations,
	)


def _pca_quantum_spec() -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.baseline.quantum_without_geqie.pca_zz_vqc_dense import (
		PCAZZVQCDenseClassifier,
	)

	model = PCAZZVQCDenseClassifier(num_qubits=12, num_layers=1, num_classes=10).eval()
	pca_proxy = nn.Linear(256, model.num_qubits, bias=False)
	pca_separator = nn.ReLU()
	quantum_proxy = nn.Linear(model.num_qubits, 2**model.num_qubits, bias=False)
	quantum_separator = nn.ReLU()
	trace_model = TraceModel(
		nn.Flatten(),
		pca_proxy,
		pca_separator,
		quantum_proxy,
		quantum_separator,
		model.head,
		model.log_softmax,
	).eval()
	presentations = _standard_presentations(trace_model)
	presentations[pca_proxy] = BlockPresentation(
		caption=r"\shortstack{Flatten + PCA\\$256 \rightarrow 12$}",
		kind="preprocess",
		output_label="12",
		width=10,
	)
	presentations[quantum_proxy] = BlockPresentation(
		caption=r"\shortstack{$ZZ$ feature map + VQC\\$q=12$, $L=1$\\$12 \rightarrow 4096$}",
		kind="quantum",
		output_label=r"$2^{12}=4096$",
		width=16,
	)
	presentations[model.head] = BlockPresentation(
		caption=r"\shortstack{Linear\\$4096 \rightarrow 10$; LogSoftmax}",
		kind="classifier",
		output_label="10 classes",
		banded=True,
	)
	return DiagramSpec(
		key="quantum_pca_zz_vqc_dense",
		title="PCA + ZZ feature map + VQC + dense",
		source=Path("experimental_pipelines/baseline/quantum_without_geqie/pca_zz_vqc_dense.py"),
		model=model,
		trace_model=trace_model,
		input_tensor=_sample_tensor(),
		presentations=presentations,
	)


def _cnn_zz_quantum_spec() -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.baseline.quantum_without_geqie.cnn_zz_vqc_dense import (
		CNNZZVQCDenseClassifier,
	)

	model = CNNZZVQCDenseClassifier(num_qubits=12, num_layers=1, num_classes=10).eval()
	cnn_layers = list(model.cnn.children())
	projection = cnn_layers[-1]
	projection_separator = nn.ReLU()
	quantum_proxy = nn.Linear(model.num_qubits, 2**model.num_qubits, bias=False)
	quantum_separator = nn.ReLU()
	trace_model = TraceModel(
		*cnn_layers,
		projection_separator,
		quantum_proxy,
		quantum_separator,
		model.head,
		model.log_softmax,
	).eval()
	presentations = _standard_presentations(trace_model)
	presentations[model.cnn[0]] = BlockPresentation(caption="", kind="conv")
	presentations[model.cnn[3]] = BlockPresentation(
		caption="", kind="conv", extra_offset=6
	)
	presentations[model.cnn[2]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$1\!\times\!16\!\times\!16 \rightarrow 16\!\times\!8\!\times\!8$}"
		),
		kind="pool",
	)
	presentations[model.cnn[5]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($3\!\times\!3$) + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$16\!\times\!8\!\times\!8 \rightarrow 32\!\times\!4\!\times\!4$}"
		),
		kind="pool",
	)
	presentations[projection] = BlockPresentation(
		caption=r"\shortstack{Flatten + Linear\\$512 \rightarrow 12$}",
		kind="projection",
		output_label="12",
		extra_offset=7,
	)
	presentations[quantum_proxy] = BlockPresentation(
		caption=r"\shortstack{$ZZ$ feature map + VQC\\$q=12$, $L=1$\\$12 \rightarrow 4096$}",
		kind="quantum",
		output_label=r"$2^{12}=4096$",
		width=16,
	)
	presentations[model.head] = BlockPresentation(
		caption=r"\shortstack{Linear\\$4096 \rightarrow 10$; LogSoftmax}",
		kind="classifier",
		output_label="10 classes",
		banded=True,
	)
	return DiagramSpec(
		key="quantum_cnn_zz_vqc_dense",
		title="CNN + ZZ feature map + VQC + dense",
		source=Path("experimental_pipelines/baseline/quantum_without_geqie/cnn_zz_vqc_dense.py"),
		model=model,
		trace_model=trace_model,
		input_tensor=_sample_tensor(),
		presentations=presentations,
	)


def _cnn_angle_quantum_spec() -> DiagramSpec:
	from experiments.QNN_integration.experimental_pipelines.baseline.quantum_without_geqie.cnn_angle_vqc_dense_senokosov2024 import (
		CNNAngleVQCDenseSenokosov2024,
	)

	model = CNNAngleVQCDenseSenokosov2024(num_qubits=9, num_layers=1, num_classes=10).eval()
	cnn_layers = list(model.cnn.children())
	feature_layers = list(model.feature_head.children())
	projection = model.feature_head[1]
	quantum_proxy = nn.Linear(model.num_qubits, 2**model.num_qubits, bias=False)
	quantum_separator = nn.ReLU()
	trace_model = TraceModel(
		*cnn_layers,
		*feature_layers,
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
	presentations[model.cnn[3]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($5\!\times\!5$) + BN + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$1\!\times\!16\!\times\!16 \rightarrow 16\!\times\!8\!\times\!8$}"
		),
		kind="pool",
	)
	presentations[model.cnn[7]] = BlockPresentation(
		caption=(
			r"\shortstack{Conv($5\!\times\!5$) + BN + ReLU\\MaxPool($2\!\times\!2$)\\"
			r"$16\!\times\!8\!\times\!8 \rightarrow 32\!\times\!4\!\times\!4$}"
		),
		kind="pool",
	)
	for activation in (model.cnn[2], model.cnn[6], model.feature_head[3]):
		presentations[activation] = BlockPresentation(caption="", kind="activation")
	presentations[projection] = BlockPresentation(
		caption=r"\shortstack{Flatten + Linear\\BN + ReLU\\$512 \rightarrow 9$}",
		kind="projection",
		output_label="9",
		banded=True,
		extra_offset=8,
	)
	presentations[quantum_proxy] = BlockPresentation(
		caption=r"\shortstack{$R_x$ embedding + VQC\\$q=9$, $L=1$\\$9 \rightarrow 512$}",
		kind="quantum",
		output_label=r"$2^{9}=512$",
		width=16,
	)
	presentations[model.head] = BlockPresentation(
		caption=r"\shortstack{Linear\\$512 \rightarrow 10$; LogSoftmax}",
		kind="classifier",
		output_label="10 classes",
		banded=True,
	)
	return DiagramSpec(
		key="quantum_cnn_angle_vqc_dense_senokosov2024",
		title="CNN + angle embedding + VQC + dense (Senokosov 2024)",
		source=Path(
			"experimental_pipelines/baseline/quantum_without_geqie/"
			"cnn_angle_vqc_dense_senokosov2024.py"
		),
		model=model,
		trace_model=trace_model,
		input_tensor=_sample_tensor(),
		presentations=presentations,
	)


BUILDERS: dict[str, Callable[[], DiagramSpec]] = {
	"classical_dense_classifier": _dense_spec,
	"classical_cnn_classifier": _cnn_spec,
	"quantum_pca_zz_vqc_dense": _pca_quantum_spec,
	"quantum_cnn_zz_vqc_dense": _cnn_zz_quantum_spec,
	"quantum_cnn_angle_vqc_dense_senokosov2024": _cnn_angle_quantum_spec,
}


def _style_block(block, presentation: BlockPresentation) -> None:
	# pytorch2tikz emits scalar values without braces.  Group captions and
	# explicit xcolor expressions because both can contain pgfkeys commas.
	caption = rf"\scriptsize {presentation.caption}" if presentation.caption else ""
	block.args["caption"] = "{" + caption + "}"
	block.args["opacity"] = 0.82
	if presentation.kind in SPECIAL_COLORS:
		block.args["fill"] = "{" + hex_to_tex_color(SPECIAL_COLORS[presentation.kind]) + "}"
	if not presentation.banded:
		block.pictype = PICTYPE.BOX
		block.args.pop("bandfill", None)
	if presentation.width is not None:
		block.args["width"] = presentation.width
		block.scale_factor[0] = 0
	if presentation.extra_offset:
		block.offset[0] += presentation.extra_offset
	# Dimension labels generated by the package are diagonal and collide with
	# captions in compact article figures.  The exact transformations are
	# already included in the captions, so suppress all three axis labels.
	block.xlabel = False
	block.ylabel = False
	block.zlabel = False
	block.args.pop("xlabel", None)
	block.args.pop("ylabel", None)
	block.args.pop("zlabel", None)
	if presentation.output_label is not None:
		block.args["height"] = 28
		block.args["depth"] = 28
		block.scale_factor[1:] = 0


def _compile_tex(tex_path: Path) -> Path:
	pdflatex = shutil.which("pdflatex")
	if pdflatex is None:
		raise RuntimeError("pdflatex was not found on PATH.")
	command = [
		pdflatex,
		"-interaction=nonstopmode",
		"-halt-on-error",
		"-file-line-error",
		tex_path.name,
	]
	result = subprocess.run(
		command,
		cwd=tex_path.parent,
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		encoding="utf-8",
		errors="replace",
		check=False,
	)
	if result.returncode != 0:
		raise RuntimeError(f"LaTeX compilation failed for {tex_path.name}:\n{result.stdout}")
	for suffix in (".aux", ".log"):
		auxiliary = tex_path.with_suffix(suffix)
		if auxiliary.exists():
			auxiliary.unlink()
	return tex_path.with_suffix(".pdf")


def _rasterize_pdf(pdf_path: Path, dpi: int) -> tuple[Path, tuple[int, int]]:
	pdftoppm = shutil.which("pdftoppm")
	if pdftoppm is None:
		raise RuntimeError("pdftoppm was not found on PATH.")
	# The Codex desktop runtime exposes a lightweight .cmd shim.  Some Windows
	# installations of that shim cannot resolve its relative target, although
	# the bundled Poppler executable is present.  Prefer that executable when
	# this known layout is detected; normal user installations are unchanged.
	pdftoppm_path = Path(pdftoppm)
	if pdftoppm_path.suffix.lower() == ".cmd" and len(pdftoppm_path.parents) >= 3:
		bundled_executable = (
			pdftoppm_path.parents[2] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
		)
		if bundled_executable.is_file():
			pdftoppm = str(bundled_executable)
	png_path = pdf_path.with_suffix(".png")
	result = subprocess.run(
		[
			pdftoppm,
			"-png",
			"-singlefile",
			"-r",
			str(dpi),
			str(pdf_path),
			str(png_path.with_suffix("")),
		],
		text=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		encoding="utf-8",
		errors="replace",
		check=False,
	)
	if result.returncode != 0 or not png_path.is_file():
		raise RuntimeError(f"PDF rasterization failed for {pdf_path.name}:\n{result.stdout}")
	with Image.open(png_path) as image:
		pixel_size = image.size
		image.save(png_path, dpi=(dpi, dpi), optimize=True)
	return png_path, pixel_size


def generate(spec: DiagramSpec, output_dir: Path, dpi: int) -> dict[str, object]:
	output_dir.mkdir(parents=True, exist_ok=True)
	stem = f"{spec.output_prefix}_{spec.key}"
	tex_path = output_dir / f"{stem}.tex"
	# Remove only artifacts owned by this model so reruns cannot leave stale
	# input images or LaTeX auxiliaries behind.
	for stale_path in output_dir.glob(f"{stem}_input_*.png"):
		stale_path.unlink()
	for suffix in (".aux", ".log"):
		stale_path = output_dir / f"{stem}{suffix}"
		if stale_path.exists():
			stale_path.unlink()
	architecture = Architecture(
		spec.trace_model,
		block_offset=10,
		height_depth_factor=0.55,
		width_factor=0.55,
		linear_factor=0.92,
		image_path=str(output_dir / f"{stem}_input_{{i}}.png"),
		ignore_layers=["flatten"],
		colors=COLORS,
	)
	# pytorch2tikz 0.0.1 leaves this value unset until it encounters a mild
	# shape change.  A direct q -> 2**q transition reaches the large-change
	# branch first and otherwise causes a NoneType assignment error.
	architecture._block_sequence.block_factory.scale_factor = np.zeros(3)
	try:
		with torch.inference_mode():
			spec.trace_model(spec.input_tensor)
		seen_modules = architecture._block_sequence._seen_modules
		for module, presentation in spec.presentations.items():
			block = seen_modules.get(module)
			if block is None:
				raise RuntimeError(
					f"pytorch2tikz did not create a block for {module!r} in {spec.key}."
				)
			_style_block(block, presentation)
		# Make the source image legible at article scale.  pytorch2tikz maps
		# literal 16x16 pixels to only 0.4 cm by default.
		for block in architecture._block_sequence.blocks:
			if block.__class__.__name__ == "ImgInputBlock":
				block.args["height"] = 48
				block.args["depth"] = 48
		tex_source = architecture.get_tex()
		# NumPy 2.x changed scalar repr inside tuples from ``1.0`` to
		# ``np.float64(1.0)``.  pytorch2tikz interpolates that repr directly into
		# TikZ coordinates, so normalize it until the upstream package does so.
		tex_source = re.sub(r"np\.float64\(([^)]+)\)", r"\1", tex_source)
		blocks = architecture._block_sequence.blocks
		if len(blocks) >= 2 and blocks[0].__class__.__name__ == "ImgInputBlock":
			input_connection = (
				rf"\draw [connection] ({blocks[0].name}.east) -- node {{\midarrow}} "
				rf"({blocks[1].name}-west);"
				"\n"
				rf"\node[below=5pt of {blocks[0].name}, font=\bfseries\scriptsize, align=center] "
				r"{Input\\$1\!\times\!16\!\times\!16$};"
				"\n"
			)
			tex_source = tex_source.replace(r"\end{tikzpicture}", input_connection + r"\end{tikzpicture}")
		# PlotNeuralNet's caption nodes do not reliably enlarge standalone's
		# bounding box.  Add transparent right padding so the last caption is not
		# clipped in either PDF or PNG output.
		tex_source = tex_source.replace(
			r"\end{tikzpicture}",
			r"\path (current bounding box.east) ++(3cm,0) node[inner sep=0pt] {};"
			"\n"
			r"\end{tikzpicture}",
		)
		tex_path.write_text(tex_source, encoding="utf-8")
	finally:
		architecture.remove_handles()

	pdf_path = _compile_tex(tex_path)
	png_path, pixel_size = _rasterize_pdf(pdf_path, dpi)
	return {
		"key": spec.key,
		"title": spec.title,
		"source": spec.source.as_posix(),
		"tex": tex_path.name,
		"pdf": pdf_path.name,
		"png": png_path.name,
		"dpi": dpi,
		"pixels": list(pixel_size),
	}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--model",
		choices=("all", *BUILDERS),
		default="all",
		help="Generate one selected baseline architecture or all of them.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=OUTPUT_DIR,
		help=f"Output directory (default: {OUTPUT_DIR}).",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=300,
		help="PNG rasterization resolution; values below 300 are rejected.",
	)
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
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
	print(f"Generated {len(entries)} architecture(s); see output/baseline.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
