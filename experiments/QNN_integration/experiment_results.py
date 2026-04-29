from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPORT_WIDTH = 118
EPOCH_COLUMNS = [
	("Epoch", 9),
	("TrLoss", 8),
	("TrAcc", 8),
	("TrPrec", 8),
	("TrRec", 8),
	("TrF1", 8),
	("VaLoss", 8),
	("VaAcc", 8),
	("VaPrec", 8),
	("VaRec", 8),
	("VaF1", 8),
]

HISTORY_COLUMNS = [
	("Epoch", None),
	("TrLoss", "train_loss"),
	("TrAcc", "train_accuracy"),
	("TrPrec", "train_precision"),
	("TrRec", "train_recall"),
	("TrF1", "train_f1"),
	("VaLoss", "val_loss"),
	("VaAcc", "val_accuracy"),
	("VaPrec", "val_precision"),
	("VaRec", "val_recall"),
	("VaF1", "val_f1"),
]

SUMMARY_COLUMNS = [
	("accuracy", "accuracy_mean", "accuracy_std"),
	("precision", "precision_mean", "precision_std"),
	("recall", "recall_mean", "recall_std"),
	("f1", "f1_mean", "f1_std"),
]


def _stringify_report_value(value: Any) -> str:
	value = _to_builtin_scalar(value)
	if isinstance(value, float):
		return f"{value:.4f}"
	if isinstance(value, (list, tuple)):
		return ", ".join(_stringify_report_value(item) for item in value)
	return str(value)


def _format_mapping(mapping: Mapping[str, Any]) -> str:
	return ", ".join(
		f"{key}={_stringify_report_value(value)}"
		for key, value in mapping.items()
	)


def _wrap_report_line(prefix: str, value: Any, width: int = REPORT_WIDTH) -> list[str]:
	inner_width = width - 4
	text = f"{prefix}{_stringify_report_value(value)}"
	lines = []

	while len(text) > inner_width:
		split_at = text.rfind(" ", 0, inner_width + 1)
		if split_at <= len(prefix):
			split_at = inner_width
		lines.append(text[:split_at])
		text = (" " * len(prefix)) + text[split_at:].lstrip()

	lines.append(text)
	return [f"# {line.ljust(inner_width)} #" for line in lines]


def _render_report_box(
	title: str,
	rows: Sequence[tuple[str, Any]],
	extra_lines: Sequence[str] | None = None,
	width: int = REPORT_WIDTH,
) -> str:
	inner_width = width - 4
	lines = [
		"#" * width,
		f"# {title.center(inner_width)} #",
		"#" * width,
	]

	for label, value in rows:
		lines.extend(_wrap_report_line(f"{label:<10}: ", value, width=width))

	for line in extra_lines or []:
		lines.extend(_wrap_report_line("", line, width=width))

	lines.append("#" * width)
	return "\n".join(lines)


def _make_table_border(fill: str = "-") -> str:
	return "+" + "+".join(fill * width for _, width in EPOCH_COLUMNS) + "+"


def _fit_table_value(value: Any, width: int) -> str:
	text = _stringify_report_value(value)
	if len(text) > width:
		return text[:width]
	return text.rjust(width)


def _to_builtin_scalar(value: Any) -> Any:
	if hasattr(value, "item"):
		try:
			return value.item()
		except ValueError:
			return value
	return value


def _to_serializable(value: Any) -> Any:
	value = _to_builtin_scalar(value)
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Mapping):
		return {str(key): _to_serializable(val) for key, val in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_serializable(item) for item in value]
	return value


def _csv_value(value: Any) -> Any:
	value = _to_builtin_scalar(value)
	if isinstance(value, float):
		return f"{value:.10g}"
	return value


def _history_length(history: Mapping[str, Sequence[Any]]) -> int:
	lengths = [len(values) for values in history.values() if hasattr(values, "__len__")]
	return max(lengths, default=0)


def _history_value(history: Mapping[str, Sequence[Any]], key: str, index: int) -> Any:
	values = history.get(key, [])
	if index >= len(values):
		return ""
	return values[index]


def _history_row_values(
	history: Mapping[str, Sequence[Any]],
	index: int,
	total_epochs: int | None = None,
) -> list[Any]:
	epoch = index + 1
	epoch_value = f"{epoch:03d}/{total_epochs:03d}" if total_epochs else f"{epoch:03d}"
	values = [epoch_value]
	for _, key in HISTORY_COLUMNS[1:]:
		values.append(_history_value(history, key, index))
	return values


def count_trainable_parameters(model: Any) -> int:
	return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def render_experiment_report(
	title: str,
	dataset_name: str,
	classifier_name: str,
	subset_name: str,
	split_sizes: Mapping[str, Any],
	training_setup: Mapping[str, Any],
	model_setup: Mapping[str, Any] | None = None,
	model_architecture: str | None = None,
	trainable_parameters: int | None = None,
	now: datetime | None = None,
) -> str:
	now = now or datetime.now()
	rows = [
		("DATE", now.strftime("%d-%m-%Y")),
		("TIME", now.strftime("%H:%M")),
		("DATASET", dataset_name),
		("CLASSIFIER", classifier_name),
		("SUBSET", subset_name),
		("SPLITS", _format_mapping(split_sizes)),
		("TRAINING", _format_mapping(training_setup)),
	]

	if model_architecture:
		rows.append(("ARCHITECT.", model_architecture))

	if trainable_parameters is not None:
		rows.append(("PARAMS", trainable_parameters))

	if model_setup:
		rows.append(("MODEL", _format_mapping(model_setup)))

	return _render_report_box(title, rows)


def print_experiment_report(**kwargs: Any) -> None:
	print(render_experiment_report(**kwargs))


def print_epoch_table_header() -> None:
	print(_make_table_border("-"))
	print("|" + "|".join(f"{name:^{width}}" for name, width in EPOCH_COLUMNS) + "|")
	print(_make_table_border("="))


def print_epoch_table_row(
	epoch: int,
	epochs: int,
	train_loss: float,
	train_metrics: Mapping[str, Any],
	val_loss: float,
	val_metrics: Mapping[str, Any],
) -> None:
	row_values = [
		f"{epoch:03d}/{epochs:03d}",
		f"{train_loss:.4f}",
		f"{train_metrics['accuracy']:.4f}",
		f"{train_metrics['precision']:.4f}",
		f"{train_metrics['recall']:.4f}",
		f"{train_metrics['f1']:.4f}",
		f"{val_loss:.4f}",
		f"{val_metrics['accuracy']:.4f}",
		f"{val_metrics['precision']:.4f}",
		f"{val_metrics['recall']:.4f}",
		f"{val_metrics['f1']:.4f}",
	]
	print("|" + "|".join(_fit_table_value(value, width) for value, (_, width) in zip(row_values, EPOCH_COLUMNS)) + "|")


def print_epoch_table_footer() -> None:
	print(_make_table_border("-"))


def render_epoch_table(
	history: Mapping[str, Sequence[Any]],
	total_epochs: int | None = None,
) -> str:
	lines = [
		_make_table_border("-"),
		"|" + "|".join(f"{name:^{width}}" for name, width in EPOCH_COLUMNS) + "|",
		_make_table_border("="),
	]

	for index in range(_history_length(history)):
		row_values = _history_row_values(history, index, total_epochs=total_epochs)
		lines.append(
			"|"
			+ "|".join(
				_fit_table_value(value, width)
				for value, (_, width) in zip(row_values, EPOCH_COLUMNS)
			)
			+ "|"
		)

	lines.append(_make_table_border("-"))
	return "\n".join(lines)


def render_metrics_report(
	title: str,
	metrics: Mapping[str, Any],
	matrix: Any | None = None,
) -> str:
	rows = [
		(name.upper(), _stringify_report_value(value))
		for name, value in metrics.items()
	]

	extra_lines = []
	if matrix is not None:
		extra_lines.append("CONFUSION MATRIX:")
		extra_lines.extend(
			np.array2string(
				np.asarray(matrix),
				max_line_width=REPORT_WIDTH - 6,
			).splitlines()
		)

	return _render_report_box(title, rows, extra_lines=extra_lines)


def print_metrics_report(title: str, metrics: Mapping[str, Any], matrix: Any | None = None) -> None:
	print(render_metrics_report(title=title, metrics=metrics, matrix=matrix))


def render_summary_report(title: str, summary: Mapping[str, Any]) -> str:
	rows = [
		("ACCURACY", f"{summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}"),
		("PRECISION", f"{summary['precision_mean']:.4f} +/- {summary['precision_std']:.4f}"),
		("RECALL", f"{summary['recall_mean']:.4f} +/- {summary['recall_std']:.4f}"),
		("F1", f"{summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}"),
	]
	return _render_report_box(title, rows)


def print_summary_report(title: str, summary: Mapping[str, Any]) -> None:
	print(render_summary_report(title=title, summary=summary))


def sanitize_path_segment(value: str) -> str:
	value = str(value).strip()
	forbidden = r'<>"/\\|?*'
	if os.name == "nt":
		forbidden += ":"
	value = re.sub(f"[{re.escape(forbidden)}]+", "_", value)
	value = re.sub(r"\s+", "_", value)
	value = value.strip(" ._")
	return value or "unnamed"


def format_run_timestamp(value: datetime) -> str:
	label = value.strftime("%d-%m-%Y-%H:%M")
	if os.name == "nt":
		# Windows cannot create path segments containing ":".
		label = label.replace(":", "-")
	return label


class ExperimentResultWriter:
	def __init__(
		self,
		pipeline_name: str,
		base_dir: str | Path | None = None,
		timestamp: datetime | None = None,
	) -> None:
		self.pipeline_name = pipeline_name
		self.pipeline_slug = sanitize_path_segment(pipeline_name)
		self.started_at = timestamp or datetime.now()
		self.base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent / ".results"
		self.run_dir = self._make_run_dir()
		self.subsets: list[dict[str, Any]] = []
		self._write_manifest()

	def _make_run_dir(self) -> Path:
		pipeline_dir = self.base_dir / self.pipeline_slug
		timestamp_label = format_run_timestamp(self.started_at)
		candidate = pipeline_dir / timestamp_label

		if not candidate.exists():
			candidate.mkdir(parents=True, exist_ok=False)
			return candidate

		if not any(candidate.iterdir()):
			return candidate

		for index in range(2, 1000):
			unique_candidate = pipeline_dir / f"{timestamp_label}_{index:02d}"
			if not unique_candidate.exists():
				unique_candidate.mkdir(parents=True, exist_ok=False)
				return unique_candidate

		raise RuntimeError(f"Could not create a unique results directory for {self.pipeline_name!r}.")

	def _write_manifest(self, completed_at: datetime | None = None) -> None:
		manifest = {
			"pipeline_name": self.pipeline_name,
			"pipeline_slug": self.pipeline_slug,
			"run_dir": str(self.run_dir),
			"started_at": self.started_at.isoformat(timespec="seconds"),
			"completed_at": completed_at.isoformat(timespec="seconds") if completed_at else None,
			"subsets": self.subsets,
		}
		self._write_json(self.run_dir / "manifest.json", manifest)

	def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
		path.write_text(
			json.dumps(_to_serializable(payload), indent=2, ensure_ascii=False),
			encoding="utf-8",
		)

	def _write_rows(self, path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
		with path.open("w", newline="", encoding="utf-8") as csv_file:
			writer = csv.writer(csv_file, delimiter=";")
			writer.writerow(header)
			writer.writerows(rows)

	def save_subset(
		self,
		subset_index: int,
		subset_count: int,
		report_context: Mapping[str, Any],
		history: Mapping[str, Sequence[Any]],
		test_metrics: Mapping[str, Any],
		confusion_matrix: Any,
	) -> dict[str, str]:
		prefix = f"subset_{subset_index:02d}"
		total_epochs = _to_builtin_scalar(report_context.get("training_setup", {}).get("epochs"))

		protocol = render_experiment_report(
			**dict(report_context),
			now=self.started_at,
		)
		report_text = "\n\n".join(
			[
				protocol,
				render_epoch_table(history, total_epochs=total_epochs),
				render_metrics_report("TEST RESULTS", test_metrics, matrix=confusion_matrix),
			]
		)

		report_path = self.run_dir / f"{prefix}_report.log"
		epochs_path = self.run_dir / f"{prefix}_epochs.csv"
		test_metrics_path = self.run_dir / f"{prefix}_test_metrics.csv"
		confusion_matrix_path = self.run_dir / f"{prefix}_confusion_matrix.csv"
		confusion_matrix_txt_path = self.run_dir / f"{prefix}_confusion_matrix.txt"

		report_path.write_text(report_text + "\n", encoding="utf-8")
		self._write_history_csv(epochs_path, history)
		self._write_metric_csv(test_metrics_path, test_metrics)
		self._write_confusion_matrix_csv(confusion_matrix_path, confusion_matrix)
		confusion_matrix_txt_path.write_text(
			np.array2string(np.asarray(confusion_matrix)) + "\n",
			encoding="utf-8",
		)

		created_files = {
			"report": str(report_path),
			"epochs": str(epochs_path),
			"test_metrics": str(test_metrics_path),
			"confusion_matrix_csv": str(confusion_matrix_path),
			"confusion_matrix_txt": str(confusion_matrix_txt_path),
		}

		self.subsets.append(
			{
				"subset_index": subset_index,
				"subset_count": subset_count,
				"report_context": dict(report_context),
				"files": created_files,
			}
		)
		self._write_manifest()
		return created_files

	def _write_history_csv(self, path: Path, history: Mapping[str, Sequence[Any]]) -> None:
		rows = []
		for index in range(_history_length(history)):
			row = [index + 1]
			for _, key in HISTORY_COLUMNS[1:]:
				row.append(_csv_value(_history_value(history, key, index)))
			rows.append(row)

		self._write_rows(path, [name for name, _ in HISTORY_COLUMNS], rows)

	def _write_metric_csv(self, path: Path, metrics: Mapping[str, Any]) -> None:
		rows = [(name, _csv_value(value)) for name, value in metrics.items()]
		self._write_rows(path, ["metric", "value"], rows)

	def _write_confusion_matrix_csv(self, path: Path, matrix: Any) -> None:
		array = np.asarray(matrix)
		rows = [[_csv_value(value) for value in row] for row in array.tolist()]
		self._write_rows(path, [f"pred_{index}" for index in range(array.shape[1])], rows)

	def save_final_summary(
		self,
		summary: Mapping[str, Any],
		subset_results: Sequence[Mapping[str, Any]] | None = None,
	) -> dict[str, str]:
		summary_log_path = self.run_dir / "final_summary_across.log"
		summary_csv_path = self.run_dir / "final_summary_across.csv"
		subset_metrics_path = self.run_dir / "subset_test_metrics.csv"
		summary_json_path = self.run_dir / "final_summary_across.json"

		summary_log_path.write_text(
			render_summary_report("FINAL SUMMARY ACROSS SUBSETS", summary) + "\n",
			encoding="utf-8",
		)

		summary_rows = [
			(metric, _csv_value(summary[mean_key]), _csv_value(summary[std_key]))
			for metric, mean_key, std_key in SUMMARY_COLUMNS
		]
		self._write_rows(summary_csv_path, ["metric", "mean", "std"], summary_rows)

		if subset_results is not None:
			self._write_subset_metrics_csv(subset_metrics_path, subset_results)

		self._write_json(summary_json_path, {"summary": dict(summary)})
		self._write_manifest(completed_at=datetime.now())

		return {
			"summary_log": str(summary_log_path),
			"summary_csv": str(summary_csv_path),
			"subset_metrics": str(subset_metrics_path),
			"summary_json": str(summary_json_path),
		}

	def _write_subset_metrics_csv(
		self,
		path: Path,
		subset_results: Sequence[Mapping[str, Any]],
	) -> None:
		header = ["subset", "loss", "accuracy", "precision", "recall", "f1"]
		rows = []
		for index, result in enumerate(subset_results, start=1):
			metrics = result.get("test_metrics", {})
			rows.append(
				[
					index,
					_csv_value(metrics.get("loss", "")),
					_csv_value(metrics.get("accuracy", "")),
					_csv_value(metrics.get("precision", "")),
					_csv_value(metrics.get("recall", "")),
					_csv_value(metrics.get("f1", "")),
				]
			)
		self._write_rows(path, header, rows)
