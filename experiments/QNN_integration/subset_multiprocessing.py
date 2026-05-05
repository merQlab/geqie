from __future__ import annotations

from concurrent import futures
from multiprocessing import cpu_count
import pickle
from typing import Any, Callable, Mapping

import cloudpickle
import numpy as np
from tqdm import tqdm

from experiments.QNN_integration.experiment_results import (
	ExperimentResultWriter,
	make_model_checkpoint_artifacts,
	print_summary_report,
)


def _make_report_context(
	*,
	subset_idx: int,
	subset_count: int,
	data_block: Any,
	epochs: int,
	batch_size: int,
	device: str,
	num_classes: int,
	num_qubits: int,
	num_layers: int,
	classifier_name: str,
	model_architecture: str,
	training_setup_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	training_setup = {
		"epochs": epochs,
		"batch_size": batch_size,
		"device": device,
		"qnn_lr": 0.001,
		"head_lr": 0.01,
	}
	if training_setup_extra:
		training_setup.update(dict(training_setup_extra))

	return {
		"title": "EXPERIMENTAL PROTOCOL REPORT",
		"dataset_name": "MNIST digits stratified",
		"classifier_name": classifier_name,
		"model_architecture": model_architecture if subset_idx == 0 else None,
		"subset_name": f"{subset_idx + 1}/{subset_count}",
		"split_sizes": {
			"train": len(data_block.train.X),
			"val": len(data_block.val.X),
			"test": len(data_block.test.X),
		},
		"training_setup": training_setup,
		"model_setup": {
			"num_classes": num_classes,
			"num_qubits": num_qubits,
			"num_layers": num_layers,
		},
	}


def _run_subset_task(
	trainer_payload: bytes,
	subset_idx: int,
	subset_count: int,
	data_block: Any,
	train_kwargs: Mapping[str, Any],
	report_kwargs: Mapping[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any]]:
	trainer = cloudpickle.loads(trainer_payload)
	report_train_kwargs = {
		key: train_kwargs[key]
		for key in (
			"num_classes",
			"num_qubits",
			"num_layers",
			"epochs",
			"batch_size",
			"device",
		)
	}
	report_context = _make_report_context(
		subset_idx=subset_idx,
		subset_count=subset_count,
		data_block=data_block,
		**report_train_kwargs,
		**dict(report_kwargs),
	)
	result = trainer(
		data_block=data_block,
		subset_idx=subset_idx,
		subset_count=subset_count,
		**dict(train_kwargs),
		report_context=report_context,
	)
	return subset_idx, report_context, _prepare_result_for_ipc(result)


def _prepare_result_for_ipc(result: dict[str, Any]) -> dict[str, Any]:
	try:
		pickle.dumps(result)
		return result
	except Exception as error:
		model = result.get("model")
		if model is None:
			raise RuntimeError("Subset result is not picklable and does not contain a model to convert.") from error

		report_context = result.get("report_context")
		torchinfo_summary = None
		if isinstance(report_context, Mapping):
			torchinfo_summary = report_context.get("torchinfo_summary")

		safe_result = dict(result)
		safe_result["model_artifacts"] = make_model_checkpoint_artifacts(
			model,
			torchinfo_summary=torchinfo_summary,
		)
		safe_result["model"] = None

		try:
			pickle.dumps(safe_result)
		except Exception as safe_error:
			raise RuntimeError("Subset result is still not picklable after converting the model to checkpoint artifacts.") from safe_error

		return safe_result


def _summarize_subset_results(subset_results: list[dict[str, Any]]) -> dict[str, float]:
	accuracies = [result["test_metrics"]["accuracy"] for result in subset_results]
	precisions = [result["test_metrics"]["precision"] for result in subset_results]
	recalls = [result["test_metrics"]["recall"] for result in subset_results]
	f1s = [result["test_metrics"]["f1"] for result in subset_results]

	return {
		"accuracy_mean": float(np.mean(accuracies)),
		"accuracy_std": float(np.std(accuracies)),
		"precision_mean": float(np.mean(precisions)),
		"precision_std": float(np.std(precisions)),
		"recall_mean": float(np.mean(recalls)),
		"recall_std": float(np.std(recalls)),
		"f1_mean": float(np.mean(f1s)),
		"f1_std": float(np.std(f1s)),
	}


def train_subsets_with_process_pool(
	*,
	dataset: Any,
	trainer: Callable[..., dict[str, Any]],
	num_classes: int,
	num_qubits: int,
	num_layers: int,
	epochs: int,
	batch_size: int,
	device: str,
	verbose: bool,
	pipeline_name: str,
	classifier_name: str,
	model_architecture: str,
	save_results: bool = True,
	training_setup_extra: Mapping[str, Any] | None = None,
	max_workers: int | None = None,
) -> dict[str, Any]:
	subset_count = len(dataset.subsets)
	all_results_by_subset: list[dict[str, Any] | None] = [None] * subset_count
	results_writer = ExperimentResultWriter(pipeline_name=pipeline_name) if save_results else None
	worker_count = max_workers or min(subset_count, max(1, cpu_count() - 1))
	trainer_payload = cloudpickle.dumps(trainer)
	train_kwargs = {
		"num_classes": num_classes,
		"num_qubits": num_qubits,
		"num_layers": num_layers,
		"epochs": epochs,
		"batch_size": batch_size,
		"device": device,
		"verbose": verbose,
	}
	report_kwargs = {
		"classifier_name": classifier_name,
		"model_architecture": model_architecture,
		"training_setup_extra": training_setup_extra,
	}

	with futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
		future_to_subset = {
			executor.submit(
				_run_subset_task,
				trainer_payload,
				subset_idx,
				subset_count,
				data_block,
				train_kwargs,
				report_kwargs,
			): subset_idx
			for subset_idx, data_block in enumerate(dataset.subsets)
		}

		for future in tqdm(
			futures.as_completed(future_to_subset),
			total=len(future_to_subset),
			desc="Processing results",
		):
			subset_idx, report_context, result = future.result()
			if results_writer is not None:
				results_writer.save_subset(
					subset_index=subset_idx + 1,
					subset_count=subset_count,
					report_context=result.get("report_context", report_context),
					history=result["history"],
					test_metrics=result["test_metrics"],
					confusion_matrix=result["confusion_matrix"],
					model=result.get("model"),
					model_artifacts=result.get("model_artifacts"),
				)
			result = dict(result)
			result.pop("model_artifacts", None)
			all_results_by_subset[subset_idx] = result

	subset_results = [
		result for result in all_results_by_subset
		if result is not None
	]
	summary = _summarize_subset_results(subset_results)

	if verbose:
		print_summary_report("FINAL SUMMARY ACROSS SUBSETS", summary)

	if results_writer is not None:
		results_writer.save_final_summary(summary=summary, subset_results=subset_results)

	return {
		"subset_results": subset_results,
		"summary": summary,
		"results_dir": str(results_writer.run_dir) if results_writer is not None else None,
	}
