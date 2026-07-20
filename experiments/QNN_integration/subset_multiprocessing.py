from __future__ import annotations

from concurrent import futures
from contextlib import ExitStack
import inspect
from multiprocessing import Manager, cpu_count
from pathlib import Path
import pickle
from queue import Empty
from typing import Any, Callable, Iterator, Mapping

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
	subset_kwargs_factory_payload: bytes | None,
	subset_idx: int,
	subset_count: int,
	data_block: Any,
	train_kwargs: Mapping[str, Any],
	report_kwargs: Mapping[str, Any],
	progress_queue: Any | None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
	trainer = cloudpickle.loads(trainer_payload)
	subset_kwargs_factory = (
		cloudpickle.loads(subset_kwargs_factory_payload)
		if subset_kwargs_factory_payload is not None
		else None
	)
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
	trainer_kwargs = dict(train_kwargs)
	if subset_kwargs_factory is not None:
		trainer_kwargs.update(dict(subset_kwargs_factory(subset_idx, data_block)))
	trainer_kwargs["report_context"] = report_context

	trainer_parameters = inspect.signature(trainer).parameters
	accepts_var_kwargs = any(
		parameter.kind == inspect.Parameter.VAR_KEYWORD
		for parameter in trainer_parameters.values()
	)
	if accepts_var_kwargs or "data_block" in trainer_parameters:
		trainer_kwargs["data_block"] = data_block
	if accepts_var_kwargs or "subset_idx" in trainer_parameters:
		trainer_kwargs["subset_idx"] = subset_idx
	if accepts_var_kwargs or "subset_count" in trainer_parameters:
		trainer_kwargs["subset_count"] = subset_count
	if progress_queue is not None:
		progress_queue.put((subset_idx, {
			"phase": "initializing",
			"status": "starting",
			"completed": 0,
		}))
		if accepts_var_kwargs or "progress_callback" in trainer_parameters:
			def report_progress(event: Mapping[str, Any]) -> None:
				progress_queue.put((subset_idx, dict(event)))

			trainer_kwargs["progress_callback"] = report_progress

	try:
		result = trainer(**trainer_kwargs)
		prepared_result = _prepare_result_for_ipc(result)
	except BaseException as error:
		if progress_queue is not None:
			progress_queue.put((subset_idx, {
				"phase": "failed",
				"status": "failed",
				"error": f"{type(error).__name__}: {error}",
			}))
		raise

	if progress_queue is not None:
		progress_queue.put((subset_idx, {
			"phase": "complete",
			"status": "complete",
		}))
	return subset_idx, report_context, prepared_result


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


def _progress_details(event: Mapping[str, Any]) -> str:
	phase = str(event.get("phase", "working"))
	details = [phase]
	epoch = event.get("epoch")
	epochs = event.get("epochs")
	if epoch is not None and epochs is not None and phase != "test":
		details.append(f"epoch {epoch}/{epochs}")
	batch = event.get("batch")
	phase_total = event.get("phase_total")
	if batch is not None and phase_total is not None:
		details.append(f"batch {batch}/{phase_total}")
	if event.get("early_stopping") or event.get("status") == "early_stopping":
		details.append("early stop")
	return " | ".join(details)


def _apply_progress_event(progress_bar: tqdm, event: Mapping[str, Any]) -> None:
	total = event.get("total")
	total_changed = total is not None and progress_bar.total != int(total)
	if total is not None:
		progress_bar.total = int(total)
	completed = event.get("completed")
	advance = 0
	if completed is not None:
		advance = int(completed) - int(progress_bar.n)

	status = event.get("status")
	if status == "complete":
		if progress_bar.total is None:
			progress_bar.total = max(1, int(progress_bar.n))
		advance = int(progress_bar.total) - int(progress_bar.n)
	elif status == "failed":
		progress_bar.set_description_str(f"{progress_bar.desc.split(' [', 1)[0]} [FAILED]")

	progress_bar.set_postfix_str(_progress_details(event), refresh=False)
	if advance > 0:
		progress_bar.update(advance)
	elif advance < 0:
		progress_bar.n += advance
		progress_bar.refresh()
	elif total_changed or status in {"starting", "early_stopping", "complete", "failed"}:
		progress_bar.refresh()


def _drain_progress_events(
	progress_queue: Any,
	progress_bars: list[tqdm],
	*,
	timeout: float,
) -> None:
	try:
		subset_idx, event = progress_queue.get(timeout=timeout)
	except Empty:
		return
	_apply_progress_event(progress_bars[subset_idx], event)

	while True:
		try:
			subset_idx, event = progress_queue.get_nowait()
		except Empty:
			return
		_apply_progress_event(progress_bars[subset_idx], event)


def _completed_futures_with_progress(
	future_to_subset: Mapping[futures.Future[Any], int],
	progress_queue: Any,
	progress_bars: list[tqdm],
) -> Iterator[futures.Future[Any]]:
	pending = set(future_to_subset)
	while pending:
		_drain_progress_events(progress_queue, progress_bars, timeout=0.1)
		done, pending = futures.wait(
			pending,
			timeout=0,
			return_when=futures.FIRST_COMPLETED,
		)
		yield from done
	_drain_progress_events(progress_queue, progress_bars, timeout=0)


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
	dataset_id: str,
	experiment_group: str,
	model_family: str,
	encoding_id: str,
	model_id: str,
	pipeline_name: str,
	classifier_name: str,
	model_architecture: str,
	save_results: bool = True,
	results_base_dir: str | Path | None = None,
	training_setup_extra: Mapping[str, Any] | None = None,
	subset_trainer_kwargs_factory: Callable[[int, Any], Mapping[str, Any]] | None = None,
	max_workers: int | None = None,
	show_progress_bars: bool | None = None,
) -> dict[str, Any]:
	"""Train subsets in separate processes with either detailed logs or fixed progress bars.

	When ``show_progress_bars`` is omitted, bars are enabled exactly when
	``verbose`` is false. Set both flags to false for completely quiet execution.
	"""
	if show_progress_bars is None:
		show_progress_bars = not verbose
	if show_progress_bars and verbose:
		raise ValueError(
			"verbose output and progress bars cannot be enabled at the same time; "
			"set either verbose=False or show_progress_bars=False."
		)

	subset_count = len(dataset.subsets)
	all_results_by_subset: list[dict[str, Any] | None] = [None] * subset_count
	results_writer = (
		ExperimentResultWriter(
			pipeline_name=pipeline_name,
			dataset_id=dataset_id,
			experiment_group=experiment_group,
			model_family=model_family,
			encoding_id=encoding_id,
			model_id=model_id,
			base_dir=results_base_dir,
		)
		if save_results
		else None
	)
	worker_count = max_workers or min(subset_count, max(1, cpu_count() - 1))
	trainer_payload = cloudpickle.dumps(trainer)
	subset_kwargs_factory_payload = (
		cloudpickle.dumps(subset_trainer_kwargs_factory)
		if subset_trainer_kwargs_factory is not None
		else None
	)
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

	with ExitStack() as stack:
		manager = stack.enter_context(Manager()) if show_progress_bars else None
		progress_queue = manager.Queue() if manager is not None else None
		executor = stack.enter_context(futures.ProcessPoolExecutor(max_workers=worker_count))
		future_to_subset = {
			executor.submit(
				_run_subset_task,
				trainer_payload,
				subset_kwargs_factory_payload,
				subset_idx,
				subset_count,
				data_block,
				train_kwargs,
				report_kwargs,
				progress_queue,
			): subset_idx
			for subset_idx, data_block in enumerate(dataset.subsets)
		}

		progress_bars = (
			[
				tqdm(
					total=None,
					desc=f"Subset {subset_idx + 1}/{subset_count}",
					position=subset_idx,
					leave=True,
					dynamic_ncols=True,
					unit="batch",
				)
				for subset_idx in range(subset_count)
			]
			if show_progress_bars
			else []
		)
		for progress_bar in progress_bars:
			progress_bar.set_postfix_str("waiting", refresh=False)
			progress_bar.refresh()

		completed_futures: Iterator[futures.Future[Any]] = (
			_completed_futures_with_progress(
				future_to_subset,
				progress_queue,
				progress_bars,
			)
			if show_progress_bars
			else futures.as_completed(future_to_subset)
		)
		try:
			for future in completed_futures:
				subset_idx, report_context, result = future.result()
				if progress_bars:
					_apply_progress_event(progress_bars[subset_idx], {
						"phase": "complete",
						"status": "complete",
					})
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
		finally:
			if progress_queue is not None:
				_drain_progress_events(progress_queue, progress_bars, timeout=0)
			for progress_bar in progress_bars:
				progress_bar.close()

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
