"""Shared infrastructure for the ``.fast_layer`` Lightning training scripts.

Every model script defines a ``ModelSpec`` (name, learning rate, accelerator and a ``build`` function) and delegates
the rest to this module: loading one of the 5 dataset subsets, training with Lightning, testing, and writing all
artifacts to ``<log_root>/<model_name>/subset_<n>/``.

The dataset is read from ``DEFAULT_DATASET_FILE`` and GEQIE-precomputed archives from ``ENCODING_ARCHIVE_TEMPLATES[encoding]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import joblib
import lightning as L
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from lightning.pytorch.loggers import CSVLogger
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from torch.utils import data as torch_data
from torchmetrics import Accuracy

FAST_LAYER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = FAST_LAYER_DIR.parents[1]

DATASET_NAME = "MNIST_Digits"
DATASET_FILENAME = "MNIST_Digits_5_subsets_train_val_test_16x16.joblib"

DATA_ROOT = Path("/mnt/data02/mkordasz")
DEFAULT_DATASET_FILE = DATA_ROOT / "datasets_joblib" / DATASET_FILENAME
ENCODING_ARCHIVE_TEMPLATES = {
    "frqi": DATA_ROOT / "circuits" / "FRQI" / DATASET_NAME / "subset_{index}.zip",
    "neqr": DATA_ROOT / "circuits" / "NEQR" / DATASET_NAME / "NEQR_subset_{index}.zip",
}
DEFAULT_LOG_ROOT = FAST_LAYER_DIR / "lightning_logs"

SUBSET_NUMBERS = (1, 2, 3, 4, 5)
IMAGE_SIZE = 16
N_CLASSES = 10
SEED = 42
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#ff7f0e"

LOGGER = logging.getLogger("fast_layer")
LOGGER.setLevel(logging.INFO)
LOG_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

for _path in (REPO_ROOT, REPO_ROOT / "geqie" / "src", REPO_ROOT / "geqie-qml" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.QNN_integration.datasets.dataset_structure import DataBlock, DataSet, DatasetSplit  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    log_root: Path
    dataset_file: Path
    precompute_workers: int


@dataclass
class SubsetContext:
    """Everything a model ``build`` function may need for one subset."""

    index: int
    data_block: DataBlock
    precompute_workers: int


@dataclass
class TrainingSetup:
    model: nn.Module
    train_loader: torch_data.DataLoader
    val_loader: torch_data.DataLoader
    test_loader: torch_data.DataLoader
    summary_input_size: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    learning_rate: float
    build: Callable[[SubsetContext], TrainingSetup]
    accelerator: str = "cpu"


class ClassifierModule(L.LightningModule):
    """Lightning wrapper; the model ends with LogSoftmax, so the loss is NLL."""

    def __init__(self, model: nn.Module, n_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        log_probs = self(x)
        loss = F.nll_loss(log_probs, y)
        preds = log_probs.argmax(dim=-1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class TrainingTimer(L.Callback):
    """Measure wall-clock time per training batch, per epoch and for the whole fit; log one line per epoch.

    Batch time covers forward, backward and the optimizer step (data loading excluded). Epoch time is the full epoch
    including validation, which is also reported separately. Per-epoch times are logged to ``metrics.csv`` as well.
    """

    def __init__(self) -> None:
        self.batch_rows: list[dict[str, Any]] = []
        self.epoch_rows: list[dict[str, Any]] = []
        self.fit_seconds = float("nan")
        self._fit_start = self._epoch_start = self._batch_start = self._val_start = 0.0
        self._val_seconds = 0.0
        self._epoch_batch_seconds: list[float] = []

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._fit_start = _now(pl_module)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.fit_seconds = _now(pl_module) - self._fit_start

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_batch_seconds = []
        self._val_seconds = 0.0
        self._epoch_start = _now(pl_module)

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int) -> None:
        self._batch_start = _now(pl_module)

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        seconds = _now(pl_module) - self._batch_start
        self._epoch_batch_seconds.append(seconds)
        self.batch_rows.append(
            {"epoch": trainer.current_epoch, "batch": batch_idx, "batch_size": len(batch[1]), "seconds": seconds}
        )

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._val_start = _now(pl_module)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._val_seconds = _now(pl_module) - self._val_start

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        batches = np.asarray(self._epoch_batch_seconds)
        row = {
            "epoch": trainer.current_epoch,
            "epoch_seconds": _now(pl_module) - self._epoch_start,
            "train_seconds": float(batches.sum()),
            "val_seconds": self._val_seconds,
            "batches": len(batches),
            "batch_seconds_mean": float(batches.mean()) if len(batches) else float("nan"),
            "batch_seconds_max": float(batches.max()) if len(batches) else float("nan"),
        }
        self.epoch_rows.append(row)
        pl_module.log_dict(
            {"epoch_time": row["epoch_seconds"], "batch_time_mean": row["batch_seconds_mean"]},
            on_step=False,
            on_epoch=True,
        )

        metrics = " ".join(
            f"{name}={float(value):.4f}"
            for name, value in sorted(trainer.callback_metrics.items())
            if name not in ("epoch_time", "batch_time_mean")
        )
        LOGGER.info(
            "epoch %d | %s | epoch %.2fs (train %.2fs, val %.2fs) | batch mean %.3fs max %.3fs (%d batches)",
            row["epoch"],
            metrics,
            row["epoch_seconds"],
            row["train_seconds"],
            row["val_seconds"],
            row["batch_seconds_mean"],
            row["batch_seconds_max"],
            row["batches"],
        )

    def summary(self) -> dict[str, float]:
        epochs = pd.DataFrame(self.epoch_rows)
        batches = pd.DataFrame(self.batch_rows)
        return {
            "fit_seconds": round(self.fit_seconds, 3),
            "epoch_seconds_mean": round(float(epochs["epoch_seconds"].mean()), 3) if len(epochs) else float("nan"),
            "batch_seconds_mean": round(float(batches["seconds"].mean()), 4) if len(batches) else float("nan"),
        }

    def save(self, run_dir: Path) -> None:
        pd.DataFrame(self.epoch_rows).to_csv(run_dir / "epoch_times.csv", index=False)
        pd.DataFrame(self.batch_rows).to_csv(run_dir / "batch_times.csv", index=False)


def _now(pl_module: L.LightningModule) -> float:
    """Wall-clock time; waits for queued GPU work so the measured interval covers it."""
    if pl_module.device.type == "cuda":
        torch.cuda.synchronize(pl_module.device)
    return time.perf_counter()


# --- data ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_dataset(dataset_file: Path) -> DataSet:
    # The joblib artifact was pickled from a notebook where these dataclasses lived in ``__main__``.
    import __main__

    for class_ in (DatasetSplit, DataBlock, DataSet):
        if not hasattr(__main__, class_.__name__):
            setattr(__main__, class_.__name__, class_)
    return joblib.load(dataset_file)


def load_data_block(dataset_file: Path, subset_index: int) -> DataBlock:
    dataset = load_dataset(dataset_file)
    if subset_index not in range(1, len(dataset.subsets) + 1):
        raise ValueError(f"Subset {subset_index} does not exist; the dataset has {len(dataset.subsets)} subsets.")
    return dataset.subsets[subset_index - 1]


def _make_loader(
    features: torch.Tensor, labels: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool = False
) -> torch_data.DataLoader:
    dataset = torch_data.TensorDataset(features, torch.as_tensor(np.asarray(labels), dtype=torch.long))
    return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _to_image_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 ``(N, H, W)`` -> float ``(N, 1, H, W)`` scaled to [0, 1]."""
    return torch.as_tensor(np.asarray(images), dtype=torch.float32).unsqueeze(1) / 255.0


def image_loaders(
    block: DataBlock, batch_size: int, drop_last_train: bool = False
) -> tuple[torch_data.DataLoader, torch_data.DataLoader, torch_data.DataLoader]:
    return (
        _make_loader(_to_image_tensor(block.train.X), block.train.y, batch_size, True, drop_last_train),
        _make_loader(_to_image_tensor(block.val.X), block.val.y, batch_size, False),
        _make_loader(_to_image_tensor(block.test.X), block.test.y, batch_size, False),
    )


def pca_loaders(
    block: DataBlock, batch_size: int, n_components: int
) -> tuple[torch_data.DataLoader, torch_data.DataLoader, torch_data.DataLoader]:
    """PCA is fit only on the training split, then used to transform train/val/test alike."""

    def flatten(images: np.ndarray) -> np.ndarray:
        return _to_image_tensor(images).flatten(start_dim=1).numpy()

    pca = PCA(n_components=n_components)
    pca.fit(flatten(block.train.X))

    def make(split: DatasetSplit, shuffle: bool) -> torch_data.DataLoader:
        features = torch.tensor(pca.transform(flatten(split.X)), dtype=torch.float32)
        return _make_loader(features, split.y, batch_size, shuffle)

    return make(block.train, True), make(block.val, False), make(block.test, False)


def precomputed_zip_loaders(
    context: SubsetContext, encoding: str, encoding_params: dict[str, Any], batch_size: int
) -> tuple[torch_data.DataLoader, torch_data.DataLoader, torch_data.DataLoader, int]:
    """Load the subset's GEQIE-encoded train/val/test matrices lazily from ``ENCODING_ARCHIVE_TEMPLATES[encoding]``.

    A missing archive is precomputed (once) with the same layout. Returns the three loaders and the number of qubits
    inferred from an encoded ``2**n x 2**n`` matrix.
    """
    import geqie_qml

    zip_path = Path(str(ENCODING_ARCHIVE_TEMPLATES[encoding]).format(index=context.index))
    if zip_path.is_file():
        LOGGER.info("Using precomputed archive: %s", zip_path)
    else:
        LOGGER.warning("Precomputed archive not found, building it: %s", zip_path)
        _build_precomputed_zip(context, encoding, encoding_params, zip_path)

    train_ds, val_ds, test_ds = geqie_qml.load_precomputed_zip_matrices(
        str(zip_path), split_names=["train", "val", "test"]
    )
    sample_matrix, _ = train_ds[0]
    n_qubits = int(round(np.log2(sample_matrix.shape[-1])))
    LOGGER.info("Encoded qubits: %d", n_qubits)

    def make(dataset: torch_data.Dataset, shuffle: bool) -> torch_data.DataLoader:
        return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return make(train_ds, True), make(val_ds, False), make(test_ds, False), n_qubits


def _build_precomputed_zip(
    context: SubsetContext, encoding: str, encoding_params: dict[str, Any], zip_path: Path
) -> None:
    import geqie_qml

    stage_dir = zip_path.with_suffix("")
    for split_name in ("train", "val", "test"):
        split = getattr(context.data_block, split_name)
        geqie_qml.compute_and_save_circuits(
            data=split.X,
            labels=split.y,
            save_dir=str(stage_dir / split_name),
            geqie_encoding=encoding,
            encoding_params=encoding_params,
            number_of_workers=context.precompute_workers,
        )

    # Build beside the final name so an interrupted run never leaves a truncated archive that would be reused.
    partial_path = zip_path.with_name(zip_path.name + ".partial")
    with zipfile.ZipFile(partial_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for matrix_file in sorted(stage_dir.glob("*/*.npz")):
            archive.write(matrix_file, arcname=matrix_file.relative_to(stage_dir).as_posix())
    partial_path.replace(zip_path)
    LOGGER.info("Wrote archive: %s", zip_path)


# --- training -----------------------------------------------------------------------------------------------------


def train_subset(spec: ModelSpec, subset_index: int, config: RunConfig) -> dict[str, Any]:
    """Train and test one model on one subset; never raises, failures are logged and reported in the result."""
    csv_logger = CSVLogger(save_dir=str(config.log_root), name=spec.name, version=f"subset_{subset_index}")
    csv_logger.experiment  # creates the run directory before train.log is written into it
    run_dir = Path(csv_logger.log_dir)
    result: dict[str, Any] = {"model": spec.name, "subset": subset_index, "status": "failed", "log_dir": str(run_dir)}

    with _run_logging(run_dir / "train.log"):
        started = time.perf_counter()
        try:
            LOGGER.info("=== %s / subset_%d ===", spec.name, subset_index)
            LOGGER.info("Dataset: %s", config.dataset_file)
            L.seed_everything(SEED, workers=True)
            context = SubsetContext(
                subset_index,
                load_data_block(config.dataset_file, subset_index),
                config.precompute_workers,
            )
            setup = spec.build(context)
            LOGGER.info(
                "Train: %d, Val: %d, Test: %d",
                len(setup.train_loader.dataset),
                len(setup.val_loader.dataset),
                len(setup.test_loader.dataset),
            )
            _log_model_summary(setup)

            module = ClassifierModule(setup.model, n_classes=N_CLASSES, lr=spec.learning_rate)
            timer = TrainingTimer()
            trainer = L.Trainer(
                max_epochs=MAX_EPOCHS,
                accelerator=spec.accelerator,
                devices=1,
                logger=csv_logger,
                log_every_n_steps=1,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                callbacks=[
                    L.pytorch.callbacks.EarlyStopping(
                        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min"
                    ),
                    timer,
                ],
            )
            trainer.fit(module, train_dataloaders=setup.train_loader, val_dataloaders=setup.val_loader)
            timer.save(run_dir)
            test_started = time.perf_counter()
            test_metrics = trainer.test(module, dataloaders=setup.test_loader)[0]

            result.update(
                status="ok",
                epochs=trainer.current_epoch,
                **timer.summary(),
                test_seconds=round(time.perf_counter() - test_started, 3),
                total_seconds=round(time.perf_counter() - started, 3),
                **test_metrics,
            )
            (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            _save_training_curves(spec.name, subset_index, run_dir)
            LOGGER.info("Finished: %s", test_metrics)
            LOGGER.info(
                "Timing: total %.1fs (fit %.1fs, test %.1fs) | epoch mean %.2fs | batch mean %.3fs",
                result["total_seconds"],
                result["fit_seconds"],
                result["test_seconds"],
                result["epoch_seconds_mean"],
                result["batch_seconds_mean"],
            )
        except Exception:
            LOGGER.exception("Training failed for %s / subset_%d", spec.name, subset_index)
            result["total_seconds"] = round(time.perf_counter() - started, 3)
    return result


def _log_model_summary(setup: TrainingSetup) -> None:
    try:
        LOGGER.info("%s", torchinfo.summary(setup.model, input_size=setup.summary_input_size, verbose=0))
    except Exception as error:  # the summary is informational only
        LOGGER.warning("Could not build the model summary: %s", error)


def _save_training_curves(model_name: str, subset_index: int, run_dir: Path) -> None:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
    for columns, col_idx in ((["train_loss", "val_loss"], 1), (["train_acc", "val_acc"], 2)):
        for column in columns:
            if column not in metrics:
                continue
            values = metrics.dropna(subset=[column])
            color = TRAIN_COLOR if column.startswith("train") else VAL_COLOR
            fig.add_trace(
                go.Scatter(
                    x=values["epoch"],
                    y=values[column],
                    mode="lines+markers",
                    name=column,
                    line=dict(color=color),
                    marker=dict(color=color),
                ), 
                row=1,
                col=col_idx,
            )
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(height=400, width=900, title_text=f"{model_name} / subset_{subset_index}")
    fig.write_html(run_dir / "training_curves.html", include_plotlyjs="cdn")


@contextmanager
def _run_logging(log_file: Path) -> Iterator[None]:
    """Mirror the project logger, Lightning's messages and Python warnings into ``log_file`` for the duration of a run."""
    handler = _file_handler(log_file)
    for logger in _file_logged_loggers():
        logger.addHandler(handler)
    try:
        yield
    finally:
        for logger in _file_logged_loggers():
            logger.removeHandler(handler)
        handler.close()


def _configure_session_logging(log_root: Path) -> Path:
    """Log to stdout and to ``<log_root>/run_<timestamp>.log`` for the whole script run; returns the log file path."""
    logging.captureWarnings(True)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(LOG_FORMATTER)
    for logger in (LOGGER, logging.getLogger("py.warnings")):
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(console)

    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"run_{datetime.now():%Y-%m-%dT%H-%M-%S}.log"
    session_handler = _file_handler(log_file)
    for logger in _file_logged_loggers():
        logger.addHandler(session_handler)
    return log_file


def _file_logged_loggers() -> tuple[logging.Logger, ...]:
    # Lightning prints its own messages to the console already, so it only needs the file handlers.
    return LOGGER, logging.getLogger("lightning.pytorch"), logging.getLogger("py.warnings")


def _file_handler(log_file: Path) -> logging.FileHandler:
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setFormatter(LOG_FORMATTER)
    return handler


# --- command line -------------------------------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subsets",
        type=int,
        nargs="+",
        default=list(SUBSET_NUMBERS),
        choices=SUBSET_NUMBERS,
        help="Subset numbers to train on (default: all 5).",
    )
    parser.add_argument(
        "--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Logs go to <log-root>/<model>/subset_<n>."
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help=f"Joblib file with the 5 train/val/test subsets (default: {DEFAULT_DATASET_FILE}).",
    )
    parser.add_argument(
        "--precompute-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Worker processes used when an archive is missing and has to be precomputed.",
    )


def run_specs(specs: Sequence[ModelSpec], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Train every spec on every requested subset, one after another, and print a summary table."""
    session_log = _configure_session_logging(args.log_root)
    LOGGER.info("Command: %s", " ".join(sys.argv))
    LOGGER.info("Session log: %s", session_log)
    if not args.dataset_file.is_file():
        LOGGER.error("Dataset file does not exist: %s", args.dataset_file)
        raise FileNotFoundError(args.dataset_file)
    config = RunConfig(args.log_root, args.dataset_file, args.precompute_workers)
    LOGGER.info("Dataset: %s | logs: %s", config.dataset_file, config.log_root)
    results = [train_subset(spec, subset, config) for spec in specs for subset in args.subsets]
    columns = [
        "model",
        "subset",
        "status",
        "epochs",
        "test_acc",
        "test_loss",
        "batch_seconds_mean",
        "epoch_seconds_mean",
        "fit_seconds",
        "total_seconds",
    ]
    LOGGER.info("Summary:\n%s", pd.DataFrame(results).reindex(columns=columns).to_string(index=False))
    return results


def main(spec: ModelSpec) -> None:
    """Entry point shared by the single-model scripts."""
    parser = argparse.ArgumentParser(description=f"Train `{spec.name}` on the dataset subsets.")
    add_common_arguments(parser)
    results = run_specs([spec], parser.parse_args())
    sys.exit(int(any(result["status"] != "ok" for result in results)))
