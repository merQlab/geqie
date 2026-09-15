"""Lightning training with the result and progress contracts of the subset runner.

Imported only by pipelines that select Lightning. Quantum simulation stays on
CPU, with one Trainer per subset process and no nested distributed strategy.
"""

from pathlib import Path
from uuid import uuid4

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix

from . import (
    classification_metrics,
    history_template,
    print_epoch_table_footer,
    print_epoch_table_header,
    print_epoch_table_row,
    print_metrics_report,
    with_torchinfo_summary,
)


class GEQIELightningModule(L.LightningModule):
    """LogSoftmax classifier with the notebook's NLL, Adam and plateau scheduler."""

    def __init__(self, model, num_classes: int, lr: float = 0.1, verbose: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.history = history_template()
        self.history["lr"] = []
        self.test_result = {}
        self._epoch_data = {}

    def forward(self, x):
        return self.model(x)

    def _start_epoch(self, phase):
        self._epoch_data[phase] = {"loss_sum": 0.0, "y_true": [], "y_pred": []}

    def _step(self, batch, phase):
        x, y = batch
        log_probs = self(x)
        loss = F.nll_loss(log_probs, y)
        preds = log_probs.argmax(dim=-1)
        data = self._epoch_data[phase]
        data["loss_sum"] += loss.detach().item() * len(y)
        data["y_true"].extend(y.detach().cpu().tolist())
        data["y_pred"].extend(preds.detach().cpu().tolist())
        self.log(f"{phase}_loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        if phase in ("train", "val"):
            accuracy = self.train_acc if phase == "train" else self.val_acc
            accuracy(preds, y)
            self.log(f"{phase}_acc", accuracy, on_step=False, on_epoch=True)
        return loss

    def _finish_epoch(self, phase):
        data = self._epoch_data.pop(phase)
        metrics = classification_metrics(data["y_true"], data["y_pred"])
        metrics["loss"] = data["loss_sum"] / len(data["y_true"])
        for name, value in metrics.items():
            if name != "loss":
                self.log(f"{phase}_{name}", value, on_step=False, on_epoch=True)
            if phase != "test":
                self.history[f"{phase}_{name}"].append(value)
        return {**metrics, "y_true": data["y_true"], "y_pred": data["y_pred"]}

    def on_train_epoch_start(self):
        self._start_epoch("train")
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.history["lr"].append(lr)
        self.log("lr", lr, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_train_epoch_end(self):
        train = self._finish_epoch("train")
        if self.hparams.verbose:
            if self.current_epoch == 0:
                print_epoch_table_header()
            val = {name: self.history[f"val_{name}"][-1]
                   for name in ("loss", "accuracy", "precision", "recall", "f1")}
            print_epoch_table_row(
                self.current_epoch + 1, self.trainer.max_epochs,
                train["loss"], train, val["loss"], val,
            )

    def on_validation_epoch_start(self):
        self._start_epoch("val")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_validation_epoch_end(self):
        self._finish_epoch("val")

    def on_test_epoch_start(self):
        self._start_epoch("test")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_test_epoch_end(self):
        self.test_result = self._finish_epoch("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class SubsetProgress(L.Callback):
    """Forward Lightning batch events to the parent process's progress bars."""

    def __init__(self, callback, epochs, train_batches, val_batches, test_batches):
        self.callback = callback
        self.epochs = epochs
        self.counts = {"train": train_batches, "validation": val_batches, "test": test_batches}
        self.total = epochs * (train_batches + val_batches) + test_batches
        self.completed = 0
        self.completed_epochs = 0
        self.stopped_early = False

    def emit(self, phase, epoch=None, batch=None, status="running"):
        if self.callback is not None:
            self.callback({
                "phase": phase, "epoch": epoch, "epochs": self.epochs,
                "batch": batch, "phase_total": self.counts.get(phase),
                "completed": self.completed, "total": self.total,
                "status": status, "early_stopping": self.stopped_early,
            })

    def on_fit_start(self, trainer, pl_module):
        self.emit("starting", status="starting")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.completed += 1
        self.emit("train", trainer.current_epoch + 1, batch_idx + 1)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.completed += 1
        self.emit("validation", trainer.current_epoch + 1, batch_idx + 1)

    def on_fit_end(self, trainer, pl_module):
        self.completed_epochs = len(pl_module.history["train_loss"])
        self.stopped_early = trainer.should_stop and self.completed_epochs < self.epochs
        self.total = self.completed + self.counts["test"]
        if self.stopped_early:
            self.emit("early stopping", self.completed_epochs, status="early_stopping")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.completed += 1
        self.emit("test", self.completed_epochs, batch_idx + 1)

    def on_test_end(self, trainer, pl_module):
        self.emit("complete", self.completed_epochs, status="complete")


def train_model_lightning(
    *, model, train_loader, val_loader, test_loader, num_classes, epochs,
    device, verbose, report_context, progress_callback=None,
    lr: float = 0.1, patience: int = 10, log_dir: str | Path | None = None,
):
    """Fit with Lightning, test the best checkpoint and return the common report."""
    if str(device) != "cpu":
        raise ValueError("The NumPy SamplerAnsatzLayer uses CPU; set device='cpu'.")
    if epochs < 1 or lr <= 0 or patience < 0:
        raise ValueError("Expected epochs >= 1, lr > 0 and patience >= 0.")
    model = model.cpu()
    report_context = with_torchinfo_summary(report_context, model)
    log_dir = Path(log_dir or Path("lightning_logs") / uuid4().hex).resolve()
    # An explicit version avoids racing CSVLogger's auto-increment in workers.
    logger = CSVLogger(save_dir=str(log_dir.parent), name=log_dir.name, version="")
    checkpoint = ModelCheckpoint(
        dirpath=str(log_dir / "checkpoints"), filename="best-{epoch:03d}",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
    )
    progress = SubsetProgress(
        progress_callback, epochs, len(train_loader), len(val_loader), len(test_loader),
    )
    module = GEQIELightningModule(model, num_classes=num_classes, lr=lr, verbose=verbose)
    trainer = L.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1,
        logger=logger, log_every_n_steps=1, num_sanity_val_steps=0,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[progress, EarlyStopping(monitor="val_loss", patience=patience, mode="min"), checkpoint],
    )
    if report_context is not None:
        report_context["training_setup"] = {
            **report_context.get("training_setup", {}),
            "training_backend": "lightning", "qnn_lr": lr, "head_lr": lr,
            "scheduler": "ReduceLROnPlateau", "scheduler_factor": 0.5,
            "scheduler_patience": 3, "early_stopping_patience": patience,
            "lightning_log_dir": str(log_dir), "lightning_version": L.__version__,
        }
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if verbose:
        print_epoch_table_footer()
    trainer.test(module, dataloaders=test_loader, ckpt_path="best", verbose=False)
    test = module.test_result
    matrix = confusion_matrix(test["y_true"], test["y_pred"], labels=list(range(num_classes)))
    test_metrics = {name: test[name] for name in ("loss", "accuracy", "precision", "recall", "f1")}
    if verbose:
        print_metrics_report(title="TEST RESULTS", metrics=test_metrics, matrix=matrix)
    # Return the plain model: no Trainer, loaders or progress queue cross IPC.
    return {
        "model": model, "history": module.history, "report_context": report_context,
        "test_metrics": test_metrics, "confusion_matrix": matrix,
        "y_true": test["y_true"], "y_pred": test["y_pred"],
        "lightning_log_dir": str(log_dir), "best_checkpoint_path": checkpoint.best_model_path,
    }
