"""Exercise real Lightning optimization, checkpoints and subset multiprocessing."""

import csv
from contextlib import contextmanager
import gc
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from qiskit.quantum_info import random_unitary

from experiments.QNN_integration.experimental_pipelines import common
from experiments.QNN_integration.experimental_pipelines.common import lightning_training
from experiments.QNN_integration.experimental_pipelines.experiment.geqie.mcqi import direct_vqc_dense
from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models import direct_geqie
from experiments.QNN_integration.datasets.dataset_structure import DataBlock, DataSet, DatasetSplit
from geqie_qml.ansatze import default_vqc_ansatz


def write_archive(path, qubits=3):
    with zipfile.ZipFile(path, "w") as archive:
        for split in ("train", "val", "test"):
            for index in range(3):
                buffer = io.BytesIO()
                np.savez(buffer, matrix=random_unitary(2 ** qubits, seed=index).data, label=index % 2)
                archive.writestr(f"{split}/matrix_{index}_label_{index % 2}.npz", buffer.getvalue())


@contextmanager
def collect_trainer_cycles():
    """Release Trainer/DataLoader cycles before Windows removes temporary ZIPs."""
    try:
        yield
    finally:
        gc.collect()


class PlateauModule(lightning_training.GEQIELightningModule):
    """A fixed validation loss makes scheduler/early-stop timing deterministic."""

    def validation_step(self, batch, batch_idx):
        _, y = batch
        data = self._epoch_data["val"]
        data["loss_sum"] += len(y)
        data["y_true"].extend(y.tolist())
        data["y_pred"].extend(y.tolist())
        self.log("val_loss", 1.0, on_step=False, on_epoch=True, batch_size=len(y))


class LightningPipelineTests(unittest.TestCase):
    def test_zip_training_updates_weights_and_restores_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, collect_trainer_cycles():
            root = Path(directory)
            write_archive(root / "subset_1.zip")
            torch.manual_seed(42)
            initial = common.GEQIEFirstClassifier(
                num_qubits=3, num_layers=1, num_classes=2,
                ansatz_factory=default_vqc_ansatz, use_sampler_ansatz=True,
            )
            torch.manual_seed(42)
            events = []
            with patch.object(common, "train_model", side_effect=AssertionError("Old training loop used")):
                result = direct_geqie.train_one_subset(
                    0, zip_path=str(root / "subset_1.zip"), model_id="direct_vqc_dense",
                    num_qubits=3, num_layers=1, num_classes=2, epochs=2, batch_size=2,
                    training_backend="lightning", lightning_options={"log_dir": root / "logs"},
                    report_context={"training_setup": {}}, progress_callback=events.append,
                )
            model = result["model"]
            self.assertFalse(torch.equal(initial.vqc[1].weights, model.vqc[1].weights))
            self.assertFalse(torch.equal(initial.head.weight, model.head.weight))
            self.assertEqual(set(common.history_template()), set(result["history"]) - {"lr"})
            self.assertTrue(all(len(values) == 2 for values in result["history"].values()))
            self.assertEqual(result["history"]["lr"], [0.1, 0.1])
            self.assertEqual(result["confusion_matrix"].shape, (2, 2))
            self.assertEqual(events[-1]["status"], "complete")
            self.assertEqual(events[-1]["completed"], events[-1]["total"])
            self.assertEqual(events[-1]["completed"], 10)
            setup = result["report_context"]["training_setup"]
            self.assertEqual(setup["qnn_lr"], setup["head_lr"])
            self.assertEqual(setup["training_backend"], "lightning")
            checkpoint = torch.load(result["best_checkpoint_path"], weights_only=False)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, checkpoint["state_dict"][f"model.{name}"])
            self.assertTrue((root / "logs" / "checkpoints" / "last.ckpt").is_file())
            with (root / "logs" / "metrics.csv").open(newline="") as stream:
                columns = csv.DictReader(stream).fieldnames
            self.assertTrue({"train_loss", "val_loss", "train_acc", "val_acc", "lr"}.issubset(columns))
            self.assertEqual(type(pickle.loads(pickle.dumps(result))["model"]), type(model))
            # The reported test loss uses all samples, including the smaller final batch.
            matrices = torch.tensor(np.stack([random_unitary(8, seed=i).data for i in range(3)]))
            with torch.no_grad():
                expected = torch.nn.functional.nll_loss(model(matrices), torch.tensor([0, 1, 0]))
            self.assertAlmostEqual(result["test_metrics"]["loss"], expected.item(), places=6)

    def test_scheduler_and_early_stopping_use_validation_loss(self):
        with tempfile.TemporaryDirectory() as directory, collect_trainer_cycles():
            root = Path(directory)
            write_archive(root / "subset_1.zip")
            events = []
            with patch.object(lightning_training, "GEQIELightningModule", PlateauModule):
                result = direct_geqie.train_one_subset(
                    0, zip_path=str(root / "subset_1.zip"), model_id="direct_vqc_dense",
                    num_qubits=3, num_layers=1, num_classes=2, epochs=10, batch_size=2,
                    training_backend="lightning",
                    lightning_options={"log_dir": root / "logs", "patience": 5},
                    progress_callback=events.append,
                )
            self.assertEqual(len(result["history"]["train_loss"]), 6)
            self.assertEqual(result["history"]["lr"], [0.1] * 5 + [0.05])
            self.assertTrue(any(event["status"] == "early_stopping" for event in events))
            self.assertTrue(events[-1]["early_stopping"])
            self.assertEqual(events[-1]["completed"], events[-1]["total"])
            checkpoint = torch.load(result["best_checkpoint_path"], weights_only=False)
            self.assertEqual(checkpoint["epoch"], 0)
            for name, value in result["model"].state_dict().items():
                torch.testing.assert_close(value, checkpoint["state_dict"][f"model.{name}"])

    def test_mcqi_entry_point_trains_two_subsets_in_separate_processes(self):
        images = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        split = DatasetSplit(images, np.array([0, 1]))
        dataset = DataSet([DataBlock(split, split, split)], {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in (2, 9):
                write_archive(root / f"subset_{number}.zip", qubits=5)
            result = direct_vqc_dense.run(
                dataset=dataset, zip_root=root, epochs=1, num_layers=1,
                num_classes=2, batch_size=2, max_workers=2, data_loader_workers=1,
                lightning_log_root=root / "logs", lightning_options={"lr": 0.02},
                show_progress_bars=True, save_results=True, results_base_dir=root / "reports",
            )
            subsets = result["subset_results"]
            self.assertEqual(len(subsets), 2)
            self.assertNotEqual(subsets[0]["lightning_log_dir"], subsets[1]["lightning_log_dir"])
            for subset, number in zip(subsets, (2, 9)):
                self.assertEqual(len(subset["history"]["train_loss"]), 1)
                self.assertEqual(subset["history"]["lr"], [0.02])
                self.assertEqual(Path(subset["lightning_log_dir"]).name, f"subset_{number}")
                self.assertTrue(Path(subset["best_checkpoint_path"]).is_file())
                self.assertIn(f"subset_{number}.zip", subset["report_context"]["subset_name"])
                self.assertEqual(subset["report_context"]["training_setup"]["data_loader_workers"], 1)
            self.assertEqual(len(list((root / "reports").rglob("subset_*_best_model.pt"))), 2)
            self.assertEqual(len(list((root / "reports").rglob("subset_*_epochs.csv"))), 2)


if __name__ == "__main__":
    unittest.main()
