from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.QNN_integration.datasets.dataset_structure import DataBlock, DataSet, DatasetSplit
from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models import (
	cnn_feature_maps_vqc_dense as shared_pipeline,
)
from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.direct_geqie import (
	default_server_zip_root,
)
from experiments.QNN_integration.experimental_pipelines.experiment.geqie.neqr import (
	cnn_feature_maps_vqc_dense as neqr_pipeline,
)


def test_encoding_qubits_are_inferred_from_encoding_module_and_params():
	assert shared_pipeline.infer_encoded_feature_map_qubits("frqi", (4, 4)) == 5
	assert shared_pipeline.infer_encoded_feature_map_qubits(
		"neqr",
		(4, 4),
		{"bitrate": 4},
	) == 8


def test_server_archive_paths_cover_mnist_and_cifar_variants():
	assert default_server_zip_root("mnist_fashion", "frqi") == Path(
		"/mnt/data02/mkordasz/circuits/FRQI/MNIST_Fashion"
	)
	assert default_server_zip_root("cifar_bw", "neqr") == Path(
		"/mnt/data02/mkordasz/circuits/NEQR/CIFAR-BW"
	)
	assert default_server_zip_root("cifar_rgb", "frqi") == Path(
		"/mnt/data02/mkordasz/circuits/FRQI/CIFAR-RGB"
	)


def test_neqr_runner_propagates_encoding_configuration(monkeypatch):
	monkeypatch.setattr(shared_pipeline, "run_subsets", lambda **kwargs: kwargs)
	images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
	labels = np.array([0, 1])
	split = DatasetSplit(images, labels)
	dataset = DataSet([DataBlock(split, split, split)], {"name": "synthetic CIFAR-RGB"})

	configuration = neqr_pipeline.run(dataset=dataset)
	subset_configuration = configuration["subset_kwargs_factory"](0, None)

	assert configuration["encoding_id"] == "neqr"
	assert configuration["num_qubits"] == 10
	assert configuration["training_setup_extra"]["encoding_method"] == "neqr"
	assert configuration["training_setup_extra"]["encoding_params"] == {"bitrate": 4}
	assert subset_configuration["encoding_id"] == "neqr"
	assert subset_configuration["encoding_params"] == {"bitrate": 4}
	assert "NEQR" in configuration["model_architecture"]


def test_feature_map_layer_keeps_selected_encoding():
	model = shared_pipeline.CNNFeatureMapsGEQIEVQCDenseClassifier(
		num_classes=10,
		num_qubits=10,
		num_layers=1,
		batch_size=2,
		encoding_id="NEQR",
		encoding_params={"bitrate": 4},
		input_shape=(3, 32, 32),
	)

	assert model.encoding_id == "neqr"
	assert model.cnn[0].in_channels == 3
	assert model.quantum.geqie_encoding == "neqr"
	assert model.quantum.encoding_params == {"bitrate": 4}
	assert model.head.in_features == 16 * (2**10)
