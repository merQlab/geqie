"""CNN feature maps -> GEQIE(FRQI) -> VQC -> dense pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
	for candidate in Path(__file__).resolve().parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			sys.path.insert(0, str(candidate))
			break

from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.cnn_feature_maps_vqc_dense import (
	run_cnn_feature_maps_vqc_dense,
)


def run(dataset_id="cifar_rgb", **kwargs):
	kwargs.setdefault("convolution_depth", 0) # Feature extraction only.
	return run_cnn_feature_maps_vqc_dense(encoding_id="mcqi", dataset_id=dataset_id, **kwargs)


if __name__ == "__main__":
	run()
