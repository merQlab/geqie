"""CNN feature maps -> GEQIE(NEQR) -> VQC -> dense pipeline."""

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


def run(*, encoding_params=None, **kwargs):
	params = {"bitrate": 4, **(encoding_params or {})}
	return run_cnn_feature_maps_vqc_dense(
		encoding_id="neqr",
		encoding_params=params,
		**kwargs)


if __name__ == "__main__":
	run()
