"""NEQR (4-bit) -> adaptive QNN inspired by No-QNN -> dense pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
	for candidate in Path(__file__).resolve().parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			sys.path.insert(0, str(candidate))
			break

from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.direct_geqie import run_direct_geqie


def run(*, encoding_params=None, create_circuits=True, **kwargs):
	params = {"bitrate": 4, **(encoding_params or {})}
	return run_direct_geqie(
		encoding_id="neqr",
		model_id="adaptive_qnn_no_qnn_inspired_dense",
		encoding_params=params,
		create_circuits=create_circuits,
		**kwargs,
	)


if __name__ == "__main__":
	run()
