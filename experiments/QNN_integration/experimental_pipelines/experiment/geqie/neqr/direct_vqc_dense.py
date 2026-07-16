"""NEQR (4-bit) -> direct VQC -> dense experimental pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
	for candidate in Path(__file__).resolve().parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			sys.path.insert(0, str(candidate))
			break

from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.direct_geqie import run_direct_geqie


def run(*, encoding_params=None, **kwargs):
	params = {"bitrate": 4, **(encoding_params or {})}
	return run_direct_geqie(
		encoding_id="neqr",
		model_id="direct_vqc_dense",
		encoding_params=params,
		**kwargs,
	)


if __name__ == "__main__":
	run()
