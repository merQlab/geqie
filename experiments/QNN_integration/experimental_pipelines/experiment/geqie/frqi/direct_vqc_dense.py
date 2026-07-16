"""FRQI -> direct VQC -> dense experimental pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
	for candidate in Path(__file__).resolve().parents:
		if (candidate / "geqie-qml" / "src" / "geqie_qml").exists():
			sys.path.insert(0, str(candidate))
			break

from experiments.QNN_integration.experimental_pipelines.experiment.geqie.models.direct_geqie import run_direct_geqie


def run(**kwargs):
	return run_direct_geqie(encoding_id="frqi", model_id="direct_vqc_dense", **kwargs)


if __name__ == "__main__":
	run()
