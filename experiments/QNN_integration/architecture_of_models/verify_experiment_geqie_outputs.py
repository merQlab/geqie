"""Verify the four encoding-independent experimental GEQIE figures."""

from verify_architecture_outputs import verify_outputs
from generate_experiment_geqie_architectures import OUTPUT_DIR


if __name__ == "__main__":
	verify_outputs(OUTPUT_DIR, expected_count=4)
