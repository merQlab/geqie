"""Verify completeness, PDF readability and PNG resolution of baseline figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "output" / "baseline"


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	return parser.parse_args()


def verify_outputs(output_dir: Path, *, expected_count: int) -> None:
	output_dir = output_dir.resolve()
	manifest_path = output_dir / "manifest.json"
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	architectures = manifest.get("architectures", [])
	if len(architectures) != expected_count:
		raise AssertionError(
			f"Expected {expected_count} architectures, found {len(architectures)}."
		)

	for entry in architectures:
		for field in ("tex", "pdf", "png"):
			path = output_dir / entry[field]
			if not path.is_file() or path.stat().st_size == 0:
				raise AssertionError(f"Missing or empty output: {path}")

		pdf_path = output_dir / entry["pdf"]
		reader = PdfReader(pdf_path)
		if len(reader.pages) != 1:
			raise AssertionError(f"Expected a one-page PDF: {pdf_path}")

		png_path = output_dir / entry["png"]
		with Image.open(png_path) as image:
			dpi = image.info.get("dpi", (0, 0))
			if min(dpi) < 299:
				raise AssertionError(f"PNG resolution is below 300 DPI: {png_path} ({dpi})")
			if tuple(entry["pixels"]) != image.size:
				raise AssertionError(f"PNG dimensions disagree with manifest: {png_path}")
		print(f"OK  {entry['key']}: {entry['pixels'][0]}x{entry['pixels'][1]} px @ {dpi[0]:.1f} DPI")


def main() -> int:
	args = _parse_args()
	verify_outputs(args.output_dir, expected_count=5)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
