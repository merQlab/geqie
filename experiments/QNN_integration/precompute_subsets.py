import argparse
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "geqie-qml" / "src").is_dir())
sys.path[:0] = [str(ROOT / "geqie" / "src"), str(ROOT / "geqie-qml" / "src")]

import geqie_qml  # noqa: E402

DATASET = ROOT / "experiments/QNN_integration/datasets/CIFAR-10/CIFAR-RGB_5_subsets_train_val_test_32x32.joblib"
ENCODING = "mcqi"
OUT_ROOT = Path("/mnt/data02/mkordasz/circuits/MCQI/CIFAR-RGB")


@dataclass
class DatasetSplit:
	X: np.ndarray
	y: np.ndarray


@dataclass
class DataBlock:
	train: DatasetSplit
	val: DatasetSplit
	test: DatasetSplit


@dataclass
class DataSet:
	subsets: list
	info: dict


def load_subsets() -> list:
	# The dataset was pickled from a notebook, so its classes resolve via __main__.
	import __main__

	for class_ in (DatasetSplit, DataBlock, DataSet):
		if not hasattr(__main__, class_.__name__):
			setattr(__main__, class_.__name__, class_)
	return joblib.load(DATASET).subsets


def pack(subset_dir: Path, zip_path: Path) -> None:
	# Build under a temporary name so an interrupted run leaves no truncated archive.
	partial = zip_path.with_suffix(".zip.partial")
	with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
		for matrix_file in sorted(subset_dir.glob("*/*.npz")):
			archive.write(matrix_file, arcname=matrix_file.relative_to(subset_dir).as_posix())
	partial.replace(zip_path)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--start-subset", type=int, default=1)
	parser.add_argument("--workers", type=int, default=None)
	parser.add_argument("--out", type=Path, default=OUT_ROOT)
	parser.add_argument("--force", action="store_true", help="Rebuild archives that already exist.")
	args = parser.parse_args()

	args.out.mkdir(parents=True, exist_ok=True)
	subsets = load_subsets()
	print(f"{DATASET.name}: {len(subsets)} subsets -> {args.out}", flush=True)

	for number, block in enumerate(subsets, start=1):
		zip_path = args.out / f"subset_{number}.zip"
		if number < args.start_subset or (zip_path.is_file() and not args.force):
			print(f"subset {number}: skipped", flush=True)
			continue

		print(f"\n=== subset {number} ===", flush=True)
		subset_dir = args.out / f"subset_{number}"
		for split in ("train", "val", "test"):
			data = getattr(block, split)
			geqie_qml.compute_and_save_circuits(
				data=data.X,
				labels=data.y,
				save_dir=str(subset_dir / split),
				geqie_encoding=ENCODING,
				encoding_params={},
				number_of_workers=args.workers,
			)
		pack(subset_dir, zip_path)
		print(f"wrote {zip_path}", flush=True)


if __name__ == "__main__":
	main()
