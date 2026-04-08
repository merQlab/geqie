from dataclasses import dataclass
import numpy as np

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
	subsets: list[DataBlock]
	info: dict