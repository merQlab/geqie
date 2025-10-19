import numpy as np
from tabulate import tabulate


def tabulate_complex(X: np.ndarray, number_format: str = ".3g") -> str:
    return tabulate(np.vectorize(lambda x: f"{x:{number_format}}")(X))
