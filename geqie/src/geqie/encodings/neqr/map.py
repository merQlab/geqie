import numpy as np
from typing import Any

from qiskit.quantum_info import Operator

I_GATE = np.eye(2)
X_GATE = np.array([
	[0, 1], 
	[1, 0]
])


def map(u: int, v: int, R: int, image: np.ndarray, bitrate:int=8, **_: Any) -> Operator:
	if bitrate <= 0 or bitrate > 8:
		raise ValueError("bitrate must be between 1 and 8")
	p = int(image[u, v])
	if p >= 2**bitrate:
		if p <= 255 and bitrate < 8:
			p >>= 8 - bitrate
		else:
			raise ValueError(f"pixel value {p} does not fit in {bitrate} bits")

	# Convert value to binary string, without '0b' and padded with 0s
	pixel_value_as_binary_string = f"{p:0{bitrate}b}"
	# Convert to logic array:
	pixel_value_as_binary_array = [int(bit) for bit in pixel_value_as_binary_string][::-1]

	if pixel_value_as_binary_array[0] == 1:
		map_operator = X_GATE
	else:
		map_operator = I_GATE

	for bit in pixel_value_as_binary_array[1:bitrate]:
		if bit == 1:
			map_operator = np.kron(X_GATE, map_operator)
		else:
			map_operator = np.kron(I_GATE, map_operator)

	return Operator(map_operator)
