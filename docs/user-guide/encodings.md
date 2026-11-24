# Encodings

An encoding in GEQIE is a set of functions describing how an image is mapped onto a quantum state. GEQIE combines the per-pixel `data` states and `map` operators into a single unitary, applies it to the `init` state, and builds the resulting circuit.

## Built-in encodings

Each built-in encoding lives in `geqie.encodings.<name>`:

| Name | Encoding | Image type | Example |
|---|---|---|---|
| `frqi` | Flexible Representation of Quantum Images ([Le et al., 2011](https://link.springer.com/article/10.1007/s11128-010-0177-y)) | Grayscale | [frqi.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/frqi.ipynb) |
| `ifrqi` | Improved FRQI ([Khan, 2019](https://link.springer.com/article/10.1007/s11128-019-2306-6)) | Grayscale | [ifrqi.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/ifrqi.ipynb) |
| `neqr` | Novel Enhanced Quantum Representation ([Zhang et al., 2013](https://link.springer.com/article/10.1007/s11128-013-0567-z)) | Grayscale | [neqr.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/neqr.ipynb) |
| `qualpi` | Quantum Log-Polar Image ([Zhang et al., 2013](https://link.springer.com/article/10.1007/s11128-013-0587-8)) | Grayscale | [qualpi.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/qualpi.ipynb) |
| `frqci` | Flexible Representation of Quantum Color Images ([Li et al., 2016](https://link.springer.com/article/10.1007/s11128-016-1413-x)) | RGB | [frqci.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/frqci.ipynb) |
| `mcqi` | Multi-Channel Quantum Images ([Sun et al., 2013](https://doi.org/10.20965/jaciii.2013.p0404)) | RGB | [mcqi.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/mcqi.ipynb) |
| `ncqi` | Novel quantum representation of Color Images ([Sang et al., 2017](https://link.springer.com/article/10.1007/s11128-016-1463-0)) | RGB | [ncqi.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/ncqi.ipynb) |
| `qrci` | Quantum Representation of Color Images ([Wang et al., 2019](https://doi.org/10.1016/j.optcom.2019.01.015)) | RGB | [qrci.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/qrci.ipynb) |
| `mfrqi` | Multidimensional FRQI ([Li et al., 2014](https://link.springer.com/article/10.1007/s11128-013-0705-7)) | N-dimensional | [mfrqi.ipynb](https://github.com/merQlab/geqie/blob/main/examples/encodings/mfrqi.ipynb) |

Grayscale encodings accept color images too: they convert them with [`ensure_grayscale`][geqie.encodings.ensure_grayscale] using the `grayscale_strategy` encoding parameter (`"luminance"` by default, `"first_channel"`, or `"average"`).

`neqr` additionally accepts a `bitrate` encoding parameter (1-8, default 8) setting the number of qubits used to store pixel values.

## Encoding interface

Each encoding module exposes:

| Function | Signature | Returns | Description |
|---|---|---|---|
| `init_function` | `(n_qubits, **encoding_params)` | `Statevector` | Initial state of the circuit |
| `data_function` | `(*coords, R, image, **encoding_params)` | `Statevector` | Data (pixel position) state for the given pixel |
| `map_function` | `(*coords, R, image, **encoding_params)` | `Operator` | Map (pixel value) operator for the given pixel |
| `retrieve_function` | `(results, **encoding_params)` | `np.ndarray` | Reconstructs the image from measurement counts (optional) |

`coords` are pixel indices (e.g. `u, v` for 2D images) and `R = ceil(log2(max(image.shape)))`. Every function should accept `**kwargs`, as all `encoding_params` are passed to all functions.

## Custom encodings

Any functions with the signatures above can be passed to [`geqie.encode`][geqie.main.encode]. For example, FRQI can be written as:

```python
import numpy as np
from PIL import Image
from qiskit.quantum_info import Operator, Statevector

import geqie


def init(n_qubits: int, **_) -> Statevector:
    return Statevector(np.tile([1, 0], 2 ** (n_qubits - 1)))


def data(u: int, v: int, R: int, image: np.ndarray, **_) -> Statevector:
    data_vector = np.zeros(2 ** (2 * R))
    data_vector[np.ravel_multi_index((u, v), image.shape)] = 1
    return Statevector(data_vector)


def map(u: int, v: int, R: int, image: np.ndarray, **_) -> Operator:
    p = image[u, v] / 255.0 * (np.pi / 2)
    return Operator([
        [np.cos(p), -np.sin(p)],
        [np.sin(p),  np.cos(p)],
    ])


image = np.asarray(Image.open("assets/test_images/grayscale/test_image_4x4.png").convert("L"))
circuit = geqie.encode(init, data, map, image)
```

To make a custom encoding available in the [CLI](cli.md), add it as a package `geqie/src/geqie/encodings/<name>/` whose `__init__.py` exposes `init_function`, `data_function`, `map_function`, and `retrieve_function` (see the built-in encodings for reference).
