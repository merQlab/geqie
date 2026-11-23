# Getting started

## Installation

GEQIE requires Python 3.10 or newer.

=== "PyPI"

    ```bash
    pip install geqie
    ```

=== "From source"

    ```bash
    git clone https://github.com/merQlab/geqie.git
    cd geqie
    pip install geqie/
    pip install geqie-qml/   # optional, for QML support
    ```

## Quick start

Encode an image with FRQI, simulate the circuit, and reconstruct the image from the measurement counts:

```python
import numpy as np
from PIL import Image

import geqie
from geqie.encodings import frqi

image = np.asarray(Image.open("assets/test_images/grayscale/test_image_4x4.png"))

circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)
counts = geqie.simulate(circuit, n_shots=1024)
reconstructed = geqie.retrieve(frqi.retrieve_function, counts)
```

The workflow always consists of the same three steps:

1. [`geqie.encode`][geqie.main.encode] builds the encoding circuit from the encoding's `init`, `data`, and `map` functions.
2. [`geqie.simulate`][geqie.main.simulate] runs it locally with Qiskit Aer, or [`geqie.execute`][geqie.main.execute] runs it on [IBM Quantum hardware](user-guide/ibm-quantum.md).
3. [`geqie.retrieve`][geqie.main.retrieve] reconstructs the image from the measurement counts.

## Encoding parameters

Extra keyword arguments are forwarded to all encoding functions via `encoding_params`:

```{.python continuation}
from geqie.encodings import neqr

circuit = geqie.encode(
    neqr.init_function, neqr.data_function, neqr.map_function, image,
    encoding_params={"bitrate": 4},
)
counts = geqie.simulate(circuit, n_shots=1024)
reconstructed = geqie.retrieve(neqr.retrieve_function, counts, encoding_params={"bitrate": 4})
```

## Multidimensional and color images

`image_dimensionality` sets how many leading image axes are treated as spatial coordinates; the remaining axes (e.g. RGB channels) are left for the encoding to interpret. Grayscale encodings convert color images with [`ensure_grayscale`][geqie.encodings.ensure_grayscale] (controlled by the `grayscale_strategy` encoding parameter).

## Logging

Pass `logging_level` to any API function to inspect what GEQIE is doing. Besides the standard `logging` levels, GEQIE defines more verbose ones in `geqie.logging_utils.levels`: `MATH` (encoding matrices), `STATE` (per-pixel states and operators), and `TRACE`.

```{.python notest}
from geqie.logging_utils import levels

circuit = geqie.encode(
    frqi.init_function, frqi.data_function, frqi.map_function, image,
    logging_level=levels.INFO,
)
```
