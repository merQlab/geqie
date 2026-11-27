# Examples

## Paper examples

The following examples reproduce the illustrative examples from the [GEQIE paper](https://arxiv.org/abs/2512.24973). Scripts and notebooks for the full experiments, raw results, and Cosmic Web input files are available in the [`experiments/`](https://github.com/merQlab/geqie/tree/main/experiments) directory.

### Basic usage

Encode a 4×4 grayscale image with FRQI, simulate the circuit, and retrieve the image:

```python
import numpy as np
from PIL import Image, ImageOps

import geqie
from geqie.encodings import frqi

image = np.asarray(ImageOps.grayscale(Image.open("assets/test_images/grayscale/test_image_4x4.png")))
circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)
counts = geqie.simulate(circuit, n_shots=4096)
retrieved = frqi.retrieve_function(counts)
```

To change the encoding method, replace the imported module, e.g. `frqi` with `neqr`. The same workflow is available in the [CLI](user-guide/cli.md); encoding-specific parameters are passed with `-e key=value`:

```bash
geqie list-encodings
geqie simulate --encoding frqi --image-path assets/test_images/grayscale/test_image_4x4.png \
    --n-shots 4096 --output-path counts.json
geqie retrieve --encoding frqi --result "$(cat counts.json)"
```

### Benchmarking encodings under noise

Because all encodings share a common interface, they can be compared under identical conditions in a single loop. Here, the single-qubit depolarizing channel is applied to the first qubit of each `unitary` instruction, and the retrieved image is compared with the original using Pearson's correlation coefficient (PCC):

```python
import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.stats import pearsonr

import geqie
from geqie.encodings import frqi, ifrqi

image = np.random.default_rng(0).integers(0, 256, size=(2, 2), dtype=np.uint8)

pccs = {}
for method in (frqi, ifrqi):
    circuit = geqie.encode(method.init_function, method.data_function, method.map_function, image)
    for lam in (0.0, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0):
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(lam, 1), ["unitary"])
        counts = geqie.simulate(circuit, n_shots=1024, noise_model=noise)
        retrieved = method.retrieve_function(counts)
        pccs[method.__name__, lam], _ = pearsonr(image.flatten(), retrieved.flatten())
```

For a 2×2 image, FRQI and IFRQI use 3 and 6 qubits. NEQR and QUALPI can be added to the loop, but they use 10 qubits (8 value qubits), which makes noisy simulation considerably slower. The paper evaluates four grayscale (`frqi`, `ifrqi`, `neqr`, `qualpi`) and four RGB (`frqci`, `mcqi`, `ncqi`, `qrci`) encodings on random 2×2, 4×4, and 8×8 images; see [`experimental_protocol_2D.ipynb`](https://github.com/merQlab/geqie/blob/main/experiments/experimental_protocol_2D.ipynb) and [`experimental_protocol_2D_rgb.ipynb`](https://github.com/merQlab/geqie/blob/main/experiments/experimental_protocol_2D_rgb.ipynb).

### Execution on IBM quantum hardware

Switching from simulation to hardware only requires replacing [`geqie.simulate`][geqie.main.simulate] with [`geqie.execute`][geqie.main.execute]. The returned counts have the same format, so retrieval is unchanged (see [Running on IBM Quantum](user-guide/ibm-quantum.md)):

```{.python notest}
counts = geqie.execute(circuit, n_shots=4096, credentials_name="my-ibm-account", backend="ibm_fez")
retrieved = frqi.retrieve_function(counts)
```

### Encoding a Cosmic Web snapshot

MFRQI encodes multidimensional data by setting `image_dimensionality`. In the paper, dark-matter particle masses from an L-PICOLA snapshot are aggregated into a 16×16×16 voxel grid, mapped to [0, 1] with the invertible function `1 - exp(-x / x̃)` (where `x̃` is the median voxel density), encoded, simulated with 2<sup>20</sup> shots, retrieved, and de-normalized. The example below uses a small synthetic grid in place of the snapshot:

```python
import numpy as np

import geqie
from geqie.encodings import mfrqi

density = np.random.default_rng(0).lognormal(size=(2, 2, 2))

scale = np.median(density)
normalized = 1 - np.exp(-density / scale)
circuit = geqie.encode(
    mfrqi.init_function, mfrqi.data_function, mfrqi.map_function, normalized,
    image_dimensionality=3,
)
counts = geqie.simulate(circuit, n_shots=2**14)
retrieved = mfrqi.retrieve_function(counts, image_dimensionality=3)
density_retrieved = -scale * np.log(1 - retrieved + 1e-60)
```

See [`cosmic_web/geqie_mfrqi_encoding.ipynb`](https://github.com/merQlab/geqie/blob/main/experiments/cosmic_web/geqie_mfrqi_encoding.ipynb) for the full experiment.

### Circuit cost and scalability

The cost of a generated circuit can be measured by transpiling it to the {CX, U} basis:

```python
import numpy as np
from PIL import Image, ImageOps
from qiskit import transpile

import geqie
from geqie.encodings import frqi

image = np.asarray(ImageOps.grayscale(Image.open("assets/test_images/grayscale/test_image_4x4.png")))
circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)

qc = transpile(circuit, basis_gates=["cx", "u"], optimization_level=1)
cnot_count = qc.count_ops()["cx"]
```

See [`circuit_cost_benchmark.ipynb`](https://github.com/merQlab/geqie/blob/main/experiments/circuit_cost_benchmark.ipynb) for the comparison with the gate-level FRQI construction and Qiskit `StatePreparation`.

## Notebooks

The [`examples/`](https://github.com/merQlab/geqie/tree/main/examples) directory contains Jupyter notebooks.

### GEQIE

| Notebook | Description |
|---|---|
| [`encodings/`](https://github.com/merQlab/geqie/tree/main/examples/encodings) | One notebook per [built-in encoding](user-guide/encodings.md), e.g. [`frqi.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/encodings/frqi.ipynb) |
| [`execute.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/execute.ipynb) | Execution on IBM Quantum Platform with `geqie.execute` |
| [`experiments_cli.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/experiments_cli.ipynb) | CLI-based experiment workflows |
| [`cosmic_web/geqie_mfrqi_encoding.ipynb`](https://github.com/merQlab/geqie/blob/main/experiments/cosmic_web/geqie_mfrqi_encoding.ipynb) | MFRQI encoding of the Cosmic Web |

### GEQIE QML

| Notebook | Description |
|---|---|
| [`geqie_qml_precompute.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/geqie_qml_precompute.ipynb) | QML precomputation pipeline |
| [`geqie_qml_training_precomputed.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/geqie_qml_training_precomputed.ipynb) | QML training on precomputed data |
| [`geqie_qml_training_live.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/geqie_qml_training_live.ipynb) | QML training on live images |

Requirements for running the notebooks are listed in [`examples/requirements/requirements.in`](https://github.com/merQlab/geqie/blob/main/examples/requirements/requirements.in).
