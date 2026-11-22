# QML integration

`geqie-qml` provides PyTorch layers and data utilities for using GEQIE encodings in quantum machine learning (QML) pipelines.

## Installation

`geqie-qml` requires `geqie` and is installed from source:

```bash
git clone https://github.com/merQlab/geqie.git
cd geqie
pip install geqie/ geqie-qml/
```

## Workflows

There are two ways of feeding images to a model.

### Live encoding

[`GEQIELayer`][geqie_qml.layers.geqie_layer.GEQIELayer] encodes images into statevectors on the fly, during the forward pass. It has no trainable parameters, but propagates input gradients (estimated with finite differences, see `grad_mode`), so preceding classical layers can be trained end-to-end.

```{.python notest}
import torch.nn as nn
from qiskit.circuit.library import real_amplitudes

from geqie_qml import GEQIELayer, SamplerAnsatzLayer

n_qubits = 7  # e.g. FRQI encoding of an 8x8 image: 2 * 3 position qubits + 1 color qubit
model = nn.Sequential(
    GEQIELayer("frqi"),
    SamplerAnsatzLayer(n_qubits, real_amplitudes(n_qubits, reps=4)),
    nn.Linear(2**n_qubits, 10),
    nn.LogSoftmax(dim=-1),
)
```

### Precomputed unitaries

Encoding is the most expensive step, so for repeated training runs it can be done once upfront:

1. [`compute_and_save_circuits`][geqie_qml.precompute.compute_and_save_circuits] encodes a dataset into unitary matrices saved as `.npz` files (in parallel).
2. Zip the resulting `train/` and `test/` directories and load them lazily with [`load_precomputed_zip_matrices`][geqie_qml.load_precomputed_zip_matrices].
3. [`UnitaryInputLayer`][geqie_qml.layers.unitary_input_layer.UnitaryInputLayer] turns each unitary `U` into the encoded statevector `U|0⟩`.

```{.python notest}
import torch.nn as nn
from torch.utils.data import DataLoader
from qiskit.circuit.library import real_amplitudes

from geqie_qml import (
    SamplerAnsatzLayer,
    UnitaryInputLayer,
    compute_and_save_circuits,
    load_precomputed_zip_matrices,
)

compute_and_save_circuits(train_images, train_labels, save_dir="circuits/train", geqie_encoding="frqi")
compute_and_save_circuits(test_images, test_labels, save_dir="circuits/test", geqie_encoding="frqi")
# ...zip the "circuits" directory into "circuits.precomputed.zip"

train_dataset, test_dataset = load_precomputed_zip_matrices("circuits.precomputed.zip")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

n_qubits = 7
model = nn.Sequential(
    UnitaryInputLayer(n_qubits),
    SamplerAnsatzLayer(n_qubits, real_amplitudes(n_qubits, reps=4)),
    nn.Linear(2**n_qubits, 10),
    nn.LogSoftmax(dim=-1),
)
```

[`SamplerAnsatzLayer`][geqie_qml.layers.sampler_ansatz_layer.SamplerAnsatzLayer] accepts any parametrized Qiskit circuit as the ansatz and computes analytic parameter-shift gradients with NumPy statevector simulation.

## Examples

| Notebook | Description |
|---|---|
| [`geqie_qml_precompute.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/geqie_qml_precompute.ipynb) | QML precomputation pipeline |
| [`geqie_qml_training_precomputed.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/geqie_qml_training_precomputed.ipynb) | QML training on precomputed data |
| [`geqie_qml_training_live.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/geqie_qml_training_live.ipynb) | QML training on live images |
