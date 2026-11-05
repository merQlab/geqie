# geqie <!-- omit in toc -->

Core quantum image encoding library.

## Table of contents <!-- omit in toc -->

- [API](#api)
  - [Quick start](#quick-start)
  - [Encodings](#encodings)
  - [`geqie.encode`](#geqieencode)
  - [`geqie.simulate`](#geqiesimulate)
  - [`geqie.execute`](#geqieexecute)
  - [`geqie.retrieve`](#geqieretrieve)
- [Examples](#examples)
- [CLI](#cli)
  - [`geqie list-encodings`](#geqie-list-encodings)
  - [`geqie simulate`](#geqie-simulate)
  - [`geqie execute`](#geqie-execute)


## API

### Quick start

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

### Encodings

Each encoding lives in `geqie.encodings.<name>` (e.g., `frqi`, `neqr`, `mcqi`) and exposes:

| Function            | Signature                                     | Returns       | Description                                         |
| ------------------- | --------------------------------------------- | ------------- | --------------------------------------------------- |
| `init_function`     | `(n_qubits, **encoding_params)`               | `Statevector` | Initial state of the circuit                        |
| `data_function`     | `(*coords, R, image, **encoding_params)`      | `Statevector` | Data (pixel position) state for the given pixel     |
| `map_function`      | `(*coords, R, image, **encoding_params)`      | `Operator`    | Map (pixel value) operator for the given pixel      |
| `retrieve_function` | `(results, **encoding_params)`                | `np.ndarray`  | Reconstructs the image from measurement counts      |

`coords` are pixel indices (e.g., `u, v` for 2D images) and `R = ceil(log2(max(image.shape)))`. Custom encodings can be created by providing your own functions with the same signatures.

### `geqie.encode`

Builds a quantum circuit encoding the image using the provided encoding functions.

```python
geqie.encode(
    init_function, data_function, map_function, image,
    image_dimensionality=2,
    perform_measurement=True,
    logging_level=None,
    encoding_params={},
) -> QuantumCircuit
```

| Parameter              | Description                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `init_function`        | Function returning the initial `Statevector`                                          |
| `data_function`        | Function returning the data `Statevector` for a pixel                                 |
| `map_function`         | Function returning the map `Operator` for a pixel                                     |
| `image`                | Image as a NumPy array                                                                |
| `image_dimensionality` | Number of leading image axes treated as spatial coordinates (rest are e.g. channels) |
| `perform_measurement`  | Append measurement of all qubits to the circuit                                       |
| `logging_level`        | Logging level (see `geqie.logging_utils.levels`)                                      |
| `encoding_params`      | Extra keyword arguments passed to the encoding functions (e.g., `{"bitrate": 4}`)     |

### `geqie.simulate`

Simulates the circuit locally with Qiskit Aer.

```python
geqie.simulate(
    circuit, n_shots,
    return_qiskit_result=False,
    return_padded_counts=False,
    device="CPU",
    method="automatic",
    noise_model=None,
    logging_level=None,
) -> dict[str, int] | qiskit.result.Result
```

| Parameter              | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `circuit`              | Circuit returned by `geqie.encode`                                  |
| `n_shots`              | Number of simulation shots                                          |
| `return_qiskit_result` | Return the raw Qiskit `Result` instead of counts                    |
| `return_padded_counts` | Include zero-count states in the returned counts                    |
| `device`               | Aer simulator device (`"CPU"` or `"GPU"`)                           |
| `method`               | Aer simulation method (e.g., `"automatic"`, `"statevector"`)        |
| `noise_model`          | Optional `qiskit_aer.noise.NoiseModel`                              |
| `logging_level`        | Logging level                                                       |

### `geqie.execute`

Executes the circuit on real hardware via IBM Quantum Platform (Qiskit Runtime `SamplerV2`).

```python
geqie.execute(
    circuit, n_shots,
    circuit_param_values={},
    return_qiskit_result=False,
    return_padded_counts=False,
    dry_run=False,
    api_token="",
    instance_crn="",
    credentials_name="",
    backend="",
    channel="ibm_quantum_platform",
    ibm_qp_runtime_args={},
    transpiler_args={},
    logging_level=None,
) -> dict[str, int] | PrimitiveResult | None
```

| Parameter              | Description                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `circuit`              | Circuit returned by `geqie.encode`                                                  |
| `n_shots`              | Number of shots                                                                     |
| `circuit_param_values` | Values for parametrized circuits                                                    |
| `return_qiskit_result` | Return the raw Qiskit Runtime result instead of counts                              |
| `return_padded_counts` | Include zero-count states in the returned counts                                    |
| `dry_run`              | Set up the backend and transpile, but don't submit the job (returns `None`)         |
| `api_token`            | IBM Quantum API token (used with `instance_crn`)                                    |
| `instance_crn`         | IBM Quantum instance CRN                                                            |
| `credentials_name`     | Name of saved `QiskitRuntimeService` credentials (alternative to `api_token`)       |
| `backend`              | Backend name; if empty, the least busy backend with enough qubits is selected      |
| `channel`              | Qiskit Runtime channel                                                              |
| `ibm_qp_runtime_args`  | Extra keyword arguments for `QiskitRuntimeService`                                  |
| `transpiler_args`      | Extra keyword arguments for `qiskit.transpile`                                      |
| `logging_level`        | Logging level                                                                       |

**Example**

```python
counts = geqie.execute(circuit, n_shots=1024, credentials_name="my-ibm-credentials")
```

### `geqie.retrieve`

Reconstructs the image from measurement counts using the encoding's retrieve function.

```python
geqie.retrieve(
    retrieve_function, results,
    image_dimensionality=2,
    encoding_params={},
) -> np.ndarray
```

| Parameter              | Description                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `retrieve_function`    | Encoding's `retrieve_function` (e.g., `frqi.retrieve_function`)                     |
| `results`              | Measurement counts returned by `geqie.simulate` or `geqie.execute`                  |
| `image_dimensionality` | Number of spatial image dimensions (used by encodings that support it, e.g. `mfrqi`) |
| `encoding_params`      | Extra keyword arguments passed to the retrieve function (e.g., `{"bitrate": 4}`)     |

## Examples

See the [`examples/`](../examples/) directory for notebooks covering individual encodings and CLI workflows.

## CLI

### `geqie list-encodings`

```bash
geqie list-encodings
```

### `geqie simulate`

```txt
Usage: geqie simulate [OPTIONS]

Options:
  --encoding TEXT                 Name of the encoding from 'encodings'
                                  directory  [required]
  --image-path TEXT               Path to the image file  [required]
  --image-dimensionality INTEGER  Number of image dimensions to consider
                                  [default: 2]
  --bitrate INTEGER               Number of color bits for encodings that
                                  support it  [default: 8]
  --verbosity-level TEXT          Set verbosity level, 0-6 (higher means more
                                  verbose)
  --n-shots INTEGER               Number of simulation shots
  --return-qiskit-result BOOLEAN  Return results directly from qiskit
                                  [default: False]
  --return-padded-counts BOOLEAN  Return state counts including zero-count
                                  states  [default: False]
  --output-path TEXT              Path to where the results will be written
  -e, --encoding-params KEY=VALUE
                                  Arbitrary extra parameters as key=value
                                  pairs. May be repeated, e.g.,
                                  -e bitrate=4 -e custom_flag=true
  --help                          Show this message and exit.
```

**Example**

```bash
geqie simulate --encoding frqi --image-path assets/test_image.png --n-shots 1024 --return-padded-counts true
```

### `geqie execute`

```txt
Usage: geqie execute [OPTIONS]

Options:
  --encoding TEXT                 Name of the encoding from 'encodings'
                                  directory  [required]
  --image-path TEXT               Path to the image file  [required]
  --image-dimensionality INTEGER  Number of image dimensions to consider
                                  [default: 2]
  --bitrate INTEGER               Number of color bits for encodings that
                                  support it  [default: 8]
  --verbosity-level TEXT          Set verbosity level, 0-6 (higher means more
                                  verbose)
  --n-shots INTEGER               Number of simulation shots
  --return-qiskit-result BOOLEAN  Return results directly from qiskit
                                  [default: False]
  --return-padded-counts BOOLEAN  Return state counts including zero-count
                                  states  [default: False]
  --output-path TEXT              Path to where the results will be written
  -e, --encoding-params KEY=VALUE
                                  Arbitrary extra parameters as key=value
                                  pairs. May be repeated, e.g.,
                                  -e bitrate=4 -e custom_flag=true
  --help                          Show this message and exit.
```

**Example**

```bash
geqie execute --encoding frqi --image-path assets/test_image.png --n-shots 1024
```
