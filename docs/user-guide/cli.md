# CLI

Installing `geqie` provides the `geqie` command.

## `geqie list-encodings`

Lists the available [encodings](encodings.md).

```bash
geqie list-encodings
```

## `geqie simulate`

Encodes an image and simulates the circuit locally with Qiskit Aer. Prints the measurement counts as JSON.

```bash
geqie simulate --encoding frqi --image-path assets/test_images/grayscale/test_image_4x4.png --n-shots 1024 --return-padded-counts true
```

| Option | Default | Description |
|---|---|---|
| `--encoding TEXT` | *required* | Name of the encoding |
| `--image-path TEXT` | *required* | Path to the image file (a `.npy` file if `--image-dimensionality` is not 2) |
| `--image-dimensionality INTEGER` | `2` | Number of image dimensions to consider |
| `--bitrate INTEGER` | `8` | Number of color bits for encodings that support it |
| `--verbosity-level TEXT` | `ERROR` | Verbosity level, 0-6 (higher means more verbose) or a level name |
| `--n-shots INTEGER` | | Number of simulation shots |
| `--return-qiskit-result BOOLEAN` | `False` | Return results directly from Qiskit |
| `--return-padded-counts BOOLEAN` | `False` | Return state counts including zero-count states |
| `--output-path TEXT` | | Path to where the results will be written |
| `-e, --encoding-params KEY=VALUE` | | Extra encoding parameters; may be repeated |

Values of `--encoding-params` are auto-cast when possible (e.g. `4`, `0.5`, `true`, `null`, `[1,2]`, `{'x':1}`):

```bash
geqie simulate --encoding neqr --image-path image.png --n-shots 1024 -e bitrate=4 -e grayscale_strategy=average
```

## `geqie execute`

Encodes an image and runs the circuit on [IBM Quantum hardware](ibm-quantum.md). Prints the measurement counts as JSON. Accepts the same options as `geqie simulate`, plus:

| Option | Default | Description |
|---|---|---|
| `--credentials-name TEXT` | | Name of saved IBM Quantum credentials |
| `--api-token TEXT` | | IBM Quantum API token (used with `--instance-crn`) |
| `--instance-crn TEXT` | | IBM Quantum instance CRN |
| `--channel TEXT` | `ibm_quantum_platform` | Qiskit Runtime channel |

If no credentials are given, the default account saved with `QiskitRuntimeService.save_account` is used.

```bash
geqie execute --encoding frqi --image-path assets/test_images/grayscale/test_image_4x4.png --n-shots 1024
geqie execute --encoding frqi --image-path assets/test_images/grayscale/test_image_4x4.png --n-shots 1024 --credentials-name my-ibm-credentials
```

## `geqie retrieve`

Reconstructs the image from measurement counts.

| Option | Default | Description |
|---|---|---|
| `--encoding TEXT` | *required* | Name of the encoding |
| `--result TEXT` | *required* | Measurement counts as JSON |
| `--bitrate INTEGER` | `8` | Number of color bits for encodings that support it |
| `-e, --encoding-params KEY=VALUE` | | Extra encoding parameters; may be repeated |

```bash
geqie retrieve --encoding frqi --result '{"00000": 512, "00001": 512}'
```

See [`experiments_cli.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/experiments_cli.ipynb) for CLI-based experiment workflows.
