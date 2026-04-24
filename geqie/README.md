# geqie <!-- omit in toc -->

Core quantum image encoding library.

## Table of contents <!-- omit in toc -->

- [Examples](#examples)
- [CLI](#cli)
  - [`geqie list-encodings`](#geqie-list-encodings)
  - [`geqie simulate`](#geqie-simulate)
  - [`geqie execute`](#geqie-execute)


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
