# optiq-geqie <!-- omit in toc -->

<img src="assets/geqie_logo.png" alt="image" width="50%" height="auto">

## Table of contents <!-- omit in toc -->

- [Installation](#installation)
- [Examples](#examples)
  - [Notebooks](#notebooks)
  - [CLI usage](#cli-usage)
    - [Command `geqie list-encodings`](#command-geqie-list-encodings)
    - [Command `geqie encode`](#command-geqie-encode)
    - [Command `geqie simulate`](#command-geqie-simulate)
- [Related research](#related-research)

## Installation

While in `optiq-geqie` directory invoke:

```bash
pip install .
```

## Examples

### Notebooks

See the `optiq-geqie/examples` directory for iPython notebooks.

### CLI usage

#### Command `geqie list-encodings`

```bash
geqie list-encodings
```

#### Command `geqie encode`

```txt
Usage: geqie encode [OPTIONS]

Custom encoding plugins:
  [all forbidden if --encoding is set, otherwise all required]
  --init TEXT
  --data TEXT
  --map TEXT

Other options:
  --encoding TEXT      Name of the encoding from 'encodings' directory
  --image TEXT         Path to the image file  [required]
  --grayscale BOOLEAN  Indication wether the image is grayscale  [default: True]
  -v, --verbose        Increase verbosity (can be used multiple times, up to
                       '-vvv')
  --help               Show this message and exit.
```

**Example**

```bash
geqie encode --encoding frqi --image assets/test_image.png
```

#### Command `geqie simulate`

```txt
Usage: geqie simulate [OPTIONS]

Custom encoding plugins:
  [all forbidden if --encoding is set, otherwise all required]
  --init TEXT
  --data TEXT
  --map TEXT

Other options:
  --encoding TEXT                 Name of the encoding from 'encodings'
                                  directory
  --image TEXT                    Path to the image file  [required]
  --grayscale BOOLEAN             Indication wether the image is grayscale
                                  [default: True]
  -v, --verbose                   Increase verbosity (can be used multiple
                                  times, up to '-vvv')
  --n-shots INTEGER               Number of simulation shots
  --return-qiskit-result BOOLEAN  Return results directly from qiskit  [default:
                                  False]
  --return-padded-counts BOOLEAN  Return state counts including zero-count
                                  states  [default: False]
  --help                          Show this message and exit.
```

**Example**

```bash
geqie simulate --encoding frqi --image assets/test_image.png --n-shots 1024 --return-padded-counts true
```

## Related research

- [ICCS 2024 Poster](https://www.researchgate.net/publication/383184874_General_Quantum_Image_Representation_Model_and_Framework)
