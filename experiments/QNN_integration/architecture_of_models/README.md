# Architecture figures for QNN integration

This directory contains reproducible scripts and publication-ready outputs for
the models used in `experimental_pipelines/baseline`.

## Generate all baseline figures

From the repository root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe experiments\QNN_integration\architecture_of_models\generate_baseline_architectures.py
```

Generate only one model by passing `--model`; list the available identifiers
with `--help`. PNG output is rendered at 300 DPI by default. The script rejects
values below 300 DPI.

The generator requires:

- `pytorch2tikz==0.0.1`;
- a `torchvision` version compatible with the installed PyTorch (an undeclared
  runtime dependency of `pytorch2tikz`);
- TeX Live/MiKTeX with `pdflatex` and the `standalone`/TikZ packages;
- Poppler's `pdftoppm` for 300-DPI rasterization.

Python dependencies can be installed from the local requirements file. Ensure
that the selected `torchvision` build is compatible with the project's PyTorch
version:

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments\QNN_integration\architecture_of_models\requirements.txt
```

## Outputs

Files are written to `output/baseline`:

- `.tex` - editable standalone TikZ source;
- `.pdf` - preferred vector format for inclusion in a LaTeX article;
- `.png` - raster image with 300-DPI metadata;
- `manifest.json` - source-model mapping and output dimensions.

The hybrid architectures are traced from their real classical PyTorch layers.
PCA and Qiskit's `TorchConnector` are outside the graph understood by
`pytorch2tikz`, so the generator inserts drawing-only proxy blocks with the
actual dimensions (`q` and `2^q`). These proxies do not modify training models.

## Verify generated files

```powershell
.\.venv\Scripts\python.exe experiments\QNN_integration\architecture_of_models\verify_architecture_outputs.py
```

## Experimental GEQIE models

Generate the four encoding-independent experiment architectures from
`experimental_pipelines/experiment/geqie`:

```powershell
.\.venv\Scripts\python.exe experiments\QNN_integration\architecture_of_models\generate_experiment_geqie_architectures.py
```

Their outputs are written to `output/experiment/geqie`. Verify them with:

```powershell
.\.venv\Scripts\python.exe experiments\QNN_integration\architecture_of_models\verify_experiment_geqie_outputs.py
```

The diagrams intentionally use the common `GEQIE` label rather than separate
FRQI and NEQR variants. Dimensions which depend on the selected image encoding
are expressed symbolically as `q` and `2^q`; the encoding choice can therefore
be documented independently in the article text without duplicating figures.
