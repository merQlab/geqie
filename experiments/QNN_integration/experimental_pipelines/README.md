# Experimental pipelines

Pipeline names describe model architecture instead of historical letters A-F.
The directory structure separates reference models from GEQIE experiments and
keeps FRQI and NEQR as variants of the same shared model implementation.

Every public `run()` accepts `dataset_id`. Configured values are:

- `mnist_digits`
- `mnist_fashion`

For example:

```python
from experiments.QNN_integration.experimental_pipelines.experiment.geqie.frqi import direct_vqc_dense

direct_vqc_dense.run(dataset_id="mnist_fashion")
```

Direct-GEQIE implementations are shared in `experiment/geqie/models`; files in
`frqi` and `neqr` are independently runnable encoding-specific entry points.

## Console progress

Subset training supports two mutually exclusive console modes:

```python
# One fixed batch-level progress bar per subset (train, validation, and test).
run(verbose=False, show_progress_bars=True)

# Detailed per-epoch reports without progress bars.
run(verbose=True, show_progress_bars=False)

# No live console output.
run(verbose=False, show_progress_bars=False)
```

When `show_progress_bars` is omitted, it defaults to `not verbose`.
