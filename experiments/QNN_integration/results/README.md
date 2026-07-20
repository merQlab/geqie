# Result layout

Runs are stored under the following hierarchy:

```text
<dataset_id>/<experiment_group>/<model_family>/<encoding_id>/<model_id>/<run_id>/
```

Example:

```text
mnist_digits/experiment/geqie/frqi/direct_vqc_dense/2026-05-06T23-18-00/
```

The same `model_id` is used below different encoding directories when only the
image encoding changes. Historical A-F identifiers are retained only as
`legacy_pipeline_name` and `legacy_pipeline_slug` in migrated JSON metadata.
