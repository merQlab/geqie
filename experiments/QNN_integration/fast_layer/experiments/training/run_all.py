"""Train every `.fast_layer` model on all 5 dataset subsets, one after another.

Logs of each run go to ``<log-root>/<model_name>/subset_<n>/`` (``train.log``, ``metrics.csv``, ``hparams.yaml``,
``test_metrics.json``, ``training_curves.html``, ``checkpoints/``). A failed run is logged and does not stop the rest.
Everything printed to stdout (plus Lightning messages and warnings) is also written to ``<log-root>/run_<timestamp>.log``.

Timing: ``batch_times.csv`` (every training batch), ``epoch_times.csv`` (epoch / train / validation time and batch
statistics per epoch) and the ``fit_seconds``, ``test_seconds``, ``total_seconds``, ``epoch_seconds_mean`` and
``batch_seconds_mean`` fields of ``test_metrics.json``; the final summary table lists them for every run.

The data is read from ``/mnt/data02/mkordasz/datasets_joblib/MNIST_Digits_5_subsets_train_val_test_16x16.joblib``
(override with ``--dataset-file``) and the precomputed archives from the per-encoding paths in
``training_common.ENCODING_ARCHIVE_TEMPLATES``.

Examples:
    python run_all.py
    python run_all.py --models dense cnn --subsets 1 2
"""

import argparse
import sys

import training_common as common
import cnn_angle_embedding_vqc_dense_training
import cnn_frqi_feature_maps_vqc_dense_training
import cnn_training
import cnn_zz_feature_map_vqc_dense_training
import dense_training
import frqi_direct_vqc_dense_training
import neqr_direct_vqc_dense_training
import pca_zz_feature_map_vqc_dense_training

SPECS = {
    spec.name: spec
    for spec in (
        dense_training.SPEC,
        cnn_training.SPEC,
        pca_zz_feature_map_vqc_dense_training.SPEC,
        cnn_zz_feature_map_vqc_dense_training.SPEC,
        cnn_angle_embedding_vqc_dense_training.SPEC,
        cnn_frqi_feature_maps_vqc_dense_training.SPEC,
        frqi_direct_vqc_dense_training.SPEC,
        neqr_direct_vqc_dense_training.SPEC,
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", choices=list(SPECS), default=list(SPECS), help="Models to train (default: all)."
    )
    common.add_common_arguments(parser)
    args = parser.parse_args()

    results = common.run_specs([SPECS[name] for name in args.models], args)
    sys.exit(int(any(result["status"] != "ok" for result in results)))


if __name__ == "__main__":
    main()
