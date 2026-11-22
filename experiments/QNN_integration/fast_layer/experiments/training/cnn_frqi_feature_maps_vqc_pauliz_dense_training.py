"""Lightning training — `cnn_frqi_feature_maps_sampler_ansatz_pauliz_dense` (`CNN-FRQI-Q7`, PauliZ read-out).

Same as `cnn_frqi_feature_maps_sampler_ansatz_dense`, but each feature map's probability vector is reduced to its
per-qubit ⟨Z_i⟩ values by `PauliZExpectationLayer`, so the head shrinks from Linear(F * 2**n, 10) to Linear(F * n, 10).
Output scaling is disabled since ⟨Z_i⟩ is already in [-1, 1].
"""

import training_common as common
from cnn_frqi_feature_maps_vqc_dense_training import CNNFRQISamplerAnsatzDenseClassifier
from training_common import IMAGE_SIZE, N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 16
CONV1_CHANNELS = 8  # CNN channels after first conv layer
FEATURE_MAPS = 8  # CNN output channels (also the number of independent ansatz layers)
N_QUBITS = 7  # FRQI on a 8x8 feature map: 2*ceil(log2(8)) + 1
N_QUANTUM_LAYERS = 1  # default_vqc_ansatz repetitions, per feature map
N_POOLING_LAYERS = 1
LEARNING_RATE = 0.001


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.image_loaders(context.data_block, BATCH_SIZE, drop_last_train=True)
    model = CNNFRQISamplerAnsatzDenseClassifier(
        N_QUBITS,
        N_QUANTUM_LAYERS,
        N_CLASSES,
        FEATURE_MAPS,
        CONV1_CHANNELS,
        N_POOLING_LAYERS,
        scale_output=False,
        pauli_z_readout=True,
    )
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))


SPEC = ModelSpec(name="cnn_frqi_feature_maps_sampler_ansatz_pauliz_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
