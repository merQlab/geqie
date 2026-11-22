"""Lightning training — `cnn_angle_embedding_vqc_pauliz_dense` (Senokosov 2024, PauliZ read-out).

Image → CNN → Linear(qubits) → RX angle embedding → SamplerAnsatzLayer(VQC) → PauliZExpectationLayer → Linear
→ LogSoftmax. The weight-free ⟨Z_i⟩ read-out shrinks the head from Linear(2**9, 10) to Linear(9, 10).
"""

import training_common as common
from cnn_angle_embedding_vqc_dense_training import CNNAngleVQCDenseClassifier
from training_common import IMAGE_SIZE, N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 16
N_QUBITS = 9
N_LAYERS = 1  # default_vqc_ansatz repetitions
LEARNING_RATE = 0.1


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.image_loaders(context.data_block, BATCH_SIZE)
    model = CNNAngleVQCDenseClassifier(
        N_QUBITS, N_LAYERS, N_CLASSES, input_shape=(1, IMAGE_SIZE, IMAGE_SIZE), pauli_z_readout=True
    )
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))


SPEC = ModelSpec(name="cnn_angle_embedding_vqc_pauliz_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
