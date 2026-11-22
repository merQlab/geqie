"""Lightning training — `pca_zz_feature_map_vqc_pauliz_dense` (PauliZ read-out).

Image → Flatten → PCA(qubits) → ZZFeatureMap statevector → SamplerAnsatzLayer(VQC) → PauliZExpectationLayer → Linear
→ LogSoftmax. The weight-free ⟨Z_i⟩ read-out shrinks the head from Linear(2**12, 10) to Linear(12, 10).
"""

import training_common as common
from pca_zz_feature_map_vqc_dense_training import PCAZZVQCDenseClassifier
from training_common import N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 16
N_QUBITS = 12  # PCA components / ZZFeatureMap register
N_LAYERS = 1  # default_vqc_ansatz repetitions
LEARNING_RATE = 0.1


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.pca_loaders(context.data_block, BATCH_SIZE, N_QUBITS)
    model = PCAZZVQCDenseClassifier(N_QUBITS, N_LAYERS, N_CLASSES, pauli_z_readout=True)
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, N_QUBITS))


SPEC = ModelSpec(name="pca_zz_feature_map_vqc_pauliz_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
