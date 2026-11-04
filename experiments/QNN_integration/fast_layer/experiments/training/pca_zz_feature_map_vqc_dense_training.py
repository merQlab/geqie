"""Lightning training — `pca_zz_feature_map_vqc_dense`.

Image → Flatten → PCA(qubits) → ZZFeatureMap statevector → SamplerAnsatzLayer(VQC) → Linear → LogSoftmax.
"""

import torch
import torch.nn as nn
from geqie_qml import SamplerAnsatzLayer
from geqie_qml.ansatze import default_vqc_ansatz

import training_common as common
from feature_maps import ZZFeatureMapStatevector
from training_common import N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 16
N_QUBITS = 12  # PCA components / ZZFeatureMap register
N_LAYERS = 1  # default_vqc_ansatz repetitions
LEARNING_RATE = 0.1


class PCAZZVQCDenseClassifier(nn.Module):
    def __init__(self, num_qubits: int, num_layers: int, num_classes: int):
        super().__init__()
        self.encoding = ZZFeatureMapStatevector(num_qubits)
        ansatz = default_vqc_ansatz(num_qubits, num_layers=num_layers)
        self.qnn = SamplerAnsatzLayer(
            num_qubits,
            ansatz,
            weight_init=torch.empty(ansatz.num_parameters).uniform_(-torch.pi, torch.pi),
        )
        self.head = nn.Linear(2 ** num_qubits, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoding(x)
        x = self.qnn(x)
        x = self.head(x)
        return self.log_softmax(x)


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.pca_loaders(context.data_block, BATCH_SIZE, N_QUBITS)
    model = PCAZZVQCDenseClassifier(N_QUBITS, N_LAYERS, N_CLASSES)
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, N_QUBITS))


SPEC = ModelSpec(name="pca_zz_feature_map_vqc_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
