"""Lightning training — `cnn_angle_embedding_vqc_dense` (Senokosov 2024).

Image → CNN → Linear(qubits) → RX angle embedding → SamplerAnsatzLayer(VQC) → Linear → LogSoftmax.
"""

import torch
import torch.nn as nn
from geqie_qml import SamplerAnsatzLayer
from geqie_qml.ansatze import default_vqc_ansatz

import training_common as common
from feature_maps import RXFeatureMapStatevector
from training_common import IMAGE_SIZE, N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 16
N_QUBITS = 9
N_LAYERS = 1  # default_vqc_ansatz repetitions
LEARNING_RATE = 0.1


class CNNAngleVQCDenseClassifier(nn.Module):
    def __init__(self, num_qubits: int, num_layers: int, num_classes: int, input_shape: tuple[int, int, int]):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            feature_count = self.cnn(torch.zeros(1, *input_shape)).numel()
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, num_qubits),
            nn.BatchNorm1d(num_qubits),
            nn.ReLU(),
        )
        self.encoding = RXFeatureMapStatevector(num_qubits)
        ansatz = default_vqc_ansatz(num_qubits, num_layers=num_layers)
        self.qnn = SamplerAnsatzLayer(
            num_qubits,
            ansatz,
            weight_init=torch.empty(ansatz.num_parameters).uniform_(-torch.pi, torch.pi),
        )
        self.head = nn.Linear(2 ** num_qubits, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.feature_head(x)
        x = self.encoding(x)
        x = self.qnn(x)
        x = self.head(x)
        return self.log_softmax(x)


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.image_loaders(context.data_block, BATCH_SIZE)
    model = CNNAngleVQCDenseClassifier(N_QUBITS, N_LAYERS, N_CLASSES, input_shape=(1, IMAGE_SIZE, IMAGE_SIZE))
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))


SPEC = ModelSpec(name="cnn_angle_embedding_vqc_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
