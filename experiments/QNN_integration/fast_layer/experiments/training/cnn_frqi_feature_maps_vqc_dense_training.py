"""Lightning training — `cnn_frqi_feature_maps_sampler_ansatz_dense` (`CNN-FRQI-Q5`).

A small CNN produces 16 feature maps of size 4x4; each map is FRQI-encoded live via `GEQIELayer` (5 qubits), run through
its own trainable `SamplerAnsatzLayer`, and the 16 probability vectors are flattened into a dense classification head.
There is no precompute step: the feature maps change every training step, so the encoding is evaluated live.
"""

import torch
import torch.nn as nn
from geqie_qml import GEQIELayer, SamplerAnsatzLayer
from geqie_qml.ansatze import default_vqc_ansatz

import training_common as common
from training_common import IMAGE_SIZE, N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 16
FEATURE_MAPS = 16  # CNN output channels (also the number of independent ansatz layers)
N_QUBITS = 5  # FRQI on a 4x4 feature map: 2*ceil(log2(4)) + 1
N_LAYERS = 1  # default_vqc_ansatz repetitions, per feature map
LEARNING_RATE = 0.1


class CNNFRQISamplerAnsatzDenseClassifier(nn.Module):
    def __init__(
        self, num_qubits: int, num_layers: int, num_classes: int, feature_maps: int, scale_output: bool = True
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.scale_output = scale_output
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder = GEQIELayer(geqie_encoding="frqi")
        self.vqcs = nn.ModuleList([
            SamplerAnsatzLayer(num_qubits, default_vqc_ansatz(num_qubits, num_layers=num_layers), seed=i)
            for i in range(feature_maps)
        ])
        self.flatten = nn.Flatten()
        self.head = nn.Linear(feature_maps * (2 ** num_qubits), num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)  # (B, F, 4, 4)
        b, f, h, w = x.shape
        states = self.encoder(x.reshape(b * f, h, w)).reshape(b, f, -1)
        x = torch.stack([vqc(states[:, i]) for i, vqc in enumerate(self.vqcs)], dim=1)
        if self.scale_output:
            x = x * 2 ** self.num_qubits
        x = self.flatten(x)
        x = self.head(x)
        return self.log_softmax(x)


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.image_loaders(context.data_block, BATCH_SIZE, drop_last_train=True)
    model = CNNFRQISamplerAnsatzDenseClassifier(N_QUBITS, N_LAYERS, N_CLASSES, FEATURE_MAPS)
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))


SPEC = ModelSpec(name="cnn_frqi_feature_maps_sampler_ansatz_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
