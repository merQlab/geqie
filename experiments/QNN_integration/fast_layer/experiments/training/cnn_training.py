"""Lightning training — `cnn`: two conv blocks plus a dense head, no quantum layer (classical upper reference)."""

import torch
import torch.nn as nn

import training_common as common
from training_common import IMAGE_SIZE, N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 64
LEARNING_RATE = 0.01


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple[int, int, int]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            feature_count = self.features(torch.zeros(1, *input_shape)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.image_loaders(context.data_block, BATCH_SIZE)
    model = CNNClassifier(N_CLASSES, input_shape=(1, IMAGE_SIZE, IMAGE_SIZE))
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))


SPEC = ModelSpec(name="cnn", learning_rate=LEARNING_RATE, build=build, accelerator="auto")

if __name__ == "__main__":
    common.main(SPEC)
