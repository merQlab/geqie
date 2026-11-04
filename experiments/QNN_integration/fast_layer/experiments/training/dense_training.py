"""Lightning training — `dense`: flattened image pixels into a small MLP classifier (classical lower reference)."""

import torch.nn as nn

import training_common as common
from training_common import IMAGE_SIZE, N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

BATCH_SIZE = 64
HIDDEN_DIM = 32
LEARNING_RATE = 0.1


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader = common.image_loaders(context.data_block, BATCH_SIZE)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(IMAGE_SIZE * IMAGE_SIZE, HIDDEN_DIM),
        nn.Sigmoid(),
        nn.Dropout(0.2),
        nn.Linear(HIDDEN_DIM, N_CLASSES),
        nn.LogSoftmax(dim=-1),
    )
    return TrainingSetup(model, train_loader, val_loader, test_loader, (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE))


SPEC = ModelSpec(name="dense", learning_rate=LEARNING_RATE, build=build, accelerator="auto")

if __name__ == "__main__":
    common.main(SPEC)
