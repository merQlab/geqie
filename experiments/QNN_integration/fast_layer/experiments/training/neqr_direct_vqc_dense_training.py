"""Lightning training — `neqr_direct_vqc_dense` (`NEQR-Q12`).

Image (16x16) → NEQR unitary (bitrate=4) → UnitaryInputLayer → SamplerAnsatzLayer(default VQC) → Linear → LogSoftmax.
NEQR on a 16x16 image with bitrate=4 needs 2*ceil(log2(16)) + 4 = 12 qubits. Encodings are precomputed once per subset.
"""

import torch.nn as nn
from geqie_qml import SamplerAnsatzLayer, UnitaryInputLayer
from geqie_qml.ansatze import default_vqc_ansatz

import training_common as common
from training_common import N_CLASSES, ModelSpec, SubsetContext, TrainingSetup

ENCODING = "neqr"
ENCODING_PARAMS = {"bitrate": 4}
BATCH_SIZE = 32
N_LAYERS = 5  # default_vqc_ansatz repetitions
LEARNING_RATE = 0.1


def build(context: SubsetContext) -> TrainingSetup:
    train_loader, val_loader, test_loader, n_qubits = common.precomputed_zip_loaders(
        context, ENCODING, ENCODING_PARAMS, BATCH_SIZE
    )
    ansatz = default_vqc_ansatz(n_qubits, num_layers=N_LAYERS)
    model = nn.Sequential(
        UnitaryInputLayer(n_qubits),  # (B, dim, dim) complex -> (B, dim) complex
        SamplerAnsatzLayer(n_qubits, ansatz),  # (B, dim) -> (B, dim) probs
        nn.Linear(2 ** n_qubits, N_CLASSES),
        nn.LogSoftmax(dim=-1),
    )
    return TrainingSetup(model, train_loader, val_loader, test_loader)


SPEC = ModelSpec(name="neqr_direct_vqc_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
