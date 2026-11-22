"""Lightning training — `neqr_direct_vqc_pauliz_dense` (`NEQR-Q12`, PauliZ read-out).

Image (16x16) → NEQR unitary (bitrate=4) → UnitaryInputLayer → SamplerAnsatzLayer(default VQC) → PauliZExpectationLayer
→ Linear → LogSoftmax. The weight-free ⟨Z_i⟩ read-out shrinks the classical head from Linear(2**12, 10) to
Linear(12, 10). Encodings are precomputed once per subset.
"""

import torch.nn as nn
from geqie_qml import PauliZExpectationLayer, SamplerAnsatzLayer, UnitaryInputLayer
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
        PauliZExpectationLayer(n_qubits),  # (B, dim) -> (B, n_qubits), no parameters
        nn.Linear(n_qubits, N_CLASSES),
        nn.LogSoftmax(dim=-1),
    )
    return TrainingSetup(model, train_loader, val_loader, test_loader)


SPEC = ModelSpec(name="neqr_direct_vqc_pauliz_dense", learning_rate=LEARNING_RATE, build=build)

if __name__ == "__main__":
    common.main(SPEC)
