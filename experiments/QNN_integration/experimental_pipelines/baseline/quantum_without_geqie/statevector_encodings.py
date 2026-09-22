"""Differentiable input state preparation for the NumPy SamplerAnsatzLayer."""

import torch
from torch import nn


class ZZFeatureMapStatevector(nn.Module):
    """Qiskit's default full-entanglement ZZ feature map with ``reps=1``.

    Hadamards prepare a uniform state. Single-qubit phase gates contribute
    ``2*x[i]`` on bit 1; each CX-P-CX pair contributes
    ``2*(pi-x[i])*(pi-x[j])`` on odd parity. Basis indices use Qiskit's
    little-endian ordering. Torch operations keep the input differentiable.
    """

    def __init__(self, num_qubits: int):
        super().__init__()
        if num_qubits < 2:
            raise ValueError("ZZFeatureMap requires at least two qubits.")
        self.num_qubits = num_qubits
        indices = torch.arange(2 ** num_qubits)
        bits = (indices.unsqueeze(0) >> torch.arange(num_qubits).unsqueeze(1)) & 1
        pairs = torch.triu_indices(num_qubits, num_qubits, offset=1)
        self.register_buffer("bits", bits, persistent=False)
        self.register_buffer("pairs", pairs, persistent=False)
        self.register_buffer("parities", bits[pairs[0]] ^ bits[pairs[1]], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_qubits:
            raise ValueError(f"Expected (batch, {self.num_qubits}) features, got {tuple(x.shape)}.")
        # Match the double-precision simulation even for float32 CNN outputs.
        x = x.to(torch.float64)
        phase = 2 * (x @ self.bits.to(x.dtype))
        pair_angles = (torch.pi - x[:, self.pairs[0]]) * (torch.pi - x[:, self.pairs[1]])
        phase = phase + 2 * (pair_angles @ self.parities.to(x.dtype))
        return torch.exp(1j * phase) / (2 ** self.num_qubits) ** 0.5


class RXFeatureMapStatevector(nn.Module):
    """Prepare ``RX(x[0]) ⊗ ...`` from zero, with qubit 0 least significant."""

    def __init__(self, num_qubits: int):
        super().__init__()
        if num_qubits < 1:
            raise ValueError("Angle embedding requires at least one qubit.")
        self.num_qubits = num_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_qubits:
            raise ValueError(f"Expected (batch, {self.num_qubits}) features, got {tuple(x.shape)}.")
        x = x.to(torch.float64)
        state = torch.ones((x.shape[0], 1), dtype=torch.complex128, device=x.device)
        for angle in x.unbind(dim=1):
            half_angle = angle.unsqueeze(1) / 2
            state = torch.cat((half_angle.cos() * state, -1j * half_angle.sin() * state), dim=1)
        return state
