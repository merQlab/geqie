import torch
import torch.nn as nn


class PauliZExpectationLayer(nn.Module):
    """
    Weight-free read-out: measurement probability distribution -> single-qubit
    ``⟨Z_i⟩`` expectation values.

    ``⟨Z_i⟩ = Σ_x p(x) · (+1 if bit i of x is 0 else -1)`` is a constant linear
    map with no trainable parameters, so a classical head placed after it only
    needs ``n_qubits`` inputs instead of ``2**n_qubits``.

    Bit ``i`` of basis index ``x`` follows Qiskit's little-endian convention
    (qubit 0 = least significant bit), matching :class:`SamplerAnsatzLayer`.

    Parameters
    ----------
    n_qubits : int

    Input / Output
    --------------
    Input  : ``(..., 2**n_qubits)`` float — probability distributions (e.g. from
             :class:`SamplerAnsatzLayer`).
    Output : ``(..., n_qubits)`` float — ``⟨Z_i⟩`` in ``[-1, 1]`` per qubit.

    Example
    -------
    >>> model = nn.Sequential(
    ...     UnitaryInputLayer(7),
    ...     SamplerAnsatzLayer(7, ansatz),
    ...     PauliZExpectationLayer(7),
    ...     nn.Linear(7, 10),
    ...     nn.LogSoftmax(dim=-1),
    ... )
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        bits = (torch.arange(self.dim).unsqueeze(1) >> torch.arange(n_qubits)) & 1  # (2**n, n)
        self.register_buffer("signs", 1.0 - 2.0 * bits.float(), persistent=False)  # +1 for |0⟩, -1 for |1⟩

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.shape[-1] != self.dim:
            raise ValueError(
                f"Expected (..., {self.dim}) probabilities, got {tuple(probs.shape)}."
            )
        return probs @ self.signs.to(probs.dtype)

    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}, dim={self.dim}"
