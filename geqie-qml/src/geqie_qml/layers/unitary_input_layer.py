import torch
import torch.nn as nn


class UnitaryInputLayer(nn.Module):
    """
    Converts a pre-computed image unitary ``U`` into the encoded statevector
    ``U|0⟩`` by extracting the first column.

    No trainable parameters. Output is a complex statevector ready to be
    passed to, e.g., :class:`SamplerAnsatzLayer` or any other quantum layer.

    Parameters
    ----------
    n_qubits : int

    Input / Output
    --------------
    Input  : ``(B, 2**n_qubits, 2**n_qubits)`` complex — pre-computed unitary matrices from the
             data-loader.
    Output : ``(B, 2**n_qubits)`` complex — encoded statevectors ``U|0⟩``.
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-2] == self.dim and x.shape[-1] == self.dim:
            return x[:, :, 0].contiguous()     # U|0⟩ = first column of U
        if x.dim() == 2 and x.shape[-1] == self.dim:
            return x                            # already a statevector
        raise ValueError(
            f"Expected (B, {self.dim}, {self.dim}) or (B, {self.dim}), "
            f"got {tuple(x.shape)}."
        )

    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}, dim={self.dim}"
