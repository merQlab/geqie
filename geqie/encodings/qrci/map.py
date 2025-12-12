import numpy as np
from qiskit.quantum_info import Operator


KET_BRA_0 = np.array([[1, 0], [0, 0]])
KET_BRA_1 = np.array([[0, 0], [0, 1]])
I_GATE = np.eye(2)
X_GATE = np.array([
    [0, 1], 
    [1, 0]
])

# Precompute label operators to avoid recalculating them in the loop
LABEL_OPERATORS = [
    np.kron(k1, np.kron(k2, k3))
    for k1 in (KET_BRA_0, KET_BRA_1)
    for k2 in (KET_BRA_0, KET_BRA_1)
    for k3 in (KET_BRA_0, KET_BRA_1)
]

def get_channel_gate(bit_value: str) -> np.ndarray:
    """Returns the appropriate gate for the given bit value."""
    return X_GATE if bit_value == '1' else I_GATE

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    u_v_pixel_operator = np.zeros((2**6, 2**6))

    R_bits = bin(image[u, v, 0])[2:].zfill(8)
    G_bits = bin(image[u, v, 1])[2:].zfill(8)
    B_bits = bin(image[u, v, 2])[2:].zfill(8)

    for bit in range(8):
        label_operator = LABEL_OPERATORS[bit]

        # Get gates for each channel based on the bit value
        R_channel_gate = get_channel_gate(R_bits[bit])
        G_channel_gate = get_channel_gate(G_bits[bit])
        B_channel_gate = get_channel_gate(B_bits[bit])

        # Combine the channel gates
        RGB_operator = np.kron(R_channel_gate, np.kron(G_channel_gate, B_channel_gate))

        # Update the pixel operator
        u_v_pixel_operator += np.kron(label_operator, RGB_operator)

    return Operator(u_v_pixel_operator)
