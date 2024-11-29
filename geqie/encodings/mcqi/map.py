import numpy as np
from qiskit.quantum_info import Operator

identity_gate = np.array([[1, 0], [0, 1]])
x_gate = np.array([[0, 1], [1, 0]])

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    # Red channel:
    p_r = image[u, v, 0] * np.pi / (2 * 255.0)
    red_channel_Ry_gate = [
        [np.cos(p_r), -np.sin(p_r)],
        [np.sin(p_r), np.cos(p_r)]]
    red_channel_map_operator = np.kron(np.kron(identity_gate, identity_gate), red_channel_Ry_gate)

    # Green channel:
    p_g = image[u, v, 1] * np.pi / (2 * 255.0)
    green_channel_Ry_gate = [
        [np.cos(p_g), -np.sin(p_g)],
        [np.sin(p_g), np.cos(p_g)]]
    green_channel_map_operator = np.kron(np.kron(identity_gate, x_gate), green_channel_Ry_gate)

    # Blue channel:
    p_b = image[u, v, 2] * np.pi / (2 * 255.0)
    blue_channel_Ry_gate = [
        [np.cos(p_b), -np.sin(p_b)],
        [np.sin(p_b), np.cos(p_b)]]
    blue_channel_map_operator = np.kron(np.kron(x_gate, identity_gate), blue_channel_Ry_gate)

    # total:
    map_operator = red_channel_map_operator + green_channel_map_operator + blue_channel_map_operator

    return Operator(map_operator)
