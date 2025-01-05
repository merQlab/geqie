import numpy as np

from qiskit.quantum_info import Operator

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    ''' EFRQI mapping function '''
    p = image[u, v]
    t = (float)(p) / 255.0    # Normalize the pixel value to [0, 1]
    i_t = 1.0j ** t           
    q_0 = (1.0 + i_t) / 2.0   # coeff for |0> 
    q_1 = (1.0 - i_t) / 2.0   # coeff for |1>
    
    map_operator = [
        [q_0, -1 * q_1],
        [q_1,      q_0],
    ]

    return Operator(map_operator)

