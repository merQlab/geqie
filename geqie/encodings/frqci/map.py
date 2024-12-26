import numpy as np

from qiskit.quantum_info import Operator

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    ''' FRQCI mapping function '''
    red   = image[u, v, 0]
    green = image[u, v, 1]
    blue  = image[u, v, 2]
    
    #  File "/home/krzysztof/Dokumenty/github-repos/geqie/geqie/encodings/frqci/map.py", line 7, in map
    # red   = image[u, v, 0]
    #         ~~~~~^^^^^^^^^
    # IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
        
    pi = np.pi;
    red_coeff = 2**16;
    green_coeff = 2**8;
    blue_coeff = 1;
    numerator = red * red_coeff + green * green_coeff + blue * blue_coeff;
    denominator = 2**24 - 1;
    theta = pi * numerator / denominator;
    sin_theta_2 = np.sin(theta / 2);
    cos_theta_2 = np.cos(theta / 2);

    map_operator = [
        [cos_theta_2, -1 * sin_theta_2],
        [sin_theta_2,      cos_theta_2],
    ]

    return Operator(map_operator)

