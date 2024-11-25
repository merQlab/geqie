# Here is minimum working formula in .py code:

import numpy as np
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

import geqie
from geqie.encodings import frqi
from geqie.encodings import neqr

image = Image.open("../assets/test_image_4x4.png")
image = ImageOps.grayscale(image)
image = np.asarray(image)
print(image)

# circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)
circuit = geqie.encode(neqr.init_function, neqr.data_function, neqr.map_function, image)
circuit.draw(output='mpl').savefig('NEQR_circ.png')

result = geqie.simulate(circuit, 1000)
print(result)