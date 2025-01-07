import numpy as np
import subprocess
from PIL import Image, ImageOps

import geqie
from geqie.encodings import frqi

def test_frqi():
    image = Image.open("assets/test_image_4x4.png")
    image = ImageOps.grayscale(image)
    image = np.asarray(image)
    circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)
    geqie.simulate(circuit, 1024)
