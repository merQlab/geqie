import numpy as np
from PIL import Image, ImageOps

import geqie
from geqie.encodings import frqi

from . import RELATIVE_TOLERANCE


def test_frqi():
    image = Image.open("assets/test_images/grayscale/test_image_4x4.png")
    image = ImageOps.grayscale(image)
    image = np.asarray(image)
    circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)
    geqie.simulate(circuit, 1024)
    results = geqie.simulate(circuit, 1024)
    retrieved_image = frqi.retrieve_function(results)
    assert np.allclose(image, retrieved_image, rtol=RELATIVE_TOLERANCE)
