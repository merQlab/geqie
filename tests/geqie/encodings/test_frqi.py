import numpy as np
from PIL import Image, ImageOps

import geqie
from geqie.encodings import frqi

RELATIVE_TOLERANCE = 0.2


def test_frqi():
    image = Image.open("assets/test_images/grayscale/test_image_4x4.png")
    image = ImageOps.grayscale(image)
    image = np.asarray(image)
    circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)
    results = geqie.simulate(circuit, 1024)
    retrieved_image = frqi.retrieve_function(results)
    assert np.allclose(image, retrieved_image, rtol=RELATIVE_TOLERANCE)



def test_frqi_symbolic_unitary():
    image = Image.open("assets/test_images/grayscale/test_image_4x4.png")
    image = ImageOps.grayscale(image)
    image = np.asarray(image)
    circuit = geqie.encode(
        frqi.init_function,
        frqi.data_function,
        frqi.map_function,
        image,
        encoding_params={"symbolic_unitary": True},
    )
    results = geqie.simulate(circuit, 1024)
    retrieved_image = frqi.retrieve_function(results)
    assert np.allclose(image, retrieved_image, rtol=RELATIVE_TOLERANCE)