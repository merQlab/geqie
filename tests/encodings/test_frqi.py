import geqie
from geqie.encodings import frqi

def test_frqi():
    image = Image.open("../assets/test_image_4x4.png")
    image = ImageOps.grayscale(image)
    image = np.asarray(image) / 255.0
    image