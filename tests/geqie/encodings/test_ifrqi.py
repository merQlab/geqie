import numpy as np

from geqie.encodings.ifrqi.retrieve import retrieve


def test_unmeasured_pixels_default_to_zero() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        image = retrieve({"001111": 8})

    np.testing.assert_array_equal(image, [[255, 0], [0, 0]])


def test_zero_count_pixels_default_to_zero() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        image = retrieve({"001111": 0, "110000": 8})

    np.testing.assert_array_equal(image, np.zeros((2, 2)))


def test_measured_pixels_preserve_decoding() -> None:
    image = retrieve({"000000": 8, "010011": 8, "101100": 8, "111111": 8})

    np.testing.assert_array_equal(image, [[0, 15], [240, 255]])
