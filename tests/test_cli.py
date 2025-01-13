import subprocess
import pytest

from dataclasses import dataclass

@dataclass
class CLIParameters:
    image_path: str
    grayscale: bool


METHOD_CLI_MAPPING = {
    "frqi": CLIParameters(image_path="assets/test_images/grayscale/test_image.png", grayscale=True),
    "ifrqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png", grayscale=True),
    "neqr": CLIParameters(image_path="assets/test_images/grayscale/test_image.png", grayscale=True),

    "mcqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png", grayscale=False),
    "ncqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png", grayscale=False),
    "qualpi": CLIParameters(image_path="assets/test_images/grayscale/test_flag_4x4.png", grayscale=False),
}


@pytest.mark.parametrize("method, params", METHOD_CLI_MAPPING.items())
def test_cli(method: str, params: CLIParameters):
    subprocess.run([
        "geqie", "simulate", 
        "--encoding", method, 
        "--image", params.image_path, 
        "--grayscale", str(params.grayscale)
    ], check=True)
