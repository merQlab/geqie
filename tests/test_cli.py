import subprocess
import pytest

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CLIParameters:
    image_path: str
    grayscale: bool
    image_dimensionality: Optional[int] = None

    def to_list(self) -> List[str]:
        result = []
        for attr, value in self.__dict__.items():
            if value is not None:
                result.append(f"--{attr.replace('_', '-')}")
                result.append(str(value))
        return result


METHOD_CLI_MAPPING = {
    "frqi": CLIParameters(image_path="assets/test_images/grayscale/test_image.png", grayscale=True),
    "ifrqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png", grayscale=True),
    "neqr": CLIParameters(image_path="assets/test_images/grayscale/test_image.png", grayscale=True),

    "frqci": CLIParameters(image_path="assets/test_images/rgb/rgb.png", grayscale=False),
    "mcqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png", grayscale=False),
    "ncqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png", grayscale=False),
    "qualpi": CLIParameters(image_path="assets/test_images/grayscale/test_flag_4x4.png", grayscale=False),

    "mfrqi": CLIParameters(image_path="assets/test_images/3d/image_0_2x2x2.npy", grayscale=True, image_dimensionality=3),
}


@pytest.mark.parametrize("method, params", METHOD_CLI_MAPPING.items())
def test_cli(method: str, params: CLIParameters):
    subprocess.run([
        "geqie", "simulate", 
        "--encoding", method, 
        *params.to_list(),
    ], check=True)
