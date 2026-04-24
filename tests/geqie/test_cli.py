import subprocess
import pytest

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CLIParameters:
    image_path: str
    image_dimensionality: Optional[int] = None
    encoding_params: Optional[List[str]] = None

    def to_list(self) -> List[str]:
        result = []
        for attr, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, list):
                    for item in value:
                        result.append(f"--{attr.replace('_', '-')}")
                        result.append(str(item))
                else:
                    result.append(f"--{attr.replace('_', '-')}")
                    result.append(str(value))
        return result


METHOD_CLI_MAPPING = {
    "frqi": CLIParameters(image_path="assets/test_images/grayscale/test_image.png"),
    "ifrqi": CLIParameters(image_path="assets/test_images/grayscale/test_image.png"),
    "neqr": CLIParameters(image_path="assets/test_images/grayscale/test_image.png", encoding_params=["bitrate=4"]),
    "qualpi": CLIParameters(image_path="assets/test_images/grayscale/test_image.png"),

    "frqci": CLIParameters(image_path="assets/test_images/rgb/rgb.png"),
    "mcqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png"),
    "ncqi": CLIParameters(image_path="assets/test_images/rgb/rgb.png"),

    "mfrqi": CLIParameters(image_path="assets/test_images/3d/image_0_2x2x2.npy", image_dimensionality=3),
}


@pytest.mark.parametrize("method, params", METHOD_CLI_MAPPING.items())
def test_cli(method: str, params: CLIParameters):
    subprocess.run([
        "geqie", "simulate", 
        "--encoding", method, 
        *params.to_list(),
    ], check=True)
