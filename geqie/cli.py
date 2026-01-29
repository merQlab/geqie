import functools
import importlib
import importlib.util
import json
import sys
import types

from pathlib import Path
from typing import Callable, Dict, NamedTuple

import cloup
from cloup import constraints
from PIL import Image, ImageOps

import qiskit
import numpy as np

import geqie.main as main
from geqie.logging_utils import levels as logging_levels

ENCODINGS_PATH = Path(__file__).parent / "encodings"


def _import_encoding(encoding: str, **_) -> types.ModuleType:
    encoding_dir = ENCODINGS_PATH / encoding
    init_file = encoding_dir / "__init__.py"
    
    if not init_file.exists():
        raise ValueError(f"Encoding '{encoding}' not found at {init_file}")
    
    # Create a unique module name to avoid collisions
    module_name = f"geqie.encodings.{encoding}"
    
    spec = importlib.util.spec_from_file_location(
        module_name, 
        init_file,
        submodule_search_locations=[str(encoding_dir)]
    )
    
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to create module spec for '{encoding}'")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def _parse_image(image_path, image_dimensionality, **_) -> np.ndarray:
    if image_dimensionality == 2:
        image = Image.open(image_path)
        return np.asarray(image)
    else:
        return np.load(image_path)


@cloup.group()
def cli() -> None:
    pass


@cli.command()
def list_encodings() -> None:
    print(sorted(dir.stem for dir in ENCODINGS_PATH.glob("*")))


def encoding_options(func) -> Callable:
    @cloup.option(
        "--encoding",
        required=True,
        help="Name of the encoding from 'encodings' directory",
    )
    @cloup.option("--image-path", required=True, help="Path to the image file")
    @cloup.option("--image-dimensionality", type=int, default=2, show_default=True, help="Number of image dimensions to consider")
    @cloup.option("--verbosity-level", default="ERROR", help=f"Set verbosity level, 0-6 (higher means more verbose) or use names {logging_levels.CLI_VERBOSITY_LEVELS.values()}")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def simulate_options(func) -> Callable:
    @cloup.option("--n-shots", type=int, help="Number of simulation shots")
    @cloup.option("--return-qiskit-result", type=cloup.BOOL, default=False, show_default=True, help="Return results directly from qiskit")
    @cloup.option("--return-padded-counts", type=cloup.BOOL, default=False, show_default=True, help="Return state counts including zero-count states")
    @cloup.option("--output-path", required=False, help="Path to where the results will be written")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def execute_options(func) -> Callable:
    @cloup.option("--n-shots", type=int, help="Number of simulation shots")
    @cloup.option("--return-qiskit-result", type=cloup.BOOL, default=False, show_default=True, help="Return results directly from qiskit")
    @cloup.option("--return-padded-counts", type=cloup.BOOL, default=False, show_default=True, help="Return state counts including zero-count states")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def retrieve_options(func) -> Callable:
    @cloup.option(
        "--encoding",
        required=True,
        type=str,
        help="Name of the encoding from 'encodings' directory.",
    )
    @cloup.option("--result", required=True, type=str, help="Result from simulation")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@encoding_options
def encode(**params) -> qiskit.QuantumCircuit:
    params["logging_level"] = logging_levels.cli_verbosity_to_logging_level(params.get("verbosity_level", 0))

    image = _parse_image(**params)
    encoding_module = _import_encoding(**params)
    return main.encode(encoding_module.init_function, encoding_module.data_function, encoding_module.map_function, image, **params)


@cli.command()
@encoding_options
@simulate_options
@cloup.pass_context
def simulate(ctx: cloup.Context, **params):
    params["logging_level"] = logging_levels.cli_verbosity_to_logging_level(params.get("verbosity_level", 0))

    circuit = ctx.invoke(encode, **params)
    result = main.simulate(circuit, **params)
    print(json.dumps(result))
    if output_path := params.get("output_path"):
        Path(output_path).write_text(json.dumps(result))
        print(f"Results written to '{output_path}'")


@cli.command()
@encoding_options
@simulate_options
@execute_options
@cloup.pass_context
def execute(ctx: cloup.Context, **params):
    params["logging_level"] = logging_levels.cli_verbosity_to_logging_level(params.get("verbosity_level", 0))

    circuit = ctx.invoke(encode, **params)
    print(json.dumps(main.execute(circuit, **params)))


@cli.command()
@retrieve_options
def retrieve(**params):
    params["logging_level"] = logging_levels.cli_verbosity_to_logging_level(params.get("verbosity_level", 0))

    retrieve_function = _import_encoding(**params).retrieve_function
    print(retrieve_function(params.get("result")))


if __name__ == '__main__':
    cli()

