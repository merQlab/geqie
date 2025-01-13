import functools
import importlib
import importlib.util
import json
import sys

from pathlib import Path
from typing import Callable, Dict, NamedTuple

import cloup
from cloup import constraints
from PIL import Image, ImageOps

import numpy as np

import geqie.main as main

ENCODINGS_PATH = Path(__file__).parent / "encodings"


class EncodingFunctions(NamedTuple):
    init_function: Callable
    data_function: Callable
    map_function: Callable


def _import_module(name: str, file_path: str):
    path = ENCODINGS_PATH / file_path
    spec = importlib.util.spec_from_file_location(name, path.absolute())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _get_encoding_functions(params: Dict):
    if params.get("init") and params.get("data") and params.get("map"):
        init_path = params.get("init")
        data_path = params.get("data")
        map_path = params.get("map")
    else:
        encoding_path = params.get("encoding")
        
        init_path = f"{encoding_path}/init.py"
        data_path = f"{encoding_path}/data.py"
        map_path = f"{encoding_path}/map.py"

    init_function = getattr(_import_module("init", init_path), "init")
    data_function = getattr(_import_module("data", data_path), "data")
    map_function = getattr(_import_module("map", map_path), "map")

    return EncodingFunctions(init_function, data_function, map_function)


def _parse_image(params):
    image = Image.open(params.get("image"))
    if params.get("grayscale"):
        return np.asarray(ImageOps.grayscale(image))
    else:
        return np.asarray(image)


@cloup.group()
def cli():
    pass


@cli.command()
def list_encodings():
    print(sorted(dir.stem for dir in ENCODINGS_PATH.glob("*")))


def encoding_options(func):
    @constraints.require_one(
        cloup.option("--encoding", help="Name of the encoding from 'encodings' directory"),
        cloup.option_group("Custom encoding plugins",
            cloup.option("--init"),
            cloup.option("--data"),
            cloup.option("--map"),
            constraint=constraints.If("encoding", then=constraints.accept_none, else_=constraints.require_all),
        )
    )
    @cloup.option("--image", required=True, help="Path to the image file")
    @cloup.option("--grayscale", type=cloup.BOOL, default=True, show_default=True, help="Indication wether the image is grayscale")
    @cloup.option("-v", "--verbose", count=True, default=0, help="Increase verbosity (can be used multiple times, up to '-vvv')")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@encoding_options
def encode(**params):
    image = _parse_image(params)
    e = _get_encoding_functions(params)
    return main.encode(e.init_function, e.data_function, e.map_function, image, params)


def simulate_options(func):
    @cloup.option("--n-shots", type=int, help="Number of simulation shots")
    @cloup.option("--return-qiskit-result", type=cloup.BOOL, default=False, show_default=True, help="Return results directly from qiskit")
    @cloup.option("--return-padded-counts", type=cloup.BOOL, default=False, show_default=True, help="Return state counts including zero-count states")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@encoding_options
@simulate_options
@cloup.pass_context
def simulate(ctx: cloup.Context, **params):
    circuit = ctx.invoke(encode, **params)
    cli_params = {name: value for name, value in params.items() if name in ["n_shots", "return_qiskit_result", "return_padded_counts"]}
    print(json.dumps(main.simulate(circuit, **cli_params)))


if __name__ == '__main__':
    cli()

