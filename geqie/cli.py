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

ENCODINGS_PATH = Path(__file__).parent / "encodings"


class EncodingFunctions(NamedTuple):
    init_function: Callable
    data_function: Callable
    map_function: Callable


def _import_module(name: str, file_path: str) -> types.ModuleType:
    path = ENCODINGS_PATH / file_path
    spec = importlib.util.spec_from_file_location(name, path.absolute())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _get_encoding_functions(params: Dict) -> EncodingFunctions:
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


def _get_retrive_functions(params: Dict):
    encoding_path = params.get("encoding")
    retrieve_path = f"{encoding_path}/retrieve.py"
    retrieve_function = getattr(_import_module("retrieve", retrieve_path), "retrieve")
    return retrieve_function


def _parse_image(image_path, grayscale, image_dimensionality, **_) -> np.ndarray:
    if image_dimensionality == 2:
        image = Image.open(image_path)
        if grayscale:
            return np.asarray(ImageOps.grayscale(image))
        else:
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
    @constraints.require_one(
        cloup.option("--encoding", help="Name of the encoding from 'encodings' directory"),
        cloup.option_group("Custom encoding plugins",
            cloup.option("--init"),
            cloup.option("--data"),
            cloup.option("--map"),
            constraint=constraints.If("encoding", then=constraints.accept_none, else_=constraints.require_all),
        )
    )
    @cloup.option("--image-path", required=True, help="Path to the image file")
    @cloup.option("--grayscale", type=cloup.BOOL, default=True, show_default=True, help="Indication wether the image is grayscale")
    @cloup.option("--image-dimensionality", type=int, default=2, show_default=True, help="Number of image dimensions to consider")
    @cloup.option("--verbosity-level", default=0, help="Set verbosity level, 0-3")
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
    @cloup.option("--encoding", required=True, type=str, help="Name of the encoding from 'encodings' directory")   
    @cloup.option("--result", required=True, type=str, help="Result from simulation")   
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@encoding_options
def encode(**params) -> qiskit.QuantumCircuit:
    image = _parse_image(**params)
    e = _get_encoding_functions(params)
    return main.encode(e.init_function, e.data_function, e.map_function, image, **params)


@cli.command()
@encoding_options
@simulate_options
@cloup.pass_context
def simulate(ctx: cloup.Context, **params):
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
    circuit = ctx.invoke(encode, **params)
    print(json.dumps(main.execute(circuit, **params)))


@cli.command()
@retrieve_options
def retrieve(**params):
    print('Retrieve CLI')
    # print(f'Params: {params}')
    print(f'Params.get("result"): {params.get("result")}')

    retrieve_fun = _get_retrive_functions(params)
    # print(f'e: {e}')
    print(retrieve_fun(params.get("result")))
    # return retrieve_fun, params


if __name__ == '__main__':
    cli()

