import cloup
from cloup import constraints
from PIL import Image, ImageOps

import numpy as np

import main


@cloup.group()
def cli():
    pass


@cli.command()
def list_encodings():
    print(main.get_all_encodings())


@cli.command()
@constraints.require_one(
    cloup.option("--encoding"),
    cloup.option_group(
        "Custom data and map",
        cloup.option("--data"),
        cloup.option("--map"),
        constraint=constraints.If("encoding", then=constraints.accept_none, else_=constraints.require_all),
    )
)
@cloup.option("--image", required=True)
@cloup.option("--grayscale", type=cloup.BOOL, default=True, show_default=True)
@cloup.option("--n-shots", type=int)
def encode(**kwargs):
    if kwargs.get("data") and kwargs.get("map"):
        data_path, map_path = kwargs.get("data"), kwargs.get("map")
    else:
        encoding_path = kwargs.get("encoding")

        data_path = f"{encoding_path}/data.py"
        map_path = f"{encoding_path}/map.py"

    data_function = getattr(main.import_module("data", data_path), "data")
    map_function = getattr(main.import_module("map", map_path), "map")

    image = Image.open(kwargs.get("image"))
    if kwargs.get("grayscale"):
        image = np.asarray(ImageOps.grayscale(image))
    else:
        image = np.asarray(image)

    image = image / 255.0

    circuit = main.encode(data_function, map_function, image)

    if (kwargs.get("n_shots") or 0) > 0:
        result = main.simulate(circuit, kwargs.get("n_shots"))
        print(result.get_counts(circuit))


if __name__ == '__main__':
    cli()

