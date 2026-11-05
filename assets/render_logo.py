"""Render assets/geqie_logo.svg into a PNG using the Inkscape CLI (Inkscape >= 1.0 must be on PATH)."""

import argparse
import shutil
import subprocess
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent
DEFAULT_SVG = ASSETS_DIR / "geqie_logo.svg"
DEFAULT_WIDTH = 900


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", nargs="?", type=Path, default=DEFAULT_SVG, help="input SVG file")
    parser.add_argument("-o", "--output", type=Path, help="output PNG file (default: SVG path with .png suffix)")
    parser.add_argument("-w", "--width", type=int, default=DEFAULT_WIDTH, help="output width in pixels; height keeps aspect ratio")
    args = parser.parse_args()

    output = args.output or args.svg.with_suffix(".png")
    render_svg(args.svg, output, args.width)
    print(f"Rendered {args.svg} -> {output}")


def render_svg(svg: Path, output: Path, width: int) -> None:
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise SystemExit("Inkscape not found on PATH; install it from https://inkscape.org")

    subprocess.run(
        [
            inkscape,
            str(svg),
            "--export-type=png",
            f"--export-filename={output}",
            f"--export-width={width}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
