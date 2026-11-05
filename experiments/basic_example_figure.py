"""Figure for the basic usage example: FRQI circuit and original vs retrieved 4x4 image."""

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.io as pio
from PIL import Image, ImageOps
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from scipy.stats import pearsonr

import geqie
from geqie.encodings import frqi

IMAGE_PATH = Path(__file__).parent / "generated_images" / "grayscale" / "image_008_4x4_.png"
PLOTS_PATH = Path(__file__).parent / "results" / "plots"
N_SHOTS = 4096

pio.templates["latex"] = pio.templates["plotly_white"].update(layout=dict(font=dict(family="CMU Serif", size=14)))
pio.templates.default = "latex"


def draw_circuit(circuit: QuantumCircuit):
    """Draw the encoding circuit with short gate labels instead of the full list of state-preparation amplitudes."""
    labels = {"state_preparation": "init", "unitary": "U"}
    display = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in labels:
            operation = Gate(labels[operation.name], operation.num_qubits, [])
        display.append(operation, instruction.qubits, instruction.clbits)
    return display.draw(output="mpl", style="bw", fold=-1)


def main() -> None:
    image = np.asarray(ImageOps.grayscale(Image.open(IMAGE_PATH)))
    circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image)
    counts = geqie.simulate(circuit, n_shots=N_SHOTS)
    retrieved = frqi.retrieve_function(counts)

    pcc, _ = pearsonr(image.flatten().astype(float), retrieved.flatten().astype(float))
    print(f"{pcc=:.3f}")

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    draw_circuit(circuit).savefig(PLOTS_PATH / "basic_frqi_circuit.pdf", bbox_inches="tight")

    fig = px.imshow(
        np.stack([image, retrieved]),
        facet_col=0,
        color_continuous_scale="gray",
        zmin=0,
        zmax=255,
        text_auto=True,
    )
    for annotation, title in zip(fig.layout.annotations, ["Original", f"Retrieved (PCC = {pcc:.3f})"]):
        annotation.text = title
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(width=600, height=330, margin=dict(l=5, r=5, t=30, b=5), coloraxis_showscale=False)
    fig.write_image(PLOTS_PATH / "basic_frqi_images.pdf")


if __name__ == "__main__":
    main()
