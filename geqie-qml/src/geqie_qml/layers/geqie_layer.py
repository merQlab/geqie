import importlib
import os
from types import ModuleType
from concurrent import futures
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from qiskit.quantum_info import Statevector

import geqie

class GRADIENT_MODES:
    FIRST_BATCH_AVERAGE = "first_batch_average"
    FULL = "full"
    NONE = "none"


class GEQIELayer(nn.Module):
    """
    PyTorch layer that encodes a batch of images into quantum statevectors on-the-fly.

    Applies ``geqie.encode`` to each image and returns the resulting statevector
    ``U|ψ_init⟩`` (where ``|ψ_init⟩`` is the encoding's initial state, typically ``|0⟩``).
    There are no trainable parameters; input gradients are propagated so that any
    preceding classical layers can be trained end-to-end.

    Parameters
    ----------
    geqie_encoding : str or ModuleType
        Encoding to use.  Pass a name string (e.g. ``"frqi"``) or an already-imported
        ``geqie.encodings.*`` module.
    encoding_params : dict, optional
        Extra keyword arguments forwarded verbatim to the encoding functions.
    finite_difference_epsilon : float
        Step size for central finite-difference gradient estimation.
    grad_mode : str
        How input gradients are produced in the backward pass:

        - ``"full"`` (default): exact per-image Jacobian estimated via
          central finite differences.  Correct for any encoding, but evaluates
          ``2 * batch_size * H * W`` extra circuits per backward pass.
        - ``"first_batch_average"``: the pixel Jacobian ``dψ/dx`` is computed once
          (lazily, on the first backward pass) by averaging the per-image Jacobians
          over all images in that first batch, then cached and reused for all
          subsequent batches.  No circuits are evaluated after the first backward
          pass.  Exact only when the encoding's Jacobian is input-independent;
          otherwise it is a biased approximation traded for speed.
        - ``"none"``: no input gradient is produced (returns ``None``).  Valid only
          when nothing trainable precedes this layer.
    """

    def __init__(
        self,
        geqie_encoding: str | ModuleType = "frqi",
        encoding_params: dict | None = None,
        finite_difference_epsilon: float = 1e-3,
        grad_mode: str = GRADIENT_MODES.FULL,
    ) -> None:
        super().__init__()
        self.encoding_module: ModuleType = resolve_encoding_module(geqie_encoding)
        self.encoding_name: str = self.encoding_module.__name__.split(".")[-1]
        self.encoding_params: dict = encoding_params or {}
        self.finite_difference_epsilon: float = finite_difference_epsilon
        self.grad_mode: str = validate_grad_mode(grad_mode)
        self.frozen_jacobian_cache: dict = {}



    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images into quantum statevectors.

        Parameters
        ----------
        images : torch.Tensor, shape ``(batch_size, H, W)``, real
            Batch of images to encode.  Each image is encoded independently.

        Returns
        -------
        torch.Tensor, shape ``(batch_size, 2**n_qubits)``, complex64
            Statevector ``U|ψ_init⟩`` for each image.
        """
        return GEQIEEncodeGrad.apply(
            images,
            self.encoding_name,
            self.encoding_params,
            self.finite_difference_epsilon,
            self.grad_mode,
            self.frozen_jacobian_cache,
        )

    def extra_repr(self) -> str:
        return (
            f"encoding={self.encoding_name!r}, grad_mode={self.grad_mode!r}, "
            f"finite_difference_epsilon={self.finite_difference_epsilon}"
        )


class GEQIEEncodeGrad(torch.autograd.Function):
    """
    Differentiable GEQIE encoding wrapped as a ``torch.autograd.Function``.

    Forward: builds the encoding circuit for each image and extracts its statevector.
    Backward: approximates ``dψ/d(pixel)`` via central finite differences and applies
    the chain rule ``dL/dx = Re(upstream* · dψ/dx)`` for real input, complex output.
    """

    @staticmethod
    def forward(
        ctx,
        images: torch.Tensor,
        encoding_name: str,
        encoding_params: dict,
        finite_difference_epsilon: float,
        grad_mode: str,
        frozen_cache: dict,
    ) -> torch.Tensor:
        images_np = images.detach().cpu().numpy().astype(np.float32, copy=False)

        raw = dispatch_parallel(
            worker_encode_statevector,
            [(i, images_np[i], encoding_name, encoding_params) for i in range(len(images_np))],
        )
        raw.sort(key=lambda r: r[0])
        statevectors = np.stack([sv for _, sv in raw])  # (B, D) complex64

        ctx.images_np = images_np
        ctx.input_dtype = images.dtype
        ctx.input_device = images.device
        ctx.encoding_name = encoding_name
        ctx.encoding_params = encoding_params
        ctx.finite_difference_epsilon = float(finite_difference_epsilon)
        ctx.grad_mode = grad_mode
        ctx.frozen_cache = frozen_cache

        return torch.as_tensor(statevectors, dtype=torch.complex64).to(images.device)

    @staticmethod
    def backward(ctx, upstream_grad: torch.Tensor):
        """
        Parameters
        ----------
        upstream_grad : torch.Tensor, shape ``(B, D)``, complex64
            Upstream gradient ``dL/dψ``.

        Notes
        -----
        Chain rule for real input / complex output:
        ``dL/dx_j = Re( conj(upstream) · dψ/dx_j )``
        where ``dψ/dx_j`` is approximated via central finite differences (evaluated
        per batch for ``"finite_difference"`` mode, or once and cached for ``"first_batch_average"``).
        """
        if not ctx.needs_input_grad[0] or ctx.grad_mode == GRADIENT_MODES.NONE:
            return (None, None, None, None, None, None)

        images_np: np.ndarray = ctx.images_np
        encoding_name: str = ctx.encoding_name
        encoding_params: dict = ctx.encoding_params
        eps: float = ctx.finite_difference_epsilon

        batch_size, height, width = images_np.shape
        upstream_np = upstream_grad.detach().cpu().numpy()  # (B, D) complex

        if ctx.grad_mode == GRADIENT_MODES.FIRST_BATCH_AVERAGE:
            jacobian = ctx.frozen_cache.get("jacobian")
            if jacobian is None:
                per_image = [
                    compute_pixel_jacobian(images_np[i], encoding_name, encoding_params, eps)
                    for i in range(batch_size)
                ]
                jacobian = np.mean(per_image, axis=0)  # (H, W, D)
                ctx.frozen_cache["jacobian"] = jacobian
            # jacobian: (H, W, D);  dL/dx[i,h,w] = Re( conj(upstream[i]) · dψ/dx[h,w] )
            image_grad = np.einsum("id,hwd->ihw", upstream_np.conj(), jacobian).real.astype(np.float32)
            return (
                torch.tensor(image_grad, dtype=ctx.input_dtype, device=ctx.input_device),
                None,  # encoding_name
                None,  # encoding_params
                None,  # finite_difference_epsilon
                None,  # grad_mode
                None,  # frozen_cache
            )

        # GRADIENT_MODES.FULL: exact per-image Jacobian-vector product.
        fd_jobs = [
            (
                (i, h, w, sign),
                perturb_pixel(images_np[i], h, w, sign * eps),
                encoding_name,
                encoding_params,
            )
            for i in range(batch_size)
            for h in range(height)
            for w in range(width)
            for sign in (+1, -1)
        ]

        fd: dict = {}
        for idx, sv in dispatch_parallel(worker_encode_statevector, fd_jobs):
            fd[idx] = sv

        image_grad = np.empty_like(images_np, dtype=np.float32)
        for i in range(batch_size):
            for h in range(height):
                for w in range(width):
                    dpsi_dx = (fd[(i, h, w, +1)] - fd[(i, h, w, -1)]) / (2.0 * eps)
                    image_grad[i, h, w] = float(np.real(upstream_np[i].conj() @ dpsi_dx))

        return (
            torch.tensor(image_grad, dtype=ctx.input_dtype, device=ctx.input_device),
            None,  # encoding_name
            None,  # encoding_params
            None,  # finite_difference_epsilon
            None,  # grad_mode
            None,  # frozen_cache
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_grad_mode(grad_mode: str) -> str:
    """Return *grad_mode* unchanged if valid, otherwise raise ``ValueError``."""
    valid = {v for k, v in vars(GRADIENT_MODES).items() if not k.startswith("_")}
    if grad_mode not in valid:
        raise ValueError(f"grad_mode must be one of {valid}; got {grad_mode!r}.")
    return grad_mode


def compute_pixel_jacobian(
    image: np.ndarray,
    encoding_name: str,
    encoding_params: dict,
    epsilon: float,
) -> np.ndarray:
    """
    Estimate ``dψ/dx`` for a single image via central finite differences.

    Parameters
    ----------
    image : numpy.ndarray, shape ``(H, W)``, real
    encoding_name : str
    encoding_params : dict
    epsilon : float
        Central-difference step size.

    Returns
    -------
    numpy.ndarray, complex64, shape ``(H, W, 2**n_qubits)``
        Per-pixel derivative of the encoded statevector.
    """
    height, width = image.shape
    jobs = [
        ((h, w, sign), perturb_pixel(image, h, w, sign * epsilon), encoding_name, encoding_params)
        for h in range(height)
        for w in range(width)
        for sign in (+1, -1)
    ]
    results = {idx: sv for idx, sv in dispatch_parallel(worker_encode_statevector, jobs)}

    dim = next(iter(results.values())).shape[0]
    jacobian = np.empty((height, width, dim), dtype=np.complex64)
    for h in range(height):
        for w in range(width):
            jacobian[h, w] = (results[(h, w, +1)] - results[(h, w, -1)]) / (2.0 * epsilon)
    return jacobian


def resolve_encoding_module(geqie_encoding: str | ModuleType) -> ModuleType:
    """
    Return a ``geqie.encodings.*`` module from a name string or pass it through.

    Parameters
    ----------
    geqie_encoding : str or ModuleType
        Encoding name (e.g. ``"frqi"``) or an already-imported module.

    Returns
    -------
    ModuleType

    Raises
    ------
    TypeError
        If *geqie_encoding* is neither a ``str`` nor a module.
    ValueError
        If the name string is empty.
    """
    if isinstance(geqie_encoding, ModuleType):
        return geqie_encoding
    if isinstance(geqie_encoding, str):
        name = geqie_encoding.strip().lower()
        if not name:
            raise ValueError("geqie_encoding must not be empty.")
        return importlib.import_module(f"geqie.encodings.{name}")
    raise TypeError(
        f"geqie_encoding must be a str or module; got {type(geqie_encoding).__name__}."
    )


def encode_image_to_statevector(
    image: np.ndarray,
    encoding_module: ModuleType,
    encoding_params: dict,
) -> np.ndarray:
    """
    Encode a single image and return its statevector as a complex64 array.

    Parameters
    ----------
    image : numpy.ndarray
        2-D real image array.
    encoding_module : ModuleType
        A ``geqie.encodings.*`` module with ``init_function``, ``data_function``,
        and ``map_function`` attributes.
    encoding_params : dict
        Extra keyword arguments forwarded to the encoding functions.

    Returns
    -------
    numpy.ndarray, complex64, shape ``(2**n_qubits,)``
    """
    circuit = geqie.encode(
        encoding_module.init_function,
        encoding_module.data_function,
        encoding_module.map_function,
        image=image,
        perform_measurement=False,
        encoding_params=encoding_params,
    )
    return Statevector(circuit).data.astype(np.complex64)


def worker_encode_statevector(args: tuple) -> tuple:
    """
    Top-level picklable worker for ``dispatch_parallel``.

    Parameters
    ----------
    args : tuple
        ``(index, image, encoding_name, encoding_params)``

    Returns
    -------
    tuple
        ``(index, statevector)``
    """
    index, image, encoding_name, encoding_params = args
    return index, encode_image_to_statevector(
        image, resolve_encoding_module(encoding_name), encoding_params
    )


def dispatch_parallel(worker_fn: callable, args_list: list) -> list:
    """
    Run ``worker_fn`` for every entry in *args_list* via a ``ProcessPoolExecutor``.

    A single-item list is handled in-process to skip spawn overhead.

    Parameters
    ----------
    worker_fn : callable
        Picklable top-level function accepting a single args tuple.
    args_list : list
        Argument tuples passed to *worker_fn*.

    Returns
    -------
    list
        Results in completion order (not necessarily submission order).
    """
    if not args_list:
        return []
    if len(args_list) == 1:
        return [worker_fn(args_list[0])]

    n = len(args_list)
    platform_cap = 61 if os.name == "nt" else n
    n_workers = min(n, platform_cap, max(1, cpu_count() - 1))

    mp_ctx = __import__("multiprocessing").get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
        pending = [executor.submit(worker_fn, args) for args in args_list]
        return [job.result() for job in pending]


def perturb_pixel(image: np.ndarray, h: int, w: int, delta: float) -> np.ndarray:
    """Return a copy of *image* with pixel ``(h, w)`` shifted by *delta*."""
    img = image.copy()
    img[h, w] += delta
    return img



