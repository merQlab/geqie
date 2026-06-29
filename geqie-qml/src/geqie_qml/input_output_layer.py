from .precompute import _compute_circuit_unitary
from types import ModuleType
from typing import Any
import os
import numpy as np
import torch
import torch.nn as nn
import importlib

from contextlib import contextmanager
from concurrent import futures
from multiprocessing import cpu_count
from torch.utils.data import Dataset
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.gradients import SPSASamplerGradient
from qiskit_machine_learning.neural_networks import SamplerQNN

import geqie
from .ansatze import default_vqc_ansatz

def _normalize_encoding_name(geqie_encoding: str) -> str:
    """Convert a string-or-module selector into a stable encoding key."""
    if not isinstance(geqie_encoding, str):
        raise TypeError(f"geqie_encoding must be a string got {type(geqie_encoding).__name__}.")        
    
    return geqie_encoding.lower()


def _import_encoding_module(encoding_name: str):
    """Resolve a stable encoding key to the corresponding GEQIE module."""
    normalized_name = _normalize_encoding_name(encoding_name)
    return importlib.import_module(f"geqie.encodings.{normalized_name}")


def _build_qnn_for_feature_map(
	feature_map_image: np.ndarray,
	geqie_encoding: str,
	ansatz_factory: callable,
	output_qubits: int,
	shots: int,
	num_layers: int,
	encoding_params: dict | None,
) -> SamplerQNN:
	"""Build a Qiskit SamplerQNN for one encoded feature map."""
	encoding_module = _import_encoding_module(_normalize_encoding_name(geqie_encoding))
	geqie_circuit = geqie.encode(
		encoding_module.init_function,
		encoding_module.data_function,
		encoding_module.map_function,
		image=np.array(feature_map_image, dtype=np.float32),
		perform_measurement=False,
		encoding_params=encoding_params or {},
	)
	num_qubits = geqie_circuit.num_qubits

	ansatz_params = {'num_layers': num_layers, 'output_qubits': output_qubits}
	vqc = ansatz_factory(num_qubits, **ansatz_params)
	qc = QuantumCircuit(num_qubits)
	qc.append(UnitaryGate(Operator.from_circuit(geqie_circuit).to_matrix()), range(num_qubits))
	qc.compose(vqc, inplace=True)
	qc.measure_all()

	sampler = Sampler(default_shots=shots)
	return SamplerQNN(
		circuit=qc,
		input_params=None,
		weight_params=list(qc.parameters),
		sampler=sampler,
		gradient=SPSASamplerGradient(sampler=sampler),
	)


def _evaluate_feature_map_probabilities(
	feature_map_image: np.ndarray,
	weights: np.ndarray,
	geqie_encoding: str,
	ansatz_factory: callable,
	output_qubits: int,
	shots: int,
	num_layers: int,
	encoding_params: dict | None,
) -> np.ndarray:
	"""Evaluate one feature map and return the output probability vector."""
	qnn = _build_qnn_for_feature_map(
		feature_map_image,
		geqie_encoding,
		ansatz_factory,
		output_qubits,
		shots,
		num_layers,
		encoding_params,
	)
	return np.asarray(
		qnn.forward(input_data=None, weights=weights),
		dtype=np.float32,
	).reshape(-1)


def _evaluate_feature_map_weight_gradient(
	feature_map_image: np.ndarray,
	weights: np.ndarray,
	geqie_encoding: str,
	ansatz_factory: callable,
	output_qubits: int,
	shots: int,
	num_layers: int,
	encoding_params: dict | None,
) -> np.ndarray:
	"""Evaluate dp/dtheta for one feature map."""
	qnn = _build_qnn_for_feature_map(
		feature_map_image,
		geqie_encoding,
		ansatz_factory,
		output_qubits,
		shots,
		num_layers,
		encoding_params,
	)
	_, weight_grad = qnn.backward(input_data=None, weights=weights)
	weight_grad = np.asarray(weight_grad, dtype=np.float32)
	if weight_grad.ndim == 3:
		weight_grad = weight_grad.squeeze(axis=0)
	return weight_grad


class GradientFunction_for_CNN_feature_maps(torch.autograd.Function):
	"""
	This is a custom autograd that allows us to evaluate gradients of QuantumLayer in respect to other PyTorch layers.
	"""

	@staticmethod
	def forward(ctx,
			 feature_maps: torch.Tensor,
			 geqie_encoding: str = "frqi", 
			 ansatz_factory: callable = default_vqc_ansatz,
			 output_qubits: int = None,
			 shots: int = 1024,
			 num_layers: int = 1,
			 vqc_weights: torch.Tensor = None,
			 encoding_params: dict = None,
			 finite_difference_epsilon: float = 1e-3) -> torch.Tensor:
		"""
		This is the forward pass of the autograd function. It takes as input the quantum weights and the input matrices, and returns the output probabilities.

		Remark about shapes of the tensors:
		- feature_maps is (batch_size, feature_maps, height, width)
		- quantum_circuit_outputs_probabilities is (batch_size, feature_maps, 2**output_qubits)
		- vqc_weights is (feature_maps, num_params)
		"""
		if output_qubits is None:
			raise ValueError("output_qubits must be set before evaluating the quantum layer.")
		if vqc_weights is None:
			raise ValueError("vqc_weights must be provided before evaluating the quantum layer.")

		feature_maps_for_numpy = feature_maps.detach().clone().cpu().numpy().astype(np.float32, copy=False)
		vqc_weights_np = vqc_weights.detach().clone().cpu().numpy()
		feature_maps_dtype = feature_maps.dtype
		feature_maps_device = feature_maps.device
		batch_size, feature_map_count, height, width = feature_maps_for_numpy.shape # (batch_size, feature_maps, height, width)
		# vqc_weights.shape = (feature_maps, num_params) obviously we have the same weights' set for all samples in the batch, but different weights for each feature map.

		quantum_circuit_outputs_probabilities = np.empty((batch_size, feature_map_count, 2**output_qubits), dtype=np.float32) # (batch_size, feature_maps_size, 2**output_qubits)

		for j in range(batch_size): # iterate over the batch dimension
			for k in range(feature_map_count): # iterate over the feature maps dimension
				feature_map_image = feature_maps_for_numpy[j, k] # we treat feature maps as images
				output_probs = _evaluate_feature_map_probabilities(
					feature_map_image,
					vqc_weights_np[k],
					geqie_encoding,
					ansatz_factory,
					output_qubits,
					shots,
					num_layers,
					encoding_params,
				) # (2**output_qubits,)
				quantum_circuit_outputs_probabilities[j, k] = output_probs # (2**output_qubits,)


		ctx.save_for_backward(vqc_weights)
		ctx.feature_maps_for_numpy = feature_maps_for_numpy
		ctx.feature_maps_dtype = feature_maps_dtype
		ctx.feature_maps_device = feature_maps_device
		ctx.geqie_encoding = geqie_encoding
		ctx.ansatz_factory = ansatz_factory
		ctx.output_qubits = output_qubits
		ctx.shots = shots
		ctx.num_layers = num_layers
		ctx.encoding_params = encoding_params
		ctx.finite_difference_epsilon = finite_difference_epsilon
				
		return torch.tensor(
			quantum_circuit_outputs_probabilities,
			dtype=vqc_weights.dtype,
			device=vqc_weights.device,
		) # (batch_size, feature_maps, 2**output_qubits)

	@staticmethod
	def backward(ctx, upstream_gradient: torch.Tensor):
		"""
		This is the backward pass of the autograd function. It takes as input the gradient of the output probabilities, and returns the gradient of the quantum weights and the input matrices.


		Remark about shapes of the tensors:
		- upstream_gradient is dL/dp
		- feature_maps_for_numpy is (batch_size, feature_maps, height, width)
		- vqc_weights is (feature_maps, num_params) 
		- quantum_circuit_gradients is dp/dtheta, and has shape (batch_size, feature_maps, 2**output_qubits, num_params)


		Remark about derivatives (order of backpropagation):
		- dL/dp - upstream_gradient is the gradient of the loss with respect to the output probabilities;
		- dp/dtheta - local gradient, evaluated by SamplerQNN - this is the gradient of the output probabilities with respect to the quantum weights;
		- dp/dx - gradient of the output probabilities with respect to the input pixels from the feature maps;
		- dL/dx - final gradient of the loss with respect to the input pixels from the feature maps - this is the gradient that we want to return to the CNN layer, so that it can update its weights accordingly.
		
		- Equations system: 
			- dL/dtheta = dL/dp * dp/dtheta
			- dL/dx = dL/dp * dp/dx
			- dtheta/dx = 0

		Remark about the shapes of the gradients:
		- dL/dp - (batch_size, feature_maps, 2**output_qubits)
		"""
		
		vqc_weights, = ctx.saved_tensors
		vqc_weights_np = vqc_weights.detach().clone().cpu().numpy()
		feature_maps_for_numpy: np.ndarray = ctx.feature_maps_for_numpy
		geqie_encoding = ctx.geqie_encoding
		ansatz_factory = ctx.ansatz_factory
		output_qubits = ctx.output_qubits
		shots = ctx.shots
		num_layers = ctx.num_layers
		encoding_params = ctx.encoding_params
		finite_difference_epsilon = float(ctx.finite_difference_epsilon)
		if finite_difference_epsilon <= 0:
			raise ValueError("finite_difference_epsilon must be positive.")
		num_params = len(vqc_weights_np[0]) # number of parameters in the ansatz


		batch_size, feature_map_count, height, width = feature_maps_for_numpy.shape
		upstream_gradient_np = upstream_gradient.detach().clone().cpu().numpy()
		quantum_circuit_gradients = np.empty((batch_size, feature_map_count, 2**output_qubits, num_params), dtype=np.float32) # (batch_size, feature_maps, 2**output_qubits, num_params)

		"""
		1. dp/dtheta - local gradient, evaluated by SamplerQNN - this is the gradient of the output probabilities with respect to the quantum weights. We will compute this gradient for each sample in the batch and for each feature map in the sample.
		"""
		
		for j in range(batch_size): # iterate over the batch dimension
			for k in range(feature_map_count): # iterate over the feature maps dimension
				feature_map_image = feature_maps_for_numpy[j, k] # we treat feature maps as images
				single_grad = _evaluate_feature_map_weight_gradient(
					feature_map_image,
					vqc_weights_np[k],
					geqie_encoding,
					ansatz_factory,
					output_qubits,
					shots,
					num_layers,
					encoding_params,
				)
				quantum_circuit_gradients[j, k] = single_grad

		"""
		2. Evaluating dL/dtheta = dL/dp * dp/dtheta
		"""
		vqc_weights_gradient = np.einsum(
			"bko,bkop->kp",
			upstream_gradient_np,
			quantum_circuit_gradients,
		).astype(np.float32)

		"""
		3. Evaluating following:
		 3.1 dp/dx
		 3.2 dL/dx = dL/dp * dp/dx
		"""
		feature_maps_gradient = np.empty_like(feature_maps_for_numpy, dtype=np.float32)
		for j in range(batch_size):
			for k in range(feature_map_count):
				feature_map_image = np.array(feature_maps_for_numpy[j, k], dtype=np.float32, copy=True)
				for h in range(height):
					for w in range(width):
						feature_map_plus = feature_map_image.copy()
						feature_map_minus = feature_map_image.copy()
						feature_map_plus[h, w] += finite_difference_epsilon
						feature_map_minus[h, w] -= finite_difference_epsilon

						probabilities_plus = _evaluate_feature_map_probabilities(
							feature_map_plus,
							vqc_weights_np[k],
							geqie_encoding,
							ansatz_factory,
							output_qubits,
							shots,
							num_layers,
							encoding_params,
						)
						probabilities_minus = _evaluate_feature_map_probabilities(
							feature_map_minus,
							vqc_weights_np[k],
							geqie_encoding,
							ansatz_factory,
							output_qubits,
							shots,
							num_layers,
							encoding_params,
						)
						dp_dx = (probabilities_plus - probabilities_minus) / (2.0 * finite_difference_epsilon)
						feature_maps_gradient[j, k, h, w] = np.dot(upstream_gradient_np[j, k], dp_dx)

		"""
		return gradients (with respect to the correct positions): dL/dtheta and dL/dx
		"""
		return (
			torch.tensor(
				feature_maps_gradient,
				dtype=ctx.feature_maps_dtype,
				device=ctx.feature_maps_device,
			),  # feature_maps
			None,  # geqie_encoding
			None,  # ansatz_factory
			None,  # output_qubits
			None,  # shots
			None,  # num_layers
			torch.tensor(
				vqc_weights_gradient,
				dtype=vqc_weights.dtype,
				device=vqc_weights.device,
			),  # vqc_weights
			None,  # encoding_params
			None,  # finite_difference_epsilon
		)


class VQCLayerForCNNFeatureMaps(nn.Module):
	"""
	Variational Quantum Circuit layer, usable as a standard PyTorch nn.Module.

	Accepts a batch of feature maps, that contains unitary matrices and returns a batch of
	probability vectors over all 2**num_qubits basis states.  Classical
	post-processing (linear head, activation, loss) is left to the caller,
	so this layer composes freely inside any nn.Sequential or custom Module.

	Both the sequential (default) and parallel paths are driven by the same
	per-sample worker functions — ``_worker_forward_eval`` for the forward
	pass and ``_worker_grad_eval`` for the parameter-shift backward pass.
	The only difference is whether those functions are called in a plain loop
	or dispatched to a ``ProcessPoolExecutor`` via ``parallel_context()``.

	Parameters
	----------
	num_qubits : int
		Number of qubits.  Input matrices must be square with side 2**num_qubits.
	num_layers : int
		Number of brickwork VQC layers (each with Rx/Ry/Rz + entanglement).
	shots : int
		Number of measurement shots per circuit evaluation.
	scale_output : bool
		When True (default), multiply output probabilities by 2**num_qubits.
		This rescales the near-zero probability values into a more numerically
		convenient range before they are passed to a classical head.

	Usage
	-----
	Recomended model for (16, 16):

		cnn_layer = nn.Sequential()
		cnn_layer.add_module("conv1", nn.Conv2d(1, 8, kernel_size=3, padding=1))
		cnn_layer.add_module("batchnorm1", nn.BatchNorm2d(8))
		cnn_layer.add_module("relu1", nn.ReLU())
		cnn_layer.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))

		cnn_layer.add_module("conv2", nn.Conv2d(8, 16, kernel_size=3, padding=1))
		cnn_layer.add_module("batchnorm2", nn.BatchNorm2d(16))
		cnn_layer.add_module("relu2", nn.ReLU())
		cnn_layer.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))

		model = nn.Sequential(
			cnn_layer,
			VQCLayerForCNNFeatureMaps(num_qubits=9, num_layers=1),
			nn.Linear(2**num_qubits, num_classes),
			nn.LogSoftmax(dim=-1),
		)
	"""

	def __init__(
		self,
		num_qubits: int = 0,
		num_layers: int = 3,
		shots: int = 1024,
		scale_output: bool = True,
		ansatz_factory=None,
		output_qubits: int = None,
		batch_size: int = -1,
		feature_maps: int = -1,
		finite_difference_epsilon: float = 1e-3,
		encoding_params: dict | None = None,
	):
		super().__init__()
		self.num_qubits: int = num_qubits
		self.num_layers: int = num_layers
		self.num_shots: int = shots
		self.scale_output: bool = scale_output
		self.ansatz_factory: callable = ansatz_factory or default_vqc_ansatz
		self.output_qubits: int = output_qubits if output_qubits is not None else num_qubits
		self.batch_size: int = batch_size
		self.feature_maps: int = feature_maps # number of input feature maps per original sample, after passing throught CNN. E. g. (16, 4, 4) -> means 16 feature maps - images.
		self.finite_difference_epsilon: float = finite_difference_epsilon
		self.encoding_params: dict | None = encoding_params

		# Trainable quantum weights, registered as a proper nn.Parameter so
		# that optimisers, state_dict, and requires_grad all work out of the box.
		# Workers reconstruct the circuit independently via _build_vqc_circuit,
		# so the Qiskit QuantumCircuit object does not need to be stored here.
		params = {'num_layers': num_layers, 'output_qubits': self.output_qubits}
		
		# create a dummy circuit to determine the number of parameters in the ansatz:
		probe_circuit = self.ansatz_factory(num_qubits, **params)
		num_params = len(probe_circuit.parameters)

		# PROBLEM: wag musi być feature_maps X num_params, bo każdy feature map ma swoje własne parametry wagowe.
		self.vqc_weights = nn.Parameter(
			torch.empty(feature_maps, num_params).uniform_(-np.pi, np.pi)
		)

		# Executor is None by default; populated only inside parallel_context().
		# self._executor: futures.ProcessPoolExecutor | None = None


	def forward(self, batched_feature_maps_original: torch.Tensor, **kwargs) -> torch.Tensor:
		"""
		Now here comes the major difference from the original VQCLayer: 
		the input is expected to have an extra dimension for feature maps, like:
		(batch_size, feature_maps, H, W)
		where:
		- batch_size is the number of samples in the batch,
		- feature_maps is the number of feature maps per sample (after passing through CNN),
		- H, W are the height and width of the feature maps (must be square, i.e., H=W=2**n for some n).

		Parameters
		----------
		batched_feature_maps_original : torch.Tensor, shape (batch_size, feature_maps, H, W), complex
			Batch of pre-encoded image unitary matrices.

		Returns
		-------
		torch.Tensor, shape (batch_size, 2**num_qubits), float
			Probability distribution over basis states for each sample.
			If ``scale_output=True``, values are multiplied by 2**num_qubits.
		"""
		geqie_encoding = kwargs.get("geqie_encoding", "frqi")
		encoding_params = kwargs.get("encoding_params", self.encoding_params)
		finite_difference_epsilon = kwargs.get("finite_difference_epsilon", self.finite_difference_epsilon)

		quantum_output = GradientFunction_for_CNN_feature_maps.apply(
			batched_feature_maps_original,
			geqie_encoding,
			self.ansatz_factory,
			self.output_qubits,
			self.num_shots,
			self.num_layers,
			self.vqc_weights,
			encoding_params,
			finite_difference_epsilon,
		)
		
		if self.scale_output:
			quantum_output = quantum_output * 2 ** self.num_qubits

		return quantum_output

	def extra_repr(self) -> str:
		"""Adds layer details to the standard nn.Module string representation."""
		return (
			f"num_qubits={self.num_qubits}, num_layers={self.num_layers}, "
			f"output_qubits={self.output_qubits}, shots={self.num_shots}, scale_output={self.scale_output}, "
			f"num_params={self.vqc_weights.numel()}"
		)

__all__ = [
	"GradientFunction_for_CNN_feature_maps",
	"VQCLayerForCNNFeatureMaps",
]
	
