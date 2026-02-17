from datasets import load_dataset
import os
import numpy as np
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Linear, CrossEntropyLoss, MSELoss, NLLLoss
from torch.optim import LBFGS, Adam
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
import geqie
from geqie.encodings import frqi
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import CircuitInstruction
import tqdm

	
class Simple_QNN:
	def __init__(self):
		'''
		Initialize the Simple_QNN class.
		This class takes as an input a quantum circuit with pre-encoded image from GEQIE.
		'''
		self.image: np.ndarray = None
		self.circuit: QuantumCircuit = None
		

	def encode_image_using_GEQIE(self, image:np.ndarray):
		'''Encode the input image using GEQIE encoding.'''
		self.image = image
		self.circuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, image, perform_measurement=False)

	def append_CNOT_cascade(self):
		'''Append a cascade (stairs) of CNOT gates to the quantum circuit.'''
		num_qubits = self.circuit.num_qubits
		for i in range(num_qubits - 1):
			self.circuit.cx(i, i + 1)

		self.circuit.cx(self.circuit.num_qubits - 1, 0)

	def append_VQC_layer(self, thetas: ParameterVector):
		'''Append a variational quantum circuit (VQC) layer to the quantum circuit.'''
		num_qubits = self.circuit.num_qubits
		if len(thetas) != 3 * num_qubits:
			raise ValueError(f"Expected {3 * num_qubits} parameters, but got {len(thetas)}.")
		
		# First layer - RX gates:
		for i in range(num_qubits):
			self.circuit.rx(thetas[i], i)
		# Second layer - RY gates:
		for i in range(num_qubits):
			self.circuit.ry(thetas[i + num_qubits], i)
		# Third layer - RZ gates:
		for i in range(num_qubits):
			self.circuit.rz(thetas[i + 2 * num_qubits], i)
	

class QNN_Pythorch_Module(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.num_qubits = None
		self.num_classes = num_classes
		self.simple_qcnn: Simple_QNN = None
		self.thetas: ParameterVector = None
		self.simple_qnn_as_sampler_qnn: SamplerQNN = None
		self.layer_TorchConnector: TorchConnector = None
		self.is_model_initialized = False

	def build_new_circuit(self, image: np.ndarray):
		'''Update the quantum circuit with the new image encoding.'''
		simple_qcnn = Simple_QNN()
		# GEQIE endcoding, and simple QNN:
		simple_qcnn.encode_image_using_GEQIE(image=image)
		self.num_qubits = simple_qcnn.circuit.num_qubits  # Update num_qubits based on the new circuit
		simple_qcnn.append_CNOT_cascade()
		thetas = ParameterVector("theta", length=3 * self.num_qubits)
		simple_qcnn.append_VQC_layer(thetas=thetas)
		self.thetas = thetas
		self.simple_qcnn = simple_qcnn
	
	def interpret_multiclass(self, outcome: int) -> int:
		return outcome % self.num_classes


	def SimpleQNN_to_SamplerQNN(self):
		'''Convert the quantum circuit to a SamplerQNN.'''
		if self.thetas is None:
			raise ValueError("VQC parameters have not been set. Please build the circuit before calling conversion to SamplerQNN.")
		self.simple_qnn_as_sampler_qnn = SamplerQNN(
			circuit=self.simple_qcnn.circuit,
			input_params=None,
			weight_params=self.thetas,
			interpret=self.interpret_multiclass,
			output_shape=self.num_classes,
			sampler=Sampler(),
		)

	def initialize(self, image: np.ndarray):
		self.build_new_circuit(image=image)
		self.SimpleQNN_to_SamplerQNN()
		self.SamplerQNN_to_TorchConnector()

	def update_geqie_unitary_matrix_inplace_in_Sampler(self, new_image: np.ndarray):
		'''Update the GEQIE unitary matrix in-place with the new image encoding.'''
		new_circuit:QuantumCircuit = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, new_image, perform_measurement=False)

		# More robust: find instruction by name/type
		for idx, instruction in enumerate(self.simple_qnn_as_sampler_qnn.circuit.data):
			if instruction.operation.name == 'Unitary':  # Adjust based on actual operation name
				geqie_unitary_matrix, qargs, cargs = new_circuit.data[1]
				self.simple_qnn_as_sampler_qnn.circuit.data[idx] = CircuitInstruction(
					geqie_unitary_matrix, qargs, cargs
				)
				break

		# geqie_unitary_matrix, qargs, cargs = new_circuit.data[geqie_unitary_matrix_position] 
		# self.simple_qnn_as_sampler_qnn.circuit.data[geqie_unitary_matrix_position] = CircuitInstruction(geqie_unitary_matrix, qargs, cargs)
		
	def SamplerQNN_to_TorchConnector(self): 
		'''Convert the SamplerQNN to a TorchConnector for integration with PyTorch.'''
		self.layer_TorchConnector = TorchConnector(self.simple_qnn_as_sampler_qnn)
	
	def forward(self, x):
		"""
		This is not a standard forward method. Its input is an image encoded using GEQIE, and placed in cricuit. So, the "forward" pass means just running the circuit with ansazt parameters and evaluating it.
		"""
		self.update_geqie_unitary_matrix_inplace_in_Sampler(new_image=x)
		output = self.layer_TorchConnector()
		return torch.log_softmax(output, dim=-1)

def load_and_process_mnist_dataset(labels_to_include=[0, 1], n_samples_per_label=100, resize=(8, 8), path_to_mnist_dataset=None):
	mnist_dataset = load_dataset("parquet", data_files=path_to_mnist_dataset)

	selected_images = []
	for image_idx in range(len(mnist_dataset["train"])):
		# First, select only images with labels in labels_to_include:
		
		if mnist_dataset["train"][image_idx]["label"] in labels_to_include:
			img_8x8 = mnist_dataset["train"][image_idx]["image"].resize(resize, resample=Image.BILINEAR)
			img = np.array(img_8x8)
			selected_images.append({"image": img, "label": mnist_dataset["train"][image_idx]["label"]})

	X = np.array([item["image"] for item in selected_images])
	y = np.array([item["label"] for item in selected_images])

	return X, y

def train_QCNN(epochs=10, X=None, y=None, num_classes=2):
	f_loss = CrossEntropyLoss()
	qnn_model = QNN_Pythorch_Module(num_classes=num_classes)  # Example values for num_qubits and num_classes
	qnn_model.initialize(image=X[0])  # Build the circuit with the first image to initialize parameters
	optimizer = Adam(qnn_model.parameters(), lr=0.001)
	qnn_model.train()  # Set the model to training mode
	loss_list = []
	
	progress_bar = tqdm(range(epochs), desc="Training")
	for epoch in range(epochs):
		total_loss = []
		for image, label in zip(X, y):			
			# Wyzeruj gradient:
			optimizer.zero_grad(set_to_none=True)
			output = qnn_model(x=image)  # Forward pass
			label_tensor = torch.tensor(label, dtype=torch.long)  # Scalar tensor for unbatched loss
			loss = f_loss(output, label_tensor)  # Calculate loss
			loss.backward()  # Backward pass
			optimizer.step()  # Optimize weights			
			
			total_loss.append(loss.item())  # Store loss
		epoch_loss = sum(total_loss) / len(total_loss)
		loss_list.append(epoch_loss)

		progress_bar.update(1)
		progress_bar.set_postfix({"loss": f"{epoch_loss:.4f}"})
		
	return loss_list, qnn_model