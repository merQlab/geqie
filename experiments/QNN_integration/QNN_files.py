import os, glob, json
from concurrent import futures

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Linear, CrossEntropyLoss, MSELoss, NLLLoss
from torch.optim import LBFGS, Adam

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.gradients import ParamShiftSamplerGradient
from qiskit_machine_learning.connectors.torch_connector import _TorchNNFunction

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	confusion_matrix,
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	classification_report
)
from PIL import Image
from tqdm import tqdm

import geqie
from geqie.encodings import frqi


class QNN_Pythorch_Module(nn.Module):
	def __init__(self, num_classes, num_qubits=7, num_layers=3):
		super().__init__()
		self.num_qubits = num_qubits
		self.num_classes = num_classes
		
		# 1. Create the parameterized VQC circuit ONCE
		self.vqc_circuit = QuantumCircuit(num_qubits)
		self.thetas = ParameterVector("theta", length=3 * num_qubits * num_layers)
		
		# Apply the brickwork architecture to the VQC circuit
		for layer in range(num_layers):
			offset = layer * 3 * num_qubits
			for i in range(num_qubits):
				self.vqc_circuit.rx(self.thetas[offset + i], i)
			for i in range(num_qubits):
				self.vqc_circuit.ry(self.thetas[offset + num_qubits + i], i)
			for i in range(num_qubits):
				self.vqc_circuit.rz(self.thetas[offset + 2*num_qubits + i], i)
			
			# Alternating brickwork entanglement
			if layer % 2 == 0:
				for i in range(0, num_qubits - 1, 2):
					self.vqc_circuit.cx(i, i + 1)
			else:
				for i in range(1, num_qubits - 1, 2):
					self.vqc_circuit.cx(i, i + 1)
				self.vqc_circuit.cx(num_qubits - 1, 0)
		
		# 2. Register the quantum weights directly as a native PyTorch Parameter.
		# Initialized from -pi to pi so the gates are fully active!
		self.quantum_weight = nn.Parameter(
			torch.empty(len(self.thetas)).uniform_(-np.pi, np.pi)
		)
		
		# 3. Classical head
		self.classical_head = nn.Linear(2**num_qubits, num_classes)

	def forward(self, matrix, default_shots=8192):
		# Build a clean, fresh circuit for this specific image safely
		qc = QuantumCircuit(self.num_qubits)
		qc.append(UnitaryGate(matrix), range(self.num_qubits))
		qc.compose(self.vqc_circuit, inplace=True)
		
		# Build the SamplerQNN using the active parameters
		fresh_sampler = Sampler(default_shots=default_shots)
		fresh_qnn = SamplerQNN(
			circuit=qc,
			input_params=None,
			weight_params=list(qc.parameters),
			sampler=fresh_sampler,
			gradient=ParamShiftSamplerGradient(sampler=fresh_sampler)
		)
		
		# Use raw PyTorch Autograd to bind our master weight to the QNN.
		# This completely bypasses TorchConnector's restrictive initialization.
		probs = _TorchNNFunction.apply(
			torch.empty(0),                 # input_data
			self.quantum_weight,  # weights (Our registered nn.Parameter!)
			fresh_qnn,            # neural_network
			False                 # sparse
		)
		# Feature scaling, drastically improves training time
		# Feels illegal, but it works, just don't let the gradient police know
		scaled_probs = probs * (2 ** self.num_qubits)
		
		# Classical classification
		logits = self.classical_head(scaled_probs)
		return torch.log_softmax(logits, dim=-1)
	
def load_precomputed_batches(split_dir):
	batch_files = sorted(glob.glob(os.path.join(split_dir, "batch_*.npz")))

	for batch_file in batch_files:
		data = np.load(batch_file, allow_pickle=True)
		matrices = data["matrices"]
		labels = data["labels"]
		yield matrices, labels


def _save_training_checkpoint(
	checkpoint_path,
	qnn_model,
	optimizer,
	epoch,
	best_val_loss,
	train_losses,
	val_losses,
	quantum_layer_weights_history,
	classical_head_weights_history,
	best_epoch,
	num_classes,
):
	if checkpoint_path is None:
		return

	os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
	torch.save(
		{
			"epoch": epoch,
			"best_epoch": best_epoch,
			"best_val_loss": best_val_loss,
			"num_classes": num_classes,
			"model_state_dict": qnn_model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"train_losses": train_losses,
			"val_losses": val_losses,
			"quantum_layer_weights_history": quantum_layer_weights_history,
			"classical_head_weights_history": classical_head_weights_history,
		},
		checkpoint_path,
	)


def train_QNN(
	epochs=20,
	num_classes=-1,
	train_dir=None,
	val_dir=None,
	checkpoint_path=None,
	resume_training=True,
) -> dict:
	f_loss = NLLLoss()
	qnn_model = QNN_Pythorch_Module(num_classes=num_classes, num_qubits=7, num_layers=1)
	
	optimizer = Adam([
		{'params': qnn_model.quantum_weight, 'lr': 0.05},
		{'params': qnn_model.classical_head.parameters(), 'lr': 0.001}
	])
	
	start_epoch = 0
	best_epoch = -1
	best_val_loss = np.inf
	quantum_layer_weights_history = []
	classical_head_weights_history = []
	train_losses = []
	val_losses = []

	if checkpoint_path is not None and resume_training and os.path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path, map_location="cpu")
		qnn_model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		start_epoch = int(checkpoint.get("epoch", -1)) + 1
		best_epoch = int(checkpoint.get("best_epoch", -1))
		best_val_loss = float(checkpoint.get("best_val_loss", np.inf))
		train_losses = list(checkpoint.get("train_losses", []))
		val_losses = list(checkpoint.get("val_losses", []))
		quantum_layer_weights_history = list(checkpoint.get("quantum_layer_weights_history", []))
		classical_head_weights_history = list(checkpoint.get("classical_head_weights_history", []))

	if start_epoch >= epochs:
		return {
			'train_losses': train_losses,
			'val_losses': val_losses,
			'qnn_model': qnn_model,
			'quantum_layer_weights_history': quantum_layer_weights_history,
			'classical_head_weights_history': classical_head_weights_history,
			'best_val_loss': best_val_loss,
			'best_epoch': best_epoch,
			'last_epoch': start_epoch - 1,
		}

	qnn_model.train()
	progress_bar_training = tqdm(range(start_epoch, epochs), desc="Training")

	for epoch in range(start_epoch, epochs):
		qnn_model.train()		

		for (i, (matrices, labels)) in enumerate(load_precomputed_batches(train_dir)):			
			for matrix, label in zip(matrices, labels):
				matrix_tensor = matrix
				label_tensor = torch.tensor([label], dtype=torch.long)

				output = qnn_model(matrix_tensor)
				if output.dim() == 1:			# To ensure the output matches
					output = output.unsqueeze(0)

				loss = f_loss(output, label_tensor)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		quantum_layer_weights_history.append(qnn_model.quantum_weight.detach().cpu().numpy().copy().flatten())
		classical_head_weights_history.append(qnn_model.classical_head.weight.detach().cpu().numpy().copy().flatten())	

		epoch_train_loss = float(loss.item())
		train_losses.append(epoch_train_loss)
		postfix = {"loss": f"{epoch_train_loss:.2f}"}			

		qnn_model.eval()
		correct = 0
		total = 0

		with torch.no_grad():
			for matrices, labels in load_precomputed_batches(val_dir):
				for matrix, label in zip(matrices, labels):
					label_tensor = torch.tensor([label], dtype=torch.long)

					output = qnn_model(matrix)
					if output.dim() == 1: 		# To ensure the output matches
						output = output.unsqueeze(0)

					loss = f_loss(output, label_tensor)						

					pred = output.argmax(dim=1).item()
					correct += int(pred == label)
					total += 1

		epoch_val_loss = float(loss.item())
		val_losses.append(epoch_val_loss)
		postfix["val_loss"] = f"{epoch_val_loss:.2f}"

		if epoch_val_loss < best_val_loss:
			best_val_loss = epoch_val_loss
			best_epoch = epoch
			_save_training_checkpoint(
				checkpoint_path=checkpoint_path,
				qnn_model=qnn_model,
				optimizer=optimizer,
				epoch=epoch,
				best_val_loss=best_val_loss,
				train_losses=train_losses,
				val_losses=val_losses,
				quantum_layer_weights_history=quantum_layer_weights_history,
				classical_head_weights_history=classical_head_weights_history,
				best_epoch=best_epoch,
				num_classes=num_classes,
			)

		val_acc = correct / total if total > 0 else 0.0
		# print(
		# 	f"Epoch {epoch+1}: "
		# 	f"train_loss={np.mean(train_losses):.4f}, "
		# 	f"val_loss={np.mean(val_losses):.4f}, "
		# 	f"val_acc={val_acc:.4f}"
		# )

		progress_bar_training.update(1)
		progress_bar_training.set_postfix(postfix)			

	return {
		'train_losses': train_losses,
		'val_losses': val_losses,
		'qnn_model': qnn_model,
		'quantum_layer_weights_history': quantum_layer_weights_history,
		'classical_head_weights_history': classical_head_weights_history,
		'best_val_loss': best_val_loss,
		'best_epoch': best_epoch,
		'last_epoch': epochs - 1,
	}

def test_QNN(qnn_model=None, num_classes=-1, test_dir=None):
	''' 
	This function takes trained QNN model and evaluates it based on the test set.
	Returns classification statistics together with confusion matrix.
	Supports multiclass classification.
	'''

	if qnn_model is None:
		raise ValueError("Parameter 'qnn_model' cannot be None.")

	qnn_model.eval()

	y_true = []
	y_pred = []
	test_losses = []

	f_loss = NLLLoss()

	with torch.no_grad():
		for matrices, labels in load_precomputed_batches(test_dir):
			for matrix, label in zip(matrices, labels):
				label_tensor = torch.tensor([int(label)], dtype=torch.long)

				output = qnn_model(matrix)
				if output.dim() == 1: 		# To ensure the output matches
					output = output.unsqueeze(0)
				loss = f_loss(output, label_tensor)
				test_losses.append(loss.item())

				pred = output.argmax(dim=1).item()

				y_true.append(int(label))
				y_pred.append(int(pred))

	if len(y_true) == 0:
		raise ValueError(f"No samples found in test_dir='{test_dir}'.")

	# Determine labels present in y_true and y_pred
	labels = list(range(num_classes))

	cm = confusion_matrix(y_true, y_pred, labels=labels)
	acc = accuracy_score(y_true, y_pred)

	# For multiclass case, weighted average is usually the safest default
	precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
	recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
	f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

	cls_report = classification_report(
		y_true,
		y_pred,
		labels=labels,
		zero_division=0
	)

	results = {
		"test_loss": float(np.mean(test_losses)),
		"accuracy": float(acc),
		"precision_weighted": float(precision),
		"recall_weighted": float(recall),
		"f1_weighted": float(f1),
		"confusion_matrix": cm,
		"labels": labels,
		"classification_report": cls_report,
		"y_true": y_true,
		"y_pred": y_pred
	}

	print(f"Test loss:  {results['test_loss']:.4f}")
	print(f"Accuracy:   {results['accuracy']:.4f}")
	print(f"Precision:  {results['precision_weighted']:.4f}")
	print(f"Recall:     {results['recall_weighted']:.4f}")
	print(f"F1-score:   {results['f1_weighted']:.4f}")
	print("\nConfusion matrix:")
	print(results["confusion_matrix"])
	print("\nClassification report:")
	print(results["classification_report"])

	return results


def load_and_process_mnist_dataset(labels_to_include=[0, 1], 
								   n_samples_per_label=100, 
								   resize=(8, 8), 
								   path_to_mnist_dataset=None):

	mnist_dataset = load_dataset("parquet", data_files=str(path_to_mnist_dataset))

	selected_images = []
	images_per_label_count = {label: 0 for label in labels_to_include}
	for image_idx in range(len(mnist_dataset["train"])):
		# First, select only images with labels in labels_to_include:

		if mnist_dataset["train"][image_idx]["label"] in labels_to_include and images_per_label_count[mnist_dataset["train"][image_idx]["label"]] < n_samples_per_label:
			img_8x8 = mnist_dataset["train"][image_idx]["image"].resize(resize, resample=Image.BILINEAR)
			img = np.array(img_8x8)
			selected_images.append({"image": img, "label": mnist_dataset["train"][image_idx]["label"]})
			images_per_label_count[mnist_dataset["train"][image_idx]["label"]] += 1

	X = np.array([item["image"] for item in selected_images])
	y = np.array([item["label"] for item in selected_images])

	return X, y

def precompute_and_save_circuits(X_data, y_data, batch_size=5, save_dir=".circuits"):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
	num_samples = len(X_data)

	for i in range(0, num_samples, batch_size):
		batch_X = X_data[i : i + batch_size]
		batch_y = y_data[i : i + batch_size]
		
		batch_matrices = []
		
		for img in batch_X:
			# Encode the raw 0-255 uint8 image into a QuantumCircuit
			qc = geqie.encode(frqi.init_function, frqi.data_function, frqi.map_function, img)
			
			# Unpack the composite instructions into base gates
			flat_qc = qc.decompose()
			
			# Keep unpacking in case geqie nested them multiple layers deep
			while len(flat_qc.data) != len(flat_qc.decompose().data):
				flat_qc = flat_qc.decompose()
				
			# Filer out non-unitary gates
			pure_qc = QuantumCircuit(flat_qc.num_qubits)
			for instruction in flat_qc.data:
				# Handle different Qiskit version data structures safely
				op_name = instruction.operation.name if hasattr(instruction, 'operation') else instruction[0].name
				
				# Now that the 'reset' is exposed, this will catch and delete it!
				if op_name not in ['reset', 'measure', 'barrier']:
					pure_qc.append(instruction)
			
			# Extract the exact unitary matrix using Qiskit Operator
			unitary_matrix = Operator(pure_qc).data
			
			# Cast as complex128 to preserve strict unitarity
			unitary_matrix = np.array(unitary_matrix, dtype=np.complex128)
			batch_matrices.append(unitary_matrix)
			
		# Save to .npz file
		batch_filename = os.path.join(save_dir, f"batch_{i//batch_size}.npz")
		np.savez(batch_filename, matrices=batch_matrices, labels=batch_y)

		print(f"Saved {batch_filename} with dtype {batch_matrices[0].dtype}")

def precompute_and_save_split(
	X_data,
	y_data,
	batch_size=5,
	save_dir=".circuits",
	split_name="train",
	num_workers=1,
):
	split_dir = os.path.join(save_dir, split_name)
	os.makedirs(split_dir, exist_ok=True)

	progress_bar = tqdm(total=range(0, len(X_data), batch_size), desc=f"Precomputing {split_name}")

	num_samples = len(X_data)
	batch_files = []

	def _save_batch(batch_idx, batch_matrices, batch_labels):
		batch_filename = os.path.join(split_dir, f"batch_{batch_idx:04d}.npz")
		np.savez_compressed(
			batch_filename,
			matrices=np.array(batch_matrices, dtype=np.complex128),
			labels=np.array(batch_labels),
		)
		batch_files.append(batch_filename)
		progress_bar.update(batch_size)
		# print(f"[{split_name}] saved: {batch_filename}")

	if num_workers is None or int(num_workers) <= 1:
		for i in range(0, num_samples, batch_size):

			batch_idx = i // batch_size
			batch_X = X_data[i:i + batch_size]
			batch_y = y_data[i:i + batch_size]
			batch_idx, batch_matrices, batch_labels = _precompute_batch(
				batch_idx=batch_idx,
				batch_X=batch_X,
				batch_y=batch_y,
			)
			_save_batch(batch_idx, batch_matrices, batch_labels)
	else:
		future_to_batch = {}
		with futures.ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
			for i in range(0, num_samples, batch_size):
				batch_idx = i // batch_size
				batch_X = X_data[i:i + batch_size]
				batch_y = y_data[i:i + batch_size]
				future = executor.submit(
					_precompute_batch,
					batch_idx,
					batch_X,
					batch_y,
				)
				future_to_batch[future] = batch_idx

			for future in tqdm(
				futures.as_completed(future_to_batch),
				total=len(future_to_batch),
				desc=f"Precomputing {split_name}",
			):
				batch_idx, batch_matrices, batch_labels = future.result()
				_save_batch(batch_idx, batch_matrices, batch_labels)

	metadata = {
		"split_name": split_name,
		"num_samples": int(num_samples),
		"batch_size": int(batch_size),
		"num_batches": int(len(batch_files)),
		"files": [os.path.basename(f) for f in sorted(batch_files)],
	}

	with open(os.path.join(split_dir, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)

	return split_dir


def _precompute_batch(batch_idx, batch_X, batch_y):
	batch_matrices = []

	for img in batch_X:
		qc = geqie.encode(
			frqi.init_function,
			frqi.data_function,
			frqi.map_function,
			img,
		)

		flat_qc = qc.decompose()
		while len(flat_qc.data) != len(flat_qc.decompose().data):
			flat_qc = flat_qc.decompose()

		pure_qc = QuantumCircuit(flat_qc.num_qubits)
		for instruction in flat_qc.data:
			op_name = (
				instruction.operation.name
				if hasattr(instruction, "operation")
				else instruction[0].name
			)

			if op_name not in ["reset", "measure", "barrier"]:
				pure_qc.append(instruction)

		unitary_matrix = np.array(Operator(pure_qc).data, dtype=np.complex128)
		batch_matrices.append(unitary_matrix)

	return batch_idx, batch_matrices, np.array(batch_y)

def prepare_train_val_precomputed(
	X,
	y,
	val_size=0.2,
	batch_size=5,
	save_dir=".circuits",
	random_state=42,
	stratify=True,
	num_workers=1,
):
	stratify_labels = y if stratify else None

	X_train, X_val, y_train, y_val = train_test_split(
		X,
		y,
		test_size=val_size,
		random_state=random_state,
		shuffle=True,
		stratify=stratify_labels
	)

	train_dir = precompute_and_save_split(
		X_train,
		y_train,
		batch_size=batch_size,
		save_dir=save_dir,
		split_name="train",
		num_workers=num_workers,
	)

	val_dir = precompute_and_save_split(
		X_val,
		y_val,
		batch_size=batch_size,
		save_dir=save_dir,
		split_name="val",
		num_workers=num_workers,
	)

	return {
		"X_train": X_train,
		"y_train": y_train,
		"X_val": X_val,
		"y_val": y_val,
		"train_dir": train_dir,
		"val_dir": val_dir,
	}

def prepare_test_precomputed(
	X,
	y,
	batch_size=5,
	save_dir=".circuits",
	random_state=42,
	stratify=True,
	num_workers=1,
):
	stratify_labels = y if stratify else None

	test_dir = precompute_and_save_split(
		X,
		y,
		batch_size=batch_size,
		save_dir=save_dir,
		split_name="test",
		num_workers=num_workers,
	)

	return {
		"X_test": X,
		"y_test": y,
		"test_dir": test_dir,
	}
