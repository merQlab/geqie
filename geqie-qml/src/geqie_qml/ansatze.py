from threading import local

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
import numpy as np
from itertools import combinations


def default_vqc_ansatz(num_qubits: int, num_layers: int):
	"""Reconstruct the parameterised VQC (without the image unitary)."""
	thetas = ParameterVector("theta", length=3 * num_qubits * num_layers)
	vqc = QuantumCircuit(num_qubits)
	for layer in range(num_layers):
		offset = layer * 3 * num_qubits
		for i in range(num_qubits):
			vqc.rx(thetas[offset + i], i)
		for i in range(num_qubits):
			vqc.ry(thetas[offset + num_qubits + i], i)
		for i in range(num_qubits):
			vqc.rz(thetas[offset + 2 * num_qubits + i], i)
		# Alternating brickwork entanglement
		if layer % 2 == 0:
			for i in range(0, num_qubits - 1, 2):
				vqc.cx(i, i + 1)
		else:
			for i in range(1, num_qubits - 1, 2):
				vqc.cx(i, i + 1)
			vqc.cx(num_qubits - 1, 0)
	return vqc

def QCNN_layer(input_qubits: int, output_qubits: int = None):
	"""Construct a QCNN layer with the specified parameters."""
	# The actual circuit construction and QNN setup would go here,
	# using ansatz_factory(num_qubits, num_layers) to get the parameterised circuit.
	quantum_register = QuantumRegister(input_qubits)
	classical_register = ClassicalRegister(output_qubits if output_qubits is not None else 1)
	qcnn_circuit = QuantumCircuit(quantum_register, classical_register)
	leftover_qubits = input_qubits
	theta = ParameterVector("theta", length=0)
	theta_index = 0			
	
	output_qubits = 1 if output_qubits is None else output_qubits

	# ================================================================================================
	# HELPER METHODS:
	# ================================================================================================

	def next_thetas(count: int = 1):
		nonlocal theta_index
		start = theta_index
		theta_index += count
		theta.resize(theta_index)
		return theta[start:theta_index]
	
	def build_convolution_layer(qubit_pairs, next_thetas, label):
		used_qubits = sorted({q for pair in qubit_pairs for q in pair})
		local = {global_q: local_q for local_q, global_q in enumerate(used_qubits)}
		conv_layer = QuantumCircuit(len(used_qubits), name=label)

		for pair in qubit_pairs:
			q0 = local[pair[0]]
			q1 = local[pair[1]]
			t0, t1 = next_thetas(2)

			conv_layer.rz(np.pi / 2, q1)
			conv_layer.cx(q1, q0)
			conv_layer.rz(2 * t0 - np.pi / 2, q0)
			conv_layer.ry(np.pi / 2 - t1, q1)
			conv_layer.cx(q1, q0)
			conv_layer.ry(-np.pi / 2, q0)

		return conv_layer, used_qubits
	
	def build_pooling_layer(qubit_pairs, next_thetas, label):
		used_qubits = sorted({q for pair in qubit_pairs for q in pair})
		local = {global_q: local_q for local_q, global_q in enumerate(used_qubits)}
		pool_layer = QuantumCircuit(len(used_qubits), name=label)		

		for pair in qubit_pairs:
			q0 = local[pair[0]]
			q1 = local[pair[1]]
	
			t0, t1 = next_thetas(2)

			pool_layer.rz(-np.pi / 2, q1)
			pool_layer.cx(q1, q0)
			pool_layer.rz(2 * t0 - np.pi / 2, q0)
			pool_layer.cx(q0, q1)
			pool_layer.ry(np.pi / 2 - t1, q1)
			pool_layer.ry(np.pi / 2, q1)

		return pool_layer, used_qubits
		
	def fix_parity_of_qubits():
		last_pair = (input_qubits - 1, input_qubits - 2)
		last_pairs = [last_pair]

		# 1. Convolution gate:
		conv_layer, used_qubits = build_convolution_layer(
			last_pairs,
			next_thetas,
			label="Odd convolution",
		)

		qcnn_circuit.append(
			conv_layer.to_instruction(label="Odd Conv"),
			[qcnn_circuit.qubits[q] for q in used_qubits],
		)
		
		# 2. Pooling gate:		
		pool_layer, used_qubits = build_pooling_layer(
			last_pairs,
			next_thetas,
			label="Odd pooling",
		)
		qcnn_circuit.append(
			pool_layer.to_instruction(label="Odd Pool"),
			[qcnn_circuit.qubits[q] for q in used_qubits],
		)
		input_qubits -= 1
		qcnn_circuit.barrier()

	# ================================================================================================
	# MAIN LAYER CONSTRUCTION LOOP:
	# ================================================================================================

	for idx in range(int(np.ceil(np.log2(input_qubits)))):
		

		# # 0. Check if the leftover qubits are odd or even. If odd, apply a convolution and pooling gate on the last 2 qubits, then decrement the leftover qubits by 1.
		# if input_qubits % 2 != 0:
		# 	fix_parity_of_qubits()

		max_even_qubits = input_qubits # if input_qubits % 2 == 0 else input_qubits - 1
		
		
		# 1. CONVOLUTION GATE:
		start_convolution_qubit_pos = input_qubits - (input_qubits // (2**idx))
		convolution_qubit_pairs= list(combinations(range(start_convolution_qubit_pos, max_even_qubits), 2))
		conv_layer, used_qubits = build_convolution_layer(
			convolution_qubit_pairs,
			next_thetas,
			label=f"Convolution {idx}",
		)

		try: 
			qcnn_circuit.append(
				conv_layer.to_instruction(label=f"Conv {idx}"),
				[qcnn_circuit.qubits[q] for q in used_qubits],
			)
		except Exception as e:
			print(f"Error appending convolution layer {idx}: {e}")
			print(f"Convolution qubit pairs: {qcnn_circuit.qubits}")
			print(f"Used qubits: {used_qubits}")
			raise e
		
		# 2. POOLING GATE:
		start_pooling_qubit_position = input_qubits - (input_qubits // (2**(idx)))
		qubits_to_be_pooled = max_even_qubits - start_pooling_qubit_position
		half_qubits_pos = int((qubits_to_be_pooled) / 2)
		least_pooling_pos = half_qubits_pos # min(input_qubits - output_qubits, half_qubits_pos)

		pooling_qubit_pairs = list(zip(
			range(start_pooling_qubit_position, start_pooling_qubit_position + least_pooling_pos),
			range(start_pooling_qubit_position + least_pooling_pos, max_even_qubits),
		))
		pool_layer, used_qubits = build_pooling_layer(
			pooling_qubit_pairs,
			next_thetas,
			label=f"Pooling {idx}",
		)
		qcnn_circuit.append(
			pool_layer.to_instruction(label=f"Pool {idx}"),
			[qcnn_circuit.qubits[q] for q in used_qubits],
		)

		if qubits_to_be_pooled % 2 != 0 and start_pooling_qubit_position + least_pooling_pos != max_even_qubits - 1:
			odd_to_even_patch_pair = [(start_pooling_qubit_position + least_pooling_pos, max_even_qubits - 1)]
			print(f"Adding extra pooling layer for odd qubits: {odd_to_even_patch_pair}, idx: {idx}")
			
			pool_layer, used_qubits = build_pooling_layer(
				odd_to_even_patch_pair,
				next_thetas,
				label=f"Pooling {idx}",
				)
			qcnn_circuit.append(
				pool_layer.to_instruction(label=f"Pool {idx}"),
				[qcnn_circuit.qubits[q] for q in used_qubits],
				)
		
		qcnn_circuit.barrier() 

	
	qcnn_circuit.measure(output_qubits, qcnn_circuit.clbits[:output_qubits])
		# input_qubits = int(input_qubits / 2)

	return qcnn_circuit
	# if is_odd_amount_of_qubits:
		
