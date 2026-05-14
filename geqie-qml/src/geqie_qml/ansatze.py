from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
import numpy as np
from itertools import combinations, pairwise


def default_vqc_ansatz(num_qubits: int, num_layers: int, **_):
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

def QCNN_layer(input_qubits: int, output_qubits: int, **_):
	"""Construct a QCNN layer with the specified parameters."""
	# The actual circuit construction and QNN setup would go here,
	# using ansatz_factory(num_qubits, num_layers) to get the parameterised circuit.	
	output_qubits = 1 if output_qubits is None else output_qubits
	quantum_register = QuantumRegister(input_qubits)
	classical_register = ClassicalRegister(output_qubits)
	qcnn_circuit = QuantumCircuit(quantum_register, classical_register)
	if output_qubits < 1 or output_qubits > input_qubits:
		raise ValueError("output_qubits must be between 1 and input_qubits")

	# ================================================================================================
	# HELPER METHODS:
	# ================================================================================================
	theta_index = 0	
	theta = ParameterVector("theta", length=0)
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

	# ================================================================================================
	# MAIN LAYER CONSTRUCTION LOOP:
	# ================================================================================================

	for idx in range(int((input_qubits))):
		# # 0. Check if the leftover qubits are odd or even. If odd, apply a convolution and pooling gate on the last 2 qubits, then decrement the leftover qubits by 1.
		# if input_qubits % 2 != 0:
		# 	fix_parity_of_qubits()

		# max_even_qubits = input_qubits # if input_qubits % 2 == 0 else input_qubits - 1		
		
		# 1. CONVOLUTION GATE:		
		start_convolution_qubit_pos = int(np.floor(input_qubits - input_qubits / (2**idx)))

		convolution_qubit_pairs= list(combinations(range(start_convolution_qubit_pos, input_qubits), 2))
		if convolution_qubit_pairs:		
			conv_layer, used_qubits = build_convolution_layer(
				convolution_qubit_pairs,
				next_thetas,
				label=f"Convolution {idx}",
				)
 
			qcnn_circuit.append(
				conv_layer.to_instruction(label=f"Conv {idx}"),
				[qcnn_circuit.qubits[q] for q in used_qubits],
				)			

		# 2. POOLINGS GATES:
		start_pooling_qubit_position = start_convolution_qubit_pos
		
		# 2.1. Special case - we are reaching the limit given by the output_qubits. In this case, we apply a "stairway" pooling.
		next_convolution_qubit_pos = int(np.floor(input_qubits - input_qubits / (2**(idx + 1))))
		length_of_next_convolution = len(range(next_convolution_qubit_pos, input_qubits))
		if length_of_next_convolution < output_qubits:
			'''In case when the next convolution would reduce the qubits below the output_qubits, we skip standard pooling.
			Instead, we apply a final pooling "stairs" (i.e. pairwise pooling).
			Once the stairway pooling reaches the output_qubits, we break the loop.'''
			pooling_steps = input_qubits - start_convolution_qubit_pos - output_qubits
			pairwise_pooling_qubits = list(pairwise(range(start_pooling_qubit_position, start_pooling_qubit_position + pooling_steps + 1)))
			if pairwise_pooling_qubits:
				pool_layer, used_qubits = build_pooling_layer(
					pairwise_pooling_qubits,
					next_thetas,
					label=f"Final pooling {idx}",
					)
				qcnn_circuit.append(
					pool_layer.to_instruction(label=f"F_pool {idx}"),
					[qcnn_circuit.qubits[q] for q in used_qubits],
					)
				
				qcnn_circuit.barrier()
			
			qcnn_circuit.measure(
				qcnn_circuit.qubits[-output_qubits:],
				qcnn_circuit.clbits[:output_qubits],
			)
			return qcnn_circuit

		# 2.2. Standard case - we can still apply the regular pooling (i.e. pooling half of the qubits).
		qubits_to_be_pooled = input_qubits - start_pooling_qubit_position
		half_qubits_pos = int((qubits_to_be_pooled) / 2)
		least_pooling_pos = half_qubits_pos 

		pooling_qubit_pairs = list(zip(
			range(start_pooling_qubit_position, start_pooling_qubit_position + least_pooling_pos),
			range(start_pooling_qubit_position + least_pooling_pos, input_qubits),
		))
		if pooling_qubit_pairs:
			pool_layer, used_qubits = build_pooling_layer(
				pooling_qubit_pairs,
				next_thetas,
				label=f"Pooling {idx}",
			)
			qcnn_circuit.append(
				pool_layer.to_instruction(label=f"Pool {idx}"),
				[qcnn_circuit.qubits[q] for q in used_qubits],
			)
		qcnn_circuit.barrier() 
	
	qcnn_circuit.measure(
		qcnn_circuit.qubits[-output_qubits:],
		qcnn_circuit.clbits[:output_qubits],
	)

	return qcnn_circuit

