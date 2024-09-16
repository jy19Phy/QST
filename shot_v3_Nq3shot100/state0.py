import pennylane as qml # type: ignore
import torch
import numpy as np
from matplotlib import pyplot as plt

def arbitrary_state_circuit(state_vector,  Nq):
	# 使用 QubitStateVector 来嵌入包含相位信息的量子态
	wires = [i for i in range(Nq)]
	qml.QubitStateVector(state_vector, wires)
	
	return qml.state()


if __name__ == '__main__':

	Nq = 5

	# 执行电路
	dev = qml.device("default.qubit", wires= Nq, shots =None)
	circuit = qml.QNode( arbitrary_state_circuit, dev )
	state_vector = torch.load('state0.pt')
	result = circuit(state_vector, Nq)
	print("Final state:", result,result.shape)
	Norm = torch.sum( torch.square( torch.abs( result)))
	print("Final state:",  Norm)
