import torch
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, interface="torch")
def circuit(params, Nq):
	# qml.RX(params[0], wires=0)
	# qml.RY(params[1], wires=1)
	qml.Rot(params[0],params[1],params[2],wires=0)
	qml.CNOT(wires=[0, 1])
	return qml.expval(qml.PauliZ(0))

# 定义 torch 张量并设置 requires_grad=True
params = torch.randn(3, requires_grad=True)

# 计算电路输出
Nq =1 
output = circuit(params, Nq)
print(f"Output: {output}")

# 计算梯度
output.backward()
print(f"Gradient: {params.grad}")



for j in range(3):
	params_add=params.clone()
	params_add[j] = params_add[j] +  np.pi/2.
	params_sub=params.clone()
	params_sub[j] = params_sub[j] -  np.pi/2.
	Pz_add = circuit(params_add, Nq)
	Pz_sub = circuit(params_sub, Nq)
	grad_fs = (Pz_add- Pz_sub)/2.0
	print('grad_fs', grad_fs.item())