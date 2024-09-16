import torch
import pennylane as qml # type: ignore
from matplotlib import pyplot as plt



def my_quantum_function(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)
    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))


dev = qml.device("default.qubit", wires=3, shots =None)
circuit = qml.QNode( my_quantum_function, dev )

params = torch.tensor([ 0.37087464, -1.1752595,  -0.51433962 , 1.94757307, -1.29836488, -0.61030051])
shots_list = [1, 5, 10, 1000, 5000, 10000]
print(circuit(params= params, shots = shots_list))

print(qml.draw(circuit)(params))


