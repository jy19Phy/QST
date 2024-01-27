import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 


def H_gate(Nd=2):
	Hadamard_gate = torch.tensor([[1.+0.j, 1.+0.j],
        						  [1.+0.j, -1.+0.j ]])/np.sqrt(2.0)
	return Hadamard_gate

def T_gate(Nd=2):
	T = torch.tensor([[1.+0.j, 0.+0.j],
        			  [0.+0.j, (1.+1.j)/np.sqrt(2.0) ]])
	return T

def S_gate(Nd=2):
	S = torch.tensor([[1.+0.j, 0.+0.j],
        			  [0.+0.j, 1.j  ]])
	return S

def Px_gate(Nd=2):
	Px = torch.tensor([[0.+0.j, 1.+0.j],
        			  [1.+0.j, 0.+0.j  ]])
	return Px

def Py_gate(Nd=2):
	Py = torch.tensor([[0.+0.j, 0.-1.j],
        			  [0.+1.j, 0.+0.j  ]])
	return Py

def Pz_gate(Nd=2):
	Pz = torch.tensor([[1.+0.j, 0.+0.j],
        			  [0.+0.j, -1.0+0.j  ]])
	return Pz


def CNOT_gate( Nd = 2):
	CT_gate = torch.tensor(    [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        						[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        						[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        						[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )
	return CT_gate



# def Gate_ensemble( gateID ):
# 	if gateID == 0  :
# 		U = Paulix_gate()
# 	elif gateID==1:
# 		U = Pauliy_gate()
# 	elif gateID==2:
# 		U = Pauliz_gate()
# 	elif gateID==3 :
# 		U = Hadamard_gate()
# 	elif gateID==4:
# 		U = CNOT_gate()
# 	else:
# 		raise ValueError("gateID does not eist in gate ensemble.")
# 	return U


if __name__ == '__main__':
	torch.set_num_threads(1)
	#======================================================
	U = H_gate()
	# U = T_gate()
	Res = torch.matmul( torch.transpose(U ,0, 1 ), torch.conj( U)  )
	print( Res )



	# U = Gate_ensemble(gateID=4)
	# print(U)
	
	# # U = Pauli_gate(ID = 0)
	# print(U)
	# U = Pauli_gate(ID = 1)
	# print(U)
	# U = Pauli_gate(ID = 2)
	# print(U)
	# U = Pauli_gate(ID = 3)
	# print(U)

	# U = Hadamard_gate()
	# print(U)

	# U = CNOT_gate()
	# print(U)


	


