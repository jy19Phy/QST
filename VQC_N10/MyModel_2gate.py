import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 


# def random_onequbit_gate_fun(theta_para , Np=3, Nd=2):
# 	theta_para= theta_para.reshape(Np)
# 	theta_x = theta_para[0]
# 	theta_y = theta_para[1]
# 	theta_z = theta_para[2]
# 	sigma_x = torch.tensor([[0.0+0.0j, 1.0+0.0j],  [1.0+0.0j, 0.0+0.0j]])
# 	sigma_y = torch.tensor([[0.0+0.0j, 0.0-1.0j],  [0.0+1.0j, 0.0+0.0j]])
# 	sigma_z = torch.tensor([[1.0+0.0j, 0.0+0.0j],  [0.0+0.0j,-1.0+0.0j]])
# 	A = theta_x*sigma_x+theta_y*sigma_y+theta_z*sigma_z 
# 	U = torch.matrix_exp(1.j*A)
# 	U = U.reshape(Nd,Nd)
# 	return U

def random_onequbit_gate_fun(theta_para , Np=3, Nd=2):
	theta_para= theta_para.reshape(Np)
	phi = theta_para[0]
	theta = theta_para[1]
	omega = theta_para[2]
	U11= torch.exp(-1.j*(phi+omega)*0.5) *torch.cos(theta/2.)	
	U22= torch.exp( 1.j*(phi+omega)*0.5) *torch.cos(theta/2.)
	U12= -1.*torch.exp(1.j*(phi-omega)*0.5) *torch.sin(theta/2.)
	U21= torch.exp(-1.j*(phi-omega)*0.5) *torch.sin(theta/2.)
	U = torch.cat( (U11.reshape(1,1), U12.reshape(1,1) , U21.reshape(1,1), U22.reshape(1,1)), dim = 1 )
	# print(U.shape)
	U = U.reshape(Nd, Nd)
	# print(torch.matmul(U, torch.conj(torch.transpose(U, 0, 1 )) ))
	return U 




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
	U = Gate_ensemble(gateID=4)
	print(U)
	
	# U = Pauli_gate(ID = 0)
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


	


