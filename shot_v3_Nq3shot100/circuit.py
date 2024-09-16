import torch
import pennylane as qml # type: ignore
from matplotlib import pyplot as plt
from pennylane import numpy as np
from MyModel_1state import random_state_fun

def arbitrary_state_circuit( Nq):	
	state_vector = torch.load('state0.pt')
	wires = [i for i in range(Nq)]
	qml.QubitStateVector(state_vector, wires)

def my_Uj_function(params_block , Nj, Num_block ):
	params_block = params_block.reshape(Num_block, Nj,3)
	for b in range(Num_block):
		for j in range(Nj):
			qml.Rot(params_block[b,j,0],params_block[b,j,1],params_block[b,j,2],wires=j)
		qml.broadcast(qml.CNOT, wires=[j for j in range(Nj)], pattern="chain")
	
def my_quantum_mz_function(params, Nq,  Nj ):
	arbitrary_state_circuit(Nq)
	for Nc in range(Nq, Nj-1,-1):
		Num_block = Nc
		params_block = params[Nc-1, 0:Num_block, 0:Nc, 0:3]
		my_Uj_function(params_block, Nc, Num_block)
	return qml.expval( qml.PauliZ(Nj-1) )

def my_quantum_p_function(params, Nq,  Nj ):
	arbitrary_state_circuit(Nq)
	for Nc in range(Nq, Nj-1,-1):
		Num_block = Nc
		params_block = params[Nc-1, 0:Num_block, 0:Nc, 0:3]
		my_Uj_function(params_block, Nc, Num_block)
	return qml.purity( wires=Nj-1 ), qml.expval( qml.PauliZ(Nj-1) )


if __name__ =='__main__':
	Nq = 3 

	# 使用 QubitStateVector 来嵌入包含相位信息的量子态
	# state_vector = random_state_fun(Nq)
	# torch.save(state_vector,'state0.pt')
	
	dev1 = qml.device("default.qubit", wires=Nq)
	dev2 = qml.device("default.qubit", wires=Nq)
	circuit_train = qml.QNode( my_quantum_mz_function, dev1 )
	circuit_test  = qml.QNode( my_quantum_p_function, dev2  )
	params = torch.randn((Nq,Nq,Nq,3), requires_grad=False)*0.01

	# Purity = circuit_test(params= params, Nq = Nq, Nj = Nq,  shots = None)


	lr = 0.01
	N_shots = 100
	for Nj in range(Nq,0,-1):
		Num_block = Nj
		epoch =0
		Pz = 0.
		while  abs(Pz+1)>0.0001:
			Pz = circuit_train(params= params, Nq = Nq, Nj = Nj,  shots = N_shots)
			Purity, Pz_ex = circuit_test(params= params, Nq = Nq, Nj = Nj,  shots = None)
			print('epoch',epoch,'\tPz', Pz.item(),'\t Pz_ex', Pz_ex.item(),'\t purity', Purity.item())
			if epoch % 5 ==0:
				with open("loss_Nj"+str(Nj)+".txt", "a+",  buffering=1000000) as file:
					file.write( str(epoch)+'\t'+str(Pz.item()) +'\t'+str(Pz_ex.item()) +'\t'+str(Purity.item())+'\n') 
			epoch = epoch+1
			for b in range(Num_block):
				for j in range(Nj):
					for i in range(0,3,1):
						params_add=params.clone()
						params_add[Nj-1,b,j,i] = params_add[Nj-1,b,j,i] +  np.pi/2.
						params_sub=params.clone()
						params_sub[Nj-1,b,j,i] = params_sub[Nj-1,b,j,i] -  np.pi/2.
						Pz_add = circuit_train(params= params_add, Nq = Nq, Nj = Nj,  shots = N_shots)
						Pz_sub = circuit_train(params= params_sub, Nq = Nq, Nj = Nj,  shots = N_shots)
						grad_ps = (Pz_add- Pz_sub)/2.0
						params_new=params.clone()
						params_new[Nj-1,b,j,i]=params_new[Nj-1,b,j,i] - lr*grad_ps
						params = params_new
						del params_new, params_add, params_sub	
		with open("loss_Nj"+str(Nj)+".txt", "a+",  buffering=1000000) as file:
			file.write( str(epoch)+'\t'+str(Pz.item()) +'\t'+str(Pz_ex.item()) +'\t'+str(Purity.item())+'\n') 	
		torch.save(params,'paramsNj'+str(Nj)+'.pt')
	



	

	# N_shots = None
	# for Nj in range(Nq,0,-1):
	# 	Num_block = Nj
	# 	for b in range(Num_block):
	# 		for j in range(Nj):
	# 			for i in range(0,3,1):
	# 				params_add=params.clone()
	# 				params_add[Nj-1,b,j,i] = params_add[Nj-1,b,j,i] +  np.pi/2.
	# 				params_sub=params.clone()
	# 				params_sub[Nj-1,b,j,i] = params_sub[Nj-1,b,j,i] -  np.pi/2.
	# 				Pz_add = circuit(params= params_add, Nq = Nq, Nj = Nj,  shots = N_shots)
	# 				Pz_sub = circuit(params= params_sub, Nq = Nq, Nj = Nj,  shots = N_shots)
	# 				grad_ps = (Pz_add- Pz_sub)/2.0

	# 				params.grad = None  # 或者使用 params.grad.zero_()
	# 				Pz = circuit(params= params, Nq = Nq, Nj = Nj,  shots = N_shots)
	# 				Pz.backward()
	# 				grad_auto =params.grad

	# 				print('grad_ps', grad_ps.item(), 'grad_auto', grad_auto[Nj-1,b,j,i])


