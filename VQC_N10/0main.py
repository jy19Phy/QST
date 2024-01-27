import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from MyModel_1state import *
from MyModel_4circuit import *
from MyModel_5Train import *
from MyModel_6Export import *
from MyModel_7Observable import *


def Exp(Nq,  repeat, train_loss ):
	state0 = random_state_fun (Nq= Nq)
	Export_SystemSetting(Nq = Nq,  Name_state0='Random')
	state = state0
	ParamSet = []
	stateSet = []
	Pset =[]
	stateSet.append(state0.reshape(1,2**Nq))
	for Nc in range(Nq,0,-1):
		Num_block = repeat*Nc
		depth = (Nc+ Nc-1)*Num_block
		param = torch.randn(  depth,  3)*0.01
		circuit_Model = circuit(Nc = Nc, Nq=Nq, Num_block = Num_block, param = param)
		qID_target = Nc-1
		MyTrain(circuit_Model).forward(state=state, qID = qID_target, Nq = Nq, train_loss = train_loss)
		state_new = circuit_Model(state)
		state = state_new.detach()
		ParamSet.append(param)
		stateSet.append(state.reshape(1,2**Nq))
		for q in range(Nq):
			P_0, _ = P_sigle_qubit_fun(state=state, qID = q,  Nq=Nq)
			print("Nc", Nc, "q = ", q, "P_0", P_0)
		P_qID,_ = P_sigle_qubit_fun(state=state, qID = qID_target,  Nq=Nq)
		Pset.append( torch.real(P_qID).reshape(1,1)  )
	# inverse process 
	state = state.detach()
	state = zero_state_fun(Nq=Nq)
	stateInvSet = []
	stateInvSet.append( state.reshape(1,2**Nq))
	for Nc in range(1,Nq+1):
		Num_block = repeat*Nc
		depth = (Nc+ Nc-1)*Num_block
		param= ParamSet[Nq-Nc]
		circuit_Model = circuit(Nc = Nc, Nq=Nq, Num_block = Num_block, param = param)
		state_new = circuit_Model.inverse(state)
		state = state_new.detach()
		stateInvSet.append( state.reshape(1,2**Nq))
	state_pred = state
	Export_param(Nq, state0, ParamSet, stateSet,  stateInvSet, Pset)
	Fidelity = torch.sum( torch.conj( state_pred.reshape(-1)) * state0.reshape(-1) )
	state_F = Fidelity
	print("state_Fidelity", Fidelity)
	rho_F  = Fidelity*torch.conj(Fidelity)
	print("rho_Fedelity", Fidelity*torch.conj(Fidelity))
	Export_endtime(Nq = Nq, state_F=state_F, rho_F = rho_F)
	return rho_F, torch.mean(torch.cat(Pset))


if __name__ == '__main__':
	torch.set_num_threads(1)
	#======================================================#
	Nq =10
	repeat = 3
	for train_loss in np.arange(0.03,0.05,0.01):
		with open("Fidelity.txt", "a+",  buffering=1000000) as file:
			file.write("Nq="+str(Nq)+"\t repeat="+str(repeat)+"\t train_loss="+str(train_loss)+"\n"  )
		for t in range(5):
			rho_F, P_ave= Exp(Nq = Nq, repeat=repeat , train_loss = train_loss)
			with open("Fidelity.txt", "a+",  buffering=1000000) as file:
				file.write(str(Nq)+"\t"+str(torch.real(rho_F).item())+"\t"+str( torch.real(P_ave).item())+"\n"  )
	




	









	# depth = (Nq + Nq-1)*Num_block

	# qID = Nq-1
	# # state1 = random_state_fun (Nq = Nq)
	# state1 = GHZ_state_fun(Nq= Nq)
	# param1 = torch.randn(  depth,  3)*0.01
	# circuit_Model1 = circuit(Nq=Nq, Num_block = Num_block, param = param1)
	# MyTrain(circuit_Model1).forward(state=state1, qID = qID, Nq = Nq, epoch = 10000)
	# state_f = circuit_Model1(state1)
	# # print(state_f.reshape(-1))
	# P_0, _ = P_sigle_qubit_fun(state=state_f, qID = qID,  Nq=Nq)
	# print("P_Nq0 =\t", P_0 ) 

	# # rho = rho_fun( state= state_f,Nq = Nq )
	# # rhoA= rhoA_partial_fun( state= state_f,Nq = Nq )
	# # rhoB= rhoB_partial_fun( state= state_f,Nq = Nq )
	# # print("rho", rho)
	# # print("rho_A", rhoA )
	# # print("rho_B", rhoB )
	# # rho_product = torch.kron(rhoA,rhoB).reshape([2**Nq, 2**Nq])
	# # # print("rho_product", rho_product)
	# # print("rho_diff", torch.sum( torch.abs(torch.real(rho-rho_product))+torch.abs(torch.imag(rho-rho_product)) )/2**(Nq*2) )

	# qID = Nq-2

	# state2 = state_f.detach()
	# Num_block = 3
	# depth = (Nq-1 + Nq-1-1)*Num_block
	# param2 = torch.randn(  depth,  3)*0.01
	# circuit_Model2 = circuit2(Nq=Nq, Num_block = Num_block, param = param2)
	# MyTrain(circuit_Model2).forward(state=state2, qID = qID, Nq = Nq, epoch = 10000)
	# state_f = circuit_Model2(state2)
	# P_0, _ = P_sigle_qubit_fun(state=state_f, qID= qID, Nq=Nq)
	# print("P_Nq0 =\t", P_0 ) 
	# P_0, _ = P_sigle_qubit_fun(state=state2, qID= Nq-1, Nq=Nq)
	# print("P_Nq0 =\t", P_0 )
	# P_0, _ = P_sigle_qubit_fun(state=state_f, qID= Nq-1, Nq=Nq)
	# print("P_Nq0 =\t", P_0 ) 





	# state = state_f
	# Nq = 2 
	# Num_block = 1
	# depth = (Nq + Nq-1)*Num_block
	# param2 = torch.randn(  depth,  3)*0.01
	# circuit_Model = circuit(Nq=Nq, Num_block = Num_block, param = param2)




	# print(state_f.reshape(-1))
	# c00 = state_f.reshape(-1)[0]
	# print("P00=\t", torch.conj(c00)*c00 ) 
	# state_pred = circuit_Model.inverse(state_zero)
	# Fidelity = torch.sum( torch.conj( state_pred.reshape(-1)) * state.reshape(-1) )
	# print("rho_Fedelity", Fidelity*torch.conj(Fidelity))

	# Obs_true = Mz_obs_fun(  state= state, Nq = Nq)
	# Obs_pred  = Mz_obs_fun( state= state_pred, Nq = Nq)
	# print("Mz_true = \t",Obs_true)
	# print("Mz_pred = \t",Obs_pred)

	# reny2_true = reny2_fun(  state= state, Nq = Nq)
	# reny2_pred  = reny2_fun( state= state_pred, Nq = Nq)
	# print("reny2_true = \t",reny2_true)
	# print("reny2_pred = \t",reny2_pred)

	# reny3_true = reny3_fun(  state= state, Nq = Nq)
	# reny3_pred  = reny3_fun( state= state_pred, Nq = Nq)
	# print("reny3_true = \t",reny3_true)
	# print("reny3_pred = \t",reny3_pred)


	# # state_sim = circuit_Model.inverse(state_f)
	# # print(torch.sum( torch.conj( state_sim.reshape(-1)) * state.reshape(-1) ) )

	
	
	


	# Export_param(param_train=param)

	# Export_endtime( )






