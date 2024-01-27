import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 
from MyModel_1state import GHZ_state_fun
from MyModel_2gate import *
from MyModel_5Train import *

def Sigle_qubit_evolution( U  , q , state,  Nq , Nd=2):
	Indexq = [Nd]*Nq 
	state = torch.reshape(state, Indexq)
	U=U.reshape(Nd,Nd)
	if q!=0:
		state = torch.transpose(state, 0, q )
	state = state.reshape( Nd,Nd**(Nq-1)) 
	state_new = torch.matmul(U,state)
	state_new = torch.reshape( state_new, Indexq )
	if q!=0:
		state_new = torch.transpose( state_new, 0,q )	
	return state_new

def Two_qubits_evolution( U , q0 , q1 , state, Nq , Nd=2):
	Indexq = [Nd]*Nq 
	state = torch.reshape(state, Indexq)
	U = U.reshape( Nd, Nd , Nd, Nd )
	if q0>q1:
		U = torch.transpose( U,  0, 1)
		U = torch.transpose( U,  2, 3)
		q = q0
		q0= q1
		q1= q
	U = U.reshape( Nd*Nd , Nd*Nd )
	if q0!=0: 
		state = torch.transpose( state, 0, q0 )
	if q1!=1:
		state = torch.transpose( state, 1, q1 )
	state_new = torch.matmul( U, state.reshape(Nd**2 , -1) )
	state_new = torch.reshape( state_new, Indexq)
	if q1!=1:
		state_new= torch.transpose( state_new, 1, q1)
	if q0!=0:
		state_new= torch.transpose( state_new, 0, q0)
	return state_new


if __name__ == '__main__':
	torch.set_num_threads(1)
	#======================================================
	Nq= 3
	state = GHZ_state_fun(Nq = Nq)
	print(state.reshape(-1))

	
	U = random_onequbit_gate_fun(theta_para=torch.randn(3), Np=3, Nd=2)
	state_new = Sigle_qubit_evolution( U =U , q=0 , state=state,  Nq=Nq )
	print(state_new)

	U = CNOT_gate()
	state_new = Two_qubits_evolution( U =U , q0=0, q1 =1 , state=state,  Nq=Nq)
	print(state_new.reshape(-1))
	P_0, _ = P_sigle_qubit_fun(state=state, qID= 2, Nq=Nq)
	print("P_Nq0 =\t", P_0 ) 
	P_0, _ = P_sigle_qubit_fun(state=state_new, qID= 2, Nq=Nq)
	print("P_Nq0 =\t", P_0 ) 
	print(state_new.reshape(2,2,2 ))
	print(state_new[0,0,0])
	print(state_new[0,0,1])
	print(state_new[0,1,0])
	print(state_new[0,1,1])
	print(state_new[1,0,0])
	print(state_new[1,0,1])
	print(state_new[1,1,0])
	print(state_new[1,1,1])
	c0 = state_new.reshape(2**2,2)[:,0]
	print(c0)
	print(torch.sum( c0* torch.conj(c0)) )











	


