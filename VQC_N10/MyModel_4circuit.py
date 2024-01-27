import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable

from MyModel_1state import *
from MyModel_2gate import *
from MyModel_3rotate import *


def qubitID_fun( q0ID, q1ID, Nq ):
	qubitID = q0ID*(Nq-1)+(q1ID-q0ID-1)%Nq
	return qubitID

def q0_q1_ID_fun( qubitID, Nq):
	q0ID = qubitID//(Nq-1)
	q1ID = ( (qubitID - q0ID*(Nq-1)  )+q0ID+1 ) % (Nq)
	return q0ID, q1ID



def config_circuit_fun(Nc,  Nq, Num_block, Nd=2):
	''' q0 ------U0---|U1|----
	 	q1 ------U0---|++|----
	 	q2 ------U0-----------
	 	q3 -------------------
	 	q4 -------------------
	 	Nq = 5 Nc = 3 
	'''
	UID_block = [0]*Nc + [1]*(Nc-1)
	qID_block = [] 
	for q in range(Nc):
		qID_block.append(q)
	for q in range(Nc-1):
		q0=q
		q1=q+1
		qID= qubitID_fun( q0ID = q0, q1ID=q1 , Nq = Nq)
		qID_block.append(qID)
	UIDset = UID_block*Num_block
	qIDset = qID_block*Num_block
	return UIDset, qIDset 

def Unitary_operation_fun( UID ,qID, theta_param, Nq, state):
	if UID==0:
		U = random_onequbit_gate_fun(theta_para=theta_param)
		state_new = Sigle_qubit_evolution( U =U , q=qID , state=state,  Nq=Nq )
	elif UID==1:
		q0ID, q1ID = q0_q1_ID_fun( qubitID=qID, Nq=Nq)
		U = CNOT_gate()
		state_new = Two_qubits_evolution( U =U , q0=q0ID, q1 =q1ID , state=state,  Nq=Nq)
	return state_new

def Unitary_dagger_operation_fun( UID ,qID, theta_param, Nq, state):
	if UID==0:
		U = random_onequbit_gate_fun(theta_para=theta_param)
		Udagger = torch.conj( torch.transpose( U , 0, 1 ) )
		state_new = Sigle_qubit_evolution( U =Udagger , q=qID , state=state,  Nq=Nq )
	elif UID==1:
		q0ID, q1ID = q0_q1_ID_fun( qubitID=qID, Nq=Nq)
		U = CNOT_gate()
		Udagger = torch.conj( torch.transpose( U , 0, 1 ) )
		state_new = Two_qubits_evolution( U =Udagger , q0=q0ID, q1 =q1ID , state=state,  Nq=Nq)
	return state_new

class circuit(nn.Module):
	def __init__(self, Nc, Nq, Num_block, param ):
		super(circuit,self).__init__()
		self.Nq  = Nq
		self.Num_block = Num_block
		self.param = nn.Parameter(param)
		self.UIDset, self.qIDset = config_circuit_fun(Nc =Nc,  Nq=Nq, Num_block=Num_block)

	def forward(self, state ):
		depth =  len(self.UIDset)
		param = self.param.reshape(depth, 3 )
		for i in range(depth):
			UID = self.UIDset[i]
			qID = self.qIDset[i]
			theta_param = param[i,:]
			state_new = Unitary_operation_fun( UID=UID ,qID=qID, theta_param=theta_param, Nq=self.Nq, state=state)
			state = state_new
		return state

	def inverse(self, state ):
		depth =  len(self.UIDset)
		param = self.param.reshape(depth, 3 )
		for i in range(depth-1,-1,-1):
			UID = self.UIDset[i]
			qID = self.qIDset[i]
			theta_param = param[i,:]
			state_new = Unitary_dagger_operation_fun( UID=UID ,qID=qID, theta_param=theta_param, Nq=self.Nq, state=state)
			state = state_new
		return state



if __name__ == '__main__':
	torch.set_num_threads(1)
	#======================================================
	Nq = 3
	Nc = 3
	Num_block = 3
	depth = (Nc + Nc-1)*Num_block
	param = torch.randn(  depth,  3)*0.01

	# UIDset, qIDset = config_circuit_fun(Nc =Nc,  Nq=Nq, Num_block=Num_block)
	# print(UIDset)
	# print(qIDset)


	circuit_Model = circuit(Nc= Nc, Nq=Nq, Num_block = Num_block, param = param)

	state = GHZ_state_fun(Nq=Nq)
	# state = random_state_fun(Nq =Nq)
	state_new = circuit_Model(state = state)
	c00 = state_new.reshape(-1)[0]
	loss = c00*torch.conj(c00)
	print(state_new)
	print(loss)

	optimizer=torch.optim.Adam(circuit_Model.parameters(),lr=0.001)
	for _ in range(20000):
		state_new = circuit_Model(state = state)
		c00 = state_new.reshape(-1)[0]
		P00 = torch.real(c00*torch.conj(c00) )
		loss = torch.square(P00-1.0)
		print("P", P00, "loss", loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print(state_new.reshape(-1))











	


