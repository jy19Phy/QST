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


def UID_set_fun( Nq):
	# action_ID   	UID		gate 	qubit 
	# 				-1 		I 		
	# 	  			0	 	H 		Nq
	#  	 			1	 	T		Nq
	# 	 			2 		S 		Nq
	# 				3 		Px 		Nq
	# 				4 		Py		Nq
	# 				5		Pz		Nq
	# 				6 		CNOT	Nq-1
	 		
	UID_set =[-1] + [0]*Nq + [1]*Nq +[2]*Nq +[3]*Nq + [4]*Nq + [5]*Nq+ [6]*(Nq-1)
	# print(UID_set )
	return UID_set

def UID_qID_fun(action_ID, Nq):
	# the target qubit is at last. 
	UID_set = UID_set_fun(Nq=Nq)
	UID = UID_set[action_ID]
	UID_set_part = UID_set[:action_ID+1]
	qID = UID_set_part.count(UID)-1
	return UID, qID

def state_after_action_fun(state, action_ID, Nq):
	target_qID = Nq-1
	UID, qID = UID_qID_fun(action_ID=action_ID, Nq=Nq) 

	if UID == -1: 	# U = I 
		state_new = state	
	if UID==0:			# U = H
		U = H_gate()
		# print("H",qID)
	if UID==1:			# U = T
		U = T_gate()
		# print("T",qID)
	if UID==2:			# U = S
		U = S_gate()
		# print("S",qID)
	if UID==3:			# U = Px
		U = Px_gate()
		# print("Px",qID)
	if UID ==4:			# U = Py
		U = Py_gate()
		# print("Py",qID)
	if UID ==5:			# U = Pz
		U = Pz_gate()
		# print("Pz",qID)

	if UID != 6 and UID!= -1 :
		state_new = Sigle_qubit_evolution( U =U , q=qID , state=state,  Nq=Nq )
		if qID != target_qID:
			# print(" action_ID ",action_ID, UID, qID)
			state = state_new
			U = CNOT_gate()
			state_new = Two_qubits_evolution( U =U , q0=qID, q1 = target_qID , state=state,  Nq=Nq)
	elif UID ==6:
		# print("CNOT",qID)
		U = CNOT_gate()
		state_new = Two_qubits_evolution( U =U , q0=qID, q1 = target_qID , state=state,  Nq=Nq)

	return state_new 

def state_before_action_fun(state, action_ID, Nq, Nd=2):
	target_qID = Nq-1
	UID, qID = UID_qID_fun(action_ID=action_ID, Nq=Nq) 

	if UID == -1: 	# U = I 
		state_new = state	
	if UID==0:			# U = H
		U = H_gate()
		# print("H",qID)
	if UID==1:			# U = T
		U = T_gate()
		# print("T",qID)
	if UID==2:			# U = S
		U = S_gate()
		# print("S",qID)
	if UID==3:			# U = Px
		U = Px_gate()
		# print("Px",qID)
	if UID ==4:			# U = Py
		U = Py_gate()
		# print("Py",qID)
	if UID ==5:			# U = Pz
		U = Pz_gate()
		# print("Pz",qID)

	if UID != 6 and UID!= -1 :
		U = U.reshape(Nd, Nd)
		Udagger = torch.transpose(torch.conj(U), 0, 1)
		if qID != target_qID:
			# print(" action_ID ",action_ID, UID, qID)
			U = CNOT_gate()
			state_new = Two_qubits_evolution( U =U , q0=qID, q1 = target_qID , state=state,  Nq=Nq)
			state = state_new
		state_new = Sigle_qubit_evolution( U =Udagger , q=qID , state=state,  Nq=Nq )
	elif UID ==6:
		# print("CNOT",qID)
		U = CNOT_gate()
		state_new = Two_qubits_evolution( U =U , q0=qID, q1 = target_qID , state=state,  Nq=Nq)

	return state_new 



if __name__ == '__main__':
	torch.set_num_threads(1)
	#======================================================
	Nq = 3
	action_ID = 7 
	UID, qID= UID_qID_fun( action_ID=action_ID, Nq=Nq)
	print("action_ID", action_ID)
	print("UID",UID)
	print("qID",qID)
	state = GHZ_state_fun(Nq=Nq)
	# print(state)
	state_new = state_after_action_fun(state=state, action_ID=action_ID, Nq=Nq)
	print(state_new)

	# for action_ID in range(0, 7*Nq):
	# 	state = GHZ_state_fun(Nq=Nq)
	# 	state_new = state_after_action_fun(state=state, action_ID=action_ID, Nq=Nq)
	# 	print(state_new)
















	


