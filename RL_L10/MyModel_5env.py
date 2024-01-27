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
from MyModel_4act import *

def P00_fun(state, Nq, Nd=2):
	if Nq>1:
		indexq = [Nd**(Nq-1)]+[Nd]	
		state = state.reshape(indexq)
		P_00 = torch.real(torch.sum(state[:,0]*torch.conj(state[:,0])) )
	elif Nq==1:	
		state = state.reshape(-1)
		P_00 = torch.real(state[0]*torch.conj(state[0]) )
	return P_00.reshape(-1)

def one_action(state, action_list, Nq):
	P00_initial = P00_fun(state = state, Nq = Nq).reshape(1,1)
	for action_ID in action_list:
		state_new = state_after_action_fun(state=state, action_ID=action_ID, Nq=Nq)
		state = state_new
	P00 = P00_fun(state = state, Nq = Nq).reshape(1,1)
	action_list_tensor = torch.as_tensor( np.array(action_list) ).reshape(1,-1)
	obs = torch.cat( (action_list_tensor,P00), dim = -1 ).reshape(1,-1)
	rew = P00 
	return obs,rew


if __name__ == '__main__':
	torch.set_num_threads(1)
	#======================================================
	Nq = 2
	state = GHZ_state_fun(Nq=Nq)
	












	


