import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torch.optim import Adam
from torch.distributions.categorical import Categorical

from MyModel_1state import *
from MyModel_2gate import *
from MyModel_3rotate import *
from MyModel_4act import *
from MyModel_5env import *
from MyModel_6RL_max import *


def Export(Nq, state, match_size, batch_size, len_ep):
	with open("./TrainRes_batch/state_input.txt", "a",  buffering=1000000) as file:
		file.write("Nq= "+str(Nq)+"\n")
		file.write("state=\n \t "+ str(state.reshape(-1))) 
		file.write("\nP0= "+str(P00_fun(state=state, Nq=Nq).reshape(-1))) 
		file.write("\nmatch_size = "+str(match_size)+"\t batch_size= "+str(batch_size)+"\t len_ep= "+str(len_ep)+"\n\n")
	return 0 

def action_list_optimal_fun(state, Nq, match_size = 9876, batch_size= 100, len_ep = 10):
	Export(Nq, state, match_size, batch_size, len_ep)
	optimal_rew_max, optimal_act_max = Train_policy( Nq=Nq, state=state, 
										match_size = match_size, batch_size = batch_size, len_ep=len_ep )
	# np.savetxt("./TrainRes_batch"+str(batch_size)+"/optimal_act_list.csv", optimal_act_max.reshape(1,-1).detach().numpy(),delimiter=',',fmt='%d')
	# np.savetxt("./TrainRes_batch"+str(batch_size)+"/optimal_rew.txt", optimal_rew_max.reshape(-1).detach().numpy(),delimiter=',')
	print("RL P00 = ", optimal_rew_max)
	return optimal_act_max

def state_reduced_fun( state,  Nq, action_list,  Nd =2):
	for a in action_list:
		# print(a, state.reshape(-1))
		state_new = state_after_action_fun(state= state, action_ID = a.long(), Nq = Nq)
		state = state_new
	indexq= [Nd]*Nq
	state = state.reshape(indexq)
	P00=P00_fun(state=state, Nq=Nq)
	state_reduced = torch.sum(state, dim = -1)
	return state_reduced, state_new,  P00


def state_random_action_fun( Nq):
	action_len = 100
	a_min = 1
	a_max = 7*Nq-1
	random_action_list = [random.randint(a_min,a_max) for _ in range(action_len)]
	random_action_list = [21, 32, 13, 2, 19, 3, 17, 14, 20, 21, 25, 26, 27, 2, 17, 1, 1, 12, 18, 1, 25, 34, 21, 20, 10, 21, 28, 8, 21, 32, 3, 1, 34, 7, 23, 3, 29, 19, 3, 29, 18, 12, 33, 34, 12, 6, 18, 23, 34, 3, 21, 32, 31, 15, 14, 29, 17, 23, 8, 13, 33, 34, 3, 12, 5, 17, 34, 29, 10, 11, 3, 24, 10, 19, 22, 34, 28, 24, 5, 18, 34, 13, 6, 9, 23, 23, 30, 26, 26, 26, 26, 25, 13, 28, 3, 5, 12, 10, 22, 27]
	print("random_action_list",random_action_list)
	state = zero_state_fun(Nq)
	for a in random_action_list:
		state_new = state_after_action_fun(state= state, action_ID = a, Nq = Nq)
		state = state_new
	return state,random_action_list


def state_extend_fun( state,  Nq, action_list,  Nd =2):
	for a in action_list[::-1]:
		# print(a, state.reshape(-1))
		state_new = state_before_action_fun(state= state, action_ID = int(a), Nq = Nq)
		state = state_new
	state = state.reshape(-1)
	state_zero = zero_state_fun(Nq = 1).reshape(-1)
	state_extend = torch.kron(state, state_zero)
	return state_extend, state_new



if __name__ == '__main__':
	torch.set_num_threads(1)
	#======================================================

	Nq_full = 5
	# state = GHZ_state_fun(Nq=Nq_full)
	state, random_action_list = state_random_action_fun(Nq = Nq_full)
	# print(random_action_list)
	# state0 = state

	# state = random_state_fun(Nq=Nq_full)
	# state = torch.tensor([ 0.3391+0.1372j, -0.2979-0.7562j, -0.0859-0.0643j, -0.2179-0.3828j])
	state0 = state

	action_set = []
	for Nq in range(Nq_full,0,-1):
		print("Nq=",Nq)
		action_list = action_list_optimal_fun(state = state, Nq = Nq)
		print(action_list)
		state_reduced,state_new, P00 = state_reduced_fun(state=state, Nq= Nq, action_list = action_list)
		action_set.append(action_list.reshape(1,-1))
		with open("./Res/state_output.txt", "a",  buffering=1000000) as file:
			file.write("Nq= "+str(Nq)+"\n")
			file.write("state="+ str(state_new.reshape(-1))) 
			file.write("\nP0= \n"+str(P00_fun(state=state_new, Nq=Nq).reshape(-1))+"\n\n") 
		state = state_reduced
		# print(P00)
	np.savetxt("./Res/action_set.csv", torch.cat(action_set).detach().numpy(),delimiter=',',fmt='%d')


	action_set = np.genfromtxt("./Res/action_set.csv",delimiter=',')
	state = zero_state_fun(Nq=1)
	# state = torch.tensor([-7.0711e-01+7.0711e-01j, -2.9802e-08+8.9407e-08j])
	for Nq in range(1,Nq_full+1,1):
		print("Nq", Nq)
		action_list = action_set[Nq_full-Nq]
		state_extend, state_new = state_extend_fun( state=state,  Nq=Nq, action_list=action_list)
		# print(state_new.reshape(-1))
		state = state_extend
	# print(state_new.reshape(-1))
	state_pred = state_new 
	
	Fidelity = torch.sum( torch.conj( state_pred.reshape(-1)) * state0.reshape(-1) )
	state_F = Fidelity
	print("state_Fidelity", Fidelity)
	rho_F  = Fidelity*torch.conj(Fidelity)
	print("rho_Fedelity", Fidelity*torch.conj(Fidelity))






	





















