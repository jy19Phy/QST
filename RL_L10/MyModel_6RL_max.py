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

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
	# Build a feedforward neural network.
	layers = []
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
	return nn.Sequential(*layers)


def Train_policy(  Nq,state,  match_size, batch_size, len_ep  ):
	hidden_sizes=[len_ep]*5
	obs_dim = len_ep+1 
	n_acts = 7*Nq 
	logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
	def get_policy(obs):
		logits = logits_net(obs)
		return Categorical(logits=logits)
	def get_action(obs):
		action= get_policy(obs).sample().item()
		return action
	def compute_loss(obs, act, weights):
		logp = get_policy(obs).log_prob(act)
		return -(logp * weights).mean()
	optimizer = Adam(logits_net.parameters(), lr=1e-2)
	
	def one_episode(len_ep, state):
		gamma = 0.5
		act_I = 0
		action_list = [act_I]*len_ep
		obs, rew  = one_action(state=state, action_list=action_list, Nq=Nq) # the initial observations 
		episode_obs=[]
		episode_act=[]
		episode_rew=[]
		Ret  = -1.0
		for p in range(len_ep): 
			episode_obs.append( obs.reshape(1,-1) )
			random_int = random.randint(1, 10)
			if torch.abs(rew-1.0)<1e-4:
				act = 0
			elif random_int<10:
				act = get_action( obs ) 
			else:
				act = random.randint(1,7*Nq-1)
			episode_act.append(act)
			action_list[p] = act
			obs, rew = one_action(state=state, action_list=action_list, Nq = Nq)
			episode_rew.append(rew)
			if rew>Ret:
				Ret = rew
				index = p 
		episode_return = [Ret]*len_ep
		return episode_obs[0:(index+1)], episode_act[0:(index+1)], episode_return[0:(index+1)], episode_rew[0:(index+1)]



	def partial_max_fun ( batch_obs, batch_act, batch_weights, batch_ret, len_ep, batch_size ) :
		eta = 1 
		N_train = np.int(eta*batch_size)*len_ep
		obs_train_full = torch.cat( batch_obs )
		act_train_full = torch.as_tensor( np.array( batch_act ) )
		wei_train_full = torch.cat( batch_weights )
		sorted_values, sorted_indices = torch.sort(torch.cat( batch_ret ), dim=0, descending=True)
		obs_train = obs_train_full[sorted_indices][0:N_train]
		act_train = act_train_full[sorted_indices][0:N_train]
		wei_train = wei_train_full[sorted_indices][0:N_train]
		# print(wei_train)
		# print(obs_train)
		# print(act_train)
		return obs_train, act_train, wei_train

	def action_list_fun(obs, act):
		obs = obs.reshape(-1)
		action_list = obs
		action_list[-1]=act
		return action_list

	def batch_optimal_fun(batch_obs_last, batch_act_last, batch_rew_last):
		max_index = torch.argmax( torch.cat( batch_rew_last) )
		batch_optimal_rew = batch_rew_last[max_index]
		batch_optimal_act = action_list_fun(batch_obs_last[max_index], batch_act_last[max_index])
		return batch_optimal_rew, batch_optimal_act


	def train_matches(state, Nq , len_ep, batch_size, match_size):
		epoch = 2				# training epoch
		optimal_rew_match =[]
		optimal_act_match=[]
		for match in range(match_size):
			batch_obs = []          # for observations
			batch_act = []         	# for actions
			# batch_rew = [] 
			batch_ret = [] 
			batch_weights = []      # for weighting in policy gradient 

			batch_obs_last = []          # for observations
			batch_act_last = []         	# for actions
			batch_weights_last = []      # for weighting in policy gradient 
			batch_rew_last = []
			for _  in range(batch_size):
				episode_obs, episode_act, episode_ret, episode_rew = one_episode(len_ep,state)
				batch_obs += episode_obs
				batch_act += episode_act
				# batch_rew += episode_rew
				batch_ret += episode_ret
				batch_weights += episode_rew

				batch_obs_last +=  episode_obs[-1]
				batch_act_last += [episode_act[-1]]
				batch_rew_last += episode_rew[-1]
			obs_train, act_train, weights_train = partial_max_fun( batch_obs, batch_act, batch_weights, batch_ret, len_ep, batch_size)
			batch_optimal_rew, batch_optimal_act = batch_optimal_fun( batch_obs_last, batch_act_last, batch_rew_last )
			batch_mean_rew = torch.mean( torch.cat(batch_rew_last ) )

			optimal_rew_match.append(batch_optimal_rew.reshape(1,1))
			optimal_act_match.append(batch_optimal_act.reshape(1,-1))
	
			with open("./TrainRes_batch/batch"+str(batch_size)+"_rew.txt", "a",  buffering=1000000) as file:
				file.write("Nq="+str(Nq)+"\tmatch="+str(match)+"/"+str(match_size-1)
					+"\trew_opt=\t"+str(batch_optimal_rew.reshape(-1).item())
					+'\tret_avg=\t'+str(batch_mean_rew.reshape(-1).item()) 
					+"\n"
					) 
			for epo in range(epoch):	
				optimizer.zero_grad()
				batch_loss = compute_loss(obs=obs_train, act=act_train, weights=weights_train)
				batch_loss.backward()
				print("epoch=", epo, "loss=", batch_loss.item())
				optimizer.step()
				if epo==0:
					with open("./TrainRes_batch/batch"+str(batch_size)+"_act.txt", "a",  buffering=1000000) as file:
						file.write("Nq="+str(Nq)+"\tmatch="+str(match)+"/"+str(match_size-1)
							+"\tloss = "+str(batch_loss.detach().numpy())
							+"\t"
							+str(batch_optimal_act.detach().numpy() )
							+"\n"
							)
			if torch.abs(batch_optimal_rew-1.0)<1e-4:
				break 
		optimal_rew_match = torch.cat(optimal_rew_match)
		optimal_act_match = torch.cat(optimal_act_match)
		max_index = torch.argmax(optimal_rew_match)
		optimal_rew_max = optimal_rew_match[max_index]
		optimal_act_max = optimal_act_match[max_index,:]
		return  optimal_rew_max, optimal_act_max
	
	
	optimal_rew_max, optimal_act_max =  train_matches(state=state, Nq=Nq, len_ep=len_ep,
											batch_size = batch_size, match_size = match_size)
	
	return optimal_rew_max, optimal_act_max


	

	


	





















