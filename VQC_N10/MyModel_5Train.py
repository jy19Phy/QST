import math
import numpy as np
import torch
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from MyModel_4circuit import *
from MyModel_1state import *
from MyModel_7Observable import *

def P_sigle_qubit_fun(state, qID,  Nq, Nd = 2):
	indexq = [Nd]*Nq
	state = state.reshape(indexq)
	if qID != Nq-1:
		state_new = torch.transpose( state, qID, Nq-1)
	else:
		state_new = state
	indexq = [Nd**(Nq-1)]+[Nd]	
	state_new = state_new.reshape(indexq)
	rho_00 = torch.sum(state_new[:,0]*torch.conj(state_new[:,0]))
	rho_01 = torch.sum(state_new[:,0]*torch.conj(state_new[:,1]))
	return rho_00, rho_01

def loss_fun(state, qID, Nq):
	P_00, P_01 = P_sigle_qubit_fun(state=state, qID = qID ,  Nq=Nq)
	loss = torch.abs(P_00-1.0)+torch.abs(torch.real(P_01))+torch.abs(torch.imag(P_01))
	loss = torch.abs(P_00-1.0)
	return loss

class MyTrain():
	def __init__(self, func_Train):
		super(MyTrain, self).__init__()
		self.func_Train = func_Train
		self.optimizer=torch.optim.Adam(self.func_Train.parameters(),lr=0.001)
	def forward(self,state, qID, Nq,  train_loss):
		
		epochmax = 10000
		epoch = -1
		loss = 1.0
		while epoch < epochmax and loss>train_loss:
		# while epoch < epochmax :
			epoch = epoch + 1
			state_new = self.func_Train(state)
			loss = loss_fun(state_new, qID,  Nq)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if epoch%1000==0 :
				print("epoch= ", epoch, "\tloss= ",loss.item())
			purity = purity_fun(state= state_new, qID = qID, Nq = Nq)
			if epoch%100==0:
				with open("./TrainResNq"+str(Nq)+"/loss_qID"+str(qID)+".txt", "a+",  buffering=5000) as file:
					file.write(str(epoch)+'\t'+str(loss.item()) +'\t'+str(torch.real(purity).item() ) +'\n')  
		# print("=============================================")
		return loss, purity


# if __name__ == '__main__':
	# torch.set_num_threads(1)
	# #======================================================
	# Nq= 3
	# Num_block = 1
	# depth = (Nq + Nq-1)*Num_block

	# param = torch.randn(  depth,  3)*0.01
	# circuit_Model = circuit(Nq=Nq, Num_block = Num_block, param = param)

	# state = GHZ_state_fun(Nq=Nq)
	# # state = random_state_fun(Nq =Nq)
	# Res = MyTrain(circuit_Model).forward(state)


