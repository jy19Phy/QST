import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch
import resource

def get_max_memory_usage():
	# 在你的程序中执行需要监测内存的代码
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss 返回的单位是kilobytes（KB）
    return max_memory / 1024 /1024 # 转换为 megabytes (GB)

# Nd = 2: number of degrees of freedom for single qubit
# Nq: number of qubits

def GHZ_state_fun(Nq ,  Nd= 2 ):
	WF = torch.zeros(Nd**Nq)*0.0j
	WF[0] = 1.0/np.sqrt(2.0)
	WF[-1] = 1.0/np.sqrt(2.0)
	index = [Nd]*Nq
	WF = WF.reshape(index)
	# index0 = [0]*Nq
	# print(index0, " ", 	WF[tuple(index0) ] )
	# index1 = [1]*Nq 
	# print(index1, " ", 	WF[tuple(index1) ] )
	return WF

def zero_state_fun( Nq, Nd=2 ):
	WF = torch.zeros(Nd**Nq)*0.0j
	WF[0] =1.0
	index = [Nd]*Nq 
	WF = WF.reshape(index)
	return WF

def one_state_fun( Nq, Nd=2):
	WF = torch.zeros(Nd**Nq)*0.0j
	WF[-1] =1.0
	index = [Nd]*Nq 
	WF = WF.reshape(index)
	return WF

def Computational_stateSet_fun(Nq,Nd=2):
	WFSet = torch.zeros(Nd**Nq, Nd**Nq)*0.0j
	for index in range(Nd**Nq):
		WFSet[index,index] =1.0
	return WFSet

def random_state_fun( Nq, Nd=2):
	WF = torch.randn(Nd**Nq)+torch.randn(Nd**Nq)*1.0j
	Nor= torch.real(torch.sum(torch.conj(WF)*WF))
	WFNor= WF/torch.sqrt(Nor)
	return WFNor





if __name__ == '__main__':
	torch.set_num_threads(1)
	
	Nd = 2
	Nq = 4
	# state = GHZ_state_fun( Nq , Nd  )
	# state = GHZ_state_fun( Nq=Nq , Nd= Nd  )
	# state = GHZ_state_fun( Nq )	

	# print( state.shape )
	# print("Max Memory Usage:", get_max_memory_usage(), "GB")

	# stateSet = Computational_stateSet_fun(Nq = Nq)

	# for WFID in range( 2**Nq):
	# 	state = stateSet[WFID,:]
	# 	print(state)

	state = random_state_fun( Nq= Nq )	
	print("random_state: ", state)



		

