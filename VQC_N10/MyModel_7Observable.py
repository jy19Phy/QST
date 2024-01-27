import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 
from MyModel_3rotate import *


def Mz_marix_fun( Nd= 2):
	sigma_z = torch.tensor([[1.0+0.0j, 0.0+0.0j],  [0.0+0.0j,-1.0+0.0j]])
	return sigma_z

def Mx_marix_fun( Nd= 2):
	sigma_x = torch.tensor([[0.0+0.0j, 1.0+0.0j],  [1.0+0.0j, 0.0+0.0j]])
	return sigma_x

def Mz_obs_fun( state , Nq):
	U = Mz_marix_fun()
	q = 0
	state_new = Sigle_qubit_evolution( U =U , q=q , state=state,  Nq =Nq)
	Obs = torch.sum ( torch.conj(state.reshape(-1) ) *state_new.reshape(-1) )
	return Obs

def Mx_obs_fun( state , Nq):
	U = Mx_marix_fun()
	q = 0
	state_new = Sigle_qubit_evolution( U =U , q=q , state=state,  Nq =Nq)
	Obs = torch.sum (state.reshape(-1)*state_new.reshape(-1) )
	return Obs

def rho_fun(state, Nq, Nd = 2):
	state = state.reshape(Nd**Nq)
	rho = torch.kron( state, torch.conj(state) )
	indexq = [Nd**Nq]+[Nd**Nq]
	rho= rho.reshape(indexq)
	return rho

def rho_partial_fun(state, Nq, Nd =2):
	state = state.reshape(Nd**Nq)
	rho = torch.kron( state, torch.conj(state) )
	nA = 1
	nAbar = Nq-1
	indexq = [Nd**nAbar]+[Nd**nA]+[Nd**nAbar]+[Nd**nA]
	rho = rho.reshape( indexq )
	rho_partial = torch.einsum( 'ijik-> jk',rho )
	rho_partial = rho_partial.reshape(Nd**nA,Nd**nA)
	return rho_partial


def rhoA_partial_fun(state, Nq, Nd =2):
	state = state.reshape(Nd**Nq)
	rho = torch.kron( state, torch.conj(state) )
	nA = Nq-1
	nB = 1
	indexq = [Nd**nA]+[Nd**nB]+[Nd**nA]+[Nd**nB]
	rho = rho.reshape( indexq )
	rho_partial = torch.einsum( 'ijkj-> ik',rho )
	rho_partial = rho_partial.reshape(Nd**nA,Nd**nA)
	return rho_partial


def rhoB_partial_fun(state, Nq, Nd =2):
	state = state.reshape(Nd**Nq)
	rho = torch.kron( state, torch.conj(state) )
	nA = Nq-1
	nB = 1
	indexq = [Nd**nA]+[Nd**nB]+[Nd**nA]+[Nd**nB]
	rho = rho.reshape( indexq )
	rho_partial = torch.einsum( 'ijik-> jk',rho )
	rho_partial = rho_partial.reshape(Nd**nB,Nd**nB)
	return rho_partial



def reny2_fun(state, Nq):
	rho_partial= rho_partial_fun(state = state, Nq = Nq)
	Exp_reny2 = torch.einsum('ij,ji->',rho_partial,rho_partial) 
	return Exp_reny2

def reny3_fun(state, Nq):
	rho_partial= rho_partial_fun(state = state, Nq = Nq)
	Exp_reny3 = torch.einsum('ij,jk,ki->',rho_partial,rho_partial,rho_partial) 
	return Exp_reny3


def purity_fun( state, qID, Nq , Nd =2 ):
	indexq = [Nd]*Nq
	state = state.reshape(indexq)
	if qID != 0 :
		state = torch.transpose( state, 0 , qID)
	state = state.reshape(Nd, Nd**(Nq-1))
	rho_qID = torch.einsum('ij,kj->ik', state, torch.conj(state) )
	# print(rho_qID)
	# purity = torch.einsum( 'ii', torch.einsum('ij, jk-> ik', rho_qID, rho_qID) )
	purity = torch.einsum('i j, j i->', rho_qID, rho_qID)
	return purity


# if __name__ == '__main__':
# 	Nq = 2
# 	state = zero_state_fun(Nq = Nq)
# 	qID = 1
# 	res = purity_fun (state = state , qID = qID, Nq = Nq)
# 	print( res)

# 	x = torch.tensor([[1.+0.j, 0.+0.j],[0.+0.j, 0.+0.j]])
	# y = torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]])
	# x = torch.tensor([[1.,0.],[0., 0.]])
	# y = torch.tensor([[1.,0.],
    #     [0., 0.]])
	# print(torch.matmul(x,y))

#	 Compute the trace of the matrix product
	# result = torch.einsum('i j, j i', x, x)

	# print(result)




