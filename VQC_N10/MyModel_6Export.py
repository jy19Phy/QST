import math
import numpy as np
import torch
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from datetime import datetime

def Export_SystemSetting(Nq,  Name_state0):
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d %H:%M:%S")
	with open("./TrainResNq"+str(Nq)+"/System.txt", "a+",  buffering=1000000) as file:
		file.write("\n"+"Running begins at \t"+time_string)
		file.write('\nnumber of qubits (Nq) =\t'+str(Nq) )
		file.write('\ninitial state = \t'+str(Name_state0))
	return 0

def Export_endtime(Nq,  state_F, rho_F ):
	now = datetime.now()
	time_string = now.strftime("%Y-%m-%d %H:%M:%S")
	with open("./TrainResNq"+str(Nq)+"/System.txt", "a+",  buffering=1000000) as file:
		file.write("\nstate Fidelity  \t"+str( state_F.detach().numpy()  )  )
		file.write("\nrho   Fidelity  \t"+str( rho_F.detach().numpy() )  )
		file.write("\nRunning ends at \t"+time_string+"\n\n")
	return 0


def Export_param(Nq, state0, param_train, stateSet,  stateInvSet, Pset):
	torch.save(state0.reshape(-1), "./TrainResNq"+str(Nq)+"/state0.pt")
	torch.save(param_train, "./TrainResNq"+str(Nq)+"/param_train.pt")
	np.savetxt("./TrainResNq"+str(Nq)+"/state0.csv", state0.reshape(-1).detach().numpy(),delimiter=',')
	np.savetxt("./TrainResNq"+str(Nq)+"/param_train.csv", torch.cat(param_train).detach().numpy(),delimiter=',')
	np.savetxt("./TrainResNq"+str(Nq)+"/stateSet.csv", torch.cat(stateSet).detach().numpy(),delimiter=',')
	np.savetxt("./TrainResNq"+str(Nq)+"/stateInvSet.csv", torch.cat(stateInvSet ).detach().numpy(),delimiter=',')
	np.savetxt("./TrainResNq"+str(Nq)+"/Pset.csv", torch.cat(Pset).detach().numpy(),delimiter=',')


# if __name__ == '__main__':
# 	Nq = 10
# 	qubitIndex= np.arange( 0, Nq) 
# 	qubitspin = []
# 	for q in qubitIndex:
# 		spin = np.base_repr(qubitIndex[q], 2)
# 		print(len(spin))
# 		padding = Nq-len(spin)
# 		spin_Nq = np.base_repr(qubitIndex[q], 2, padding = padding)
# 		qubitspin.append(spin_Nq)
# 	print(qubitspin)
# 	state0 = np.arange(0,Nq)

# 	Res = np.stack( (qubitspin , state0) )
# 	print(Res)






