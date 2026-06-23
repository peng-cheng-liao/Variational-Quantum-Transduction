import numpy as np
import torch

from QTorch.Transduction import *


Nt = 30
eta = 0.35
depth = 20
parameters = np.load("Job Submission/92/parameters/eta=0.35/parameters_best_feasible.npy")
parameters = torch.tensor(parameters)
print(len(parameters))
print(transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
        eta,
        parameters,
        depth,
        Nt,
        state_initial_RS=None,
        state_initial_PA=None,
        initial_p_thermal_nbar=0.1,
        kappa_o=0.99,
        n_o=0.0,
        kappa_m=0.99,
        n_m=0.0,
        kappa_a=0.99,
        n_a=0.0,))