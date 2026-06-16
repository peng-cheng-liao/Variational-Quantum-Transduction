import torch

from QTorch.Transduction import *
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import os
torch.autograd.set_detect_anomaly(True)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

statetype = "_CoherentInfo_GKP_"
Nt = 30
N = 5000  # number of training steps
disp = True
disp_frequency = 100
save = True
save_frequency = 10
ns_constraint = 2
np_constraint = 2
n_coefficient = 10



eta = float(os.environ.get("ETA", "0.10"))
randomization = int(os.environ.get("RANDOMIZATION", "0"))
d1 = int(os.environ.get("D1", "2"))
d2 = int(os.environ.get("D2", "1"))
j2 = int(os.environ.get("J2", "0"))

NR = d1
#filepath = f"Data/Transduction" + statetype + f"Nt={Nt}_d1={d1}_d2={d2}_j2={j2}_eta={eta}_ns={ns_constraint}_np={np_constraint}_N={N}_randomization={randomization}_"
#run_dir = os.environ.get("RUN_DIR", filepath)


# directory: grouped by eta, d1, d2
dirpath = (f"Data/"f"eta={eta}_d1={d1}_d2={d2}/")
os.makedirs(dirpath, exist_ok=True)
# filename: j2 and randomization moved here
filename = (
    f"j2={j2}"
    f"_randomization={randomization}_")
    
filepath = os.path.join(dirpath, filename)



def penalty_energy(n, n_constraint, ncoefficient):
    n_constraint = torch.tensor(n_constraint, dtype=torch.float64)
    ncoefficient = torch.tensor(ncoefficient)
    penalty = ncoefficient * (n - n_constraint) ** 2
    return penalty


def target(x):
    CI, ns_input, np_input, state_RS, state_P = transduction_protocol_CoherentInfo_GKP2(eta, d1, d2, j2, x, Nt,NR)
    loss = (-CI +
            penalty_energy(ns_input, ns_constraint, n_coefficient)+
            penalty_energy(np_input, np_constraint, n_coefficient))
    return loss, CI, ns_input, np_input, state_RS, state_P


def minimize_target(N=N, disp=disp, save=save):
    time1 = datetime.now()
    x = torch.rand(8, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.001)
    ci_list = []
    ns_list = []
    np_list = []
    xlist = []
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)
    for nitern in range(N):
        loss, CI, ns_input, np_input, state_RS, state_P = target(x)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(x, 0.01)
        optimizer.step()
        optimizer.zero_grad()

        # save the training data
        ci_list.append(CI.detach().item())
        ns_list.append(ns_input.detach().item())
        np_list.append(np_input.detach().item())



        if (save and np.argmax(ci_list) == nitern - 1 and ns_list[nitern] <= ns_constraint + 0.05 and np_list[nitern] <= np_constraint + 0.05 ):
            state_RS_save = state_RS.detach().numpy()
            state_P_save = state_P.detach().numpy()
            x_save = x.detach().numpy()
            print("save this data")
            np.save(filepath + "state_RS", state_RS_save)
            np.save(filepath + "state_P", state_P_save)
            np.save(filepath + "parameters", x_save)

        if (nitern % save_frequency == 0 or nitern == N - 1) and save:

            np.save(filepath + "ci_list", ci_list)
            np.save(filepath + "ns_list", ns_list)
            np.save(filepath + "np_list", np_list)


        # adjust the learning rate
        if optimizer.param_groups[0]['lr'] > 1e-6:
            scheduler.step()

        # display the data
        if nitern > 1 and nitern % disp_frequency == 0 and disp:
            print(
                "Nt:", Nt,
                "eta:", eta,
                "ns_constraint:", ns_constraint,
                "np_constraint:", np_constraint,
                "iteration:", nitern,
                "ci:", ci_list[nitern],
                "ns:", ns_list[nitern],
                "np:", np_list[nitern],
                "learning rate:", optimizer.param_groups[0]['lr'],
                "running time:", datetime.now() - time1,
                "remaining time:", (datetime.now() - time1) * (N - nitern) / nitern)

    return ci_list, np_list, xlist


minimize_target()
