import numpy as np

from QTorch.Transduction import *
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


statetype = "_CoherentInfo_EATele_M_"
Nt = 30

eta = 0.1
theta = np.arcsin(np.sqrt(eta))
depth = 20
N = 10000  # number of training steps
randomization = 0
disp = True
disp_frequency = 100
save = True
save_frequency = 10
ns_constraint = 2
np_constraint = 2
n_coefficient = 10
filepath = f"Data/Transduction" + statetype + f"Nt={Nt}_depth={depth}_eta={eta}_ns={ns_constraint}_np={np_constraint}_N={N}_randomization={randomization}_"


def penalty_energy(n, n_constraint, ncoefficient):
    n_constraint = torch.tensor(n_constraint, dtype=torch.float64)
    ncoefficient = torch.tensor(ncoefficient)
    penalty = ncoefficient * (n - n_constraint) ** 2
    return penalty


def target(x):
    CI, ns_input, np_input, state_RS, state_P = transduction_protocol_CoherentInfo_ECD_M_EATele(theta, x, depth, Nt)
    loss = (-CI +
            penalty_energy(ns_input, ns_constraint, n_coefficient) +
            penalty_energy(np_input, np_constraint, n_coefficient))
    return loss, CI, ns_input, np_input, state_RS, state_P


def minimize_target(N=N, disp=disp, save=save):
    global state_RS_save, state_P_save, x_save
    time1 = datetime.now()
    np.random.seed()
    std_dev = np.sqrt(ns_constraint / (2 * depth) / 2)
    alphas = np.random.randn(10 * depth) * std_dev
    betas = np.random.uniform(-np.pi, np.pi, 10 * depth)
    x = np.concatenate((alphas, betas, np.random.rand(1)))
    x = torch.tensor(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.01)
    ci_list = []
    ns_list = []
    np_list = []
    xlist = []
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
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
        xlist.append(x.detach().numpy())

        if (np.argmax(ci_list) == nitern - 1 and
                ns_list[nitern] <= ns_constraint + 0.05 and np_list[nitern] <= np_constraint + 0.05):
            x_save = x.detach().numpy()
            state_RS_save = state_RS.detach().numpy()
            state_P_save = state_P.detach().numpy()
            print("save this data")
            np.save(filepath + "state_RS", state_RS_save)
            np.save(filepath + "state_P", state_P_save)
            np.save(filepath + "parameters", x_save)

        if (nitern % save_frequency == 0 or nitern == N - 1) and save:
            np.save(filepath + "ci_list", ci_list)
            np.save(filepath + "np_list", np_list)
            np.save(filepath + "ns_list", ns_list)


        # adjust the learning rate
        if optimizer.param_groups[0]['lr'] > 1e-6:
            scheduler.step()

        # display the data
        if nitern > 1 and nitern % disp_frequency == 0 and disp:
            print(
                "Nt:", Nt,
                "depth:", depth,
                "eta:", eta,
                "ns_constraint:", ns_constraint,
                "np_constraint:", np_constraint,
                "iteration:", nitern,
                "ci:", ci_list[nitern],
                "np:", np_list[nitern],
                "ns:", ns_list[nitern],
                "learning rate:", optimizer.param_groups[0]['lr'],
                "running time:", datetime.now() - time1,
                "remaining time:", (datetime.now() - time1) * (N - nitern) / nitern)

    return ci_list, np_list, xlist


minimize_target()
