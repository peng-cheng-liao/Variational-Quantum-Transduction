import numpy as np
import torch

from main import *

statetype= "_train_encoding_"
Nt = 20


eta = 0.1
theta = np.arcsin(np.sqrt(eta))
depth = 20
N = 10000  # number of training steps
randomization = 0
disp = True
disp_frequency = 100
save = True
save_frequency = 1
n_constraint = 5


def energy_penalty_qform(n, n_constraint, k=10):
    n_constraint = torch.tensor(n_constraint, dtype=torch.float64)
    k = torch.tensor(k)
    penalty = k * (n - n_constraint) ** 2
    return penalty

def energy_penalty_linear_heavside(n,n_constraint,k=0.05):
    if n > n_constraint:
        penalty = 100+k*n
    else:
        penalty = k * n
    return penalty

def target(x):
    parameters_signals = x[0:4*depth]
    parameters = x[4*depth:16*depth]
    ef, np_input, ns, state_in_P, l0, l1, encoding_fidelity,entropy_SQ2 = transduction_protocol_ECD_M_TE_EF_QMMDecoding(parameters_signals, theta, parameters, depth, Nt)
    fidelity = ef
    loss = (-fidelity
            + energy_penalty_linear_heavside(np_input, n_constraint)
            + energy_penalty_linear_heavside(ns, n_constraint)
            + 10 * encoding_fidelity)
            #- 10 * entropy_SQ2)
    return loss, ef, np_input, ns, state_in_P, l0, l1, encoding_fidelity, entropy_SQ2


def minimize_target(N=N, disp=disp, save=save):
    time1 = datetime.now()
    np.random.seed()
    std_dev = np.sqrt(n_constraint / (2 * depth) / 2)
    alphas_encoding = np.random.randn(2 * depth) * std_dev
    betas_encoding  = np.random.uniform(-np.pi, np.pi, 2 * depth)

    alphas = np.random.randn(6 * depth) * std_dev
    betas = np.random.uniform(-np.pi, np.pi, 6 * depth)
    x = np.concatenate((alphas_encoding,betas_encoding,alphas, betas))
    x = torch.tensor(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.05)
    fidelitylist = []
    inputenergylist = []
    inputstatelist = []
    nslist =[]
    l0list = []
    l1list =[]
    encoding_fidelity_list = []
    xlist = []
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    n_flag = 0
    for nitern in range(N):
        loss, ef, np_input, ns, state_in_P, l0,l1, encoding_fidelity, entropy_SQ2= target(x)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(x, 0.01)
        optimizer.step()
        optimizer.zero_grad()

        # save the training data
        fidelitylist.append(ef.detach().item())
        inputenergylist.append(np_input.detach().item())
        nslist.append(ns.detach().item())
        inputstatelist.append(state_in_P.detach().numpy())
        xlist.append(x.detach().numpy())
        l0list.append(l0.detach().numpy())
        l1list.append(l1.detach().numpy())
        encoding_fidelity_list.append(encoding_fidelity.detach().item())

        if (nitern % save_frequency == 0 or nitern == N - 1) and save:
            filepath = f"Data/Transduction" + statetype + f"Nt={Nt}_depth={depth}_eta={eta}_energy={n_constraint}_N={N}_randomization={randomization}_"
            indexmax = np.argmax(fidelitylist)
            np.save(filepath + "fidelity", fidelitylist)
            np.save(filepath + "energy_np", inputenergylist)
            np.save(filepath + "energy_ns", nslist)
            np.save(filepath + "encoding_fidelity", encoding_fidelity_list)

            np.save(filepath + "inputstate", inputstatelist[indexmax])
            np.save(filepath + "xlist", xlist[indexmax])
            np.save(filepath + "l0", l0list[indexmax])
            np.save(filepath + "l1", l1list[indexmax])

        # adjust the learning rate
        if optimizer.param_groups[0]['lr'] > 1e-6:
            scheduler.step()

        # display the data
        if nitern > 1 and nitern % disp_frequency == 0 and disp:
            print(
                "Nt:", Nt,
                "depth:", depth,
                "n_constraint:", n_constraint,
                "iteration:", nitern,
                "ef:", fidelitylist[nitern],
                "np:", inputenergylist[nitern],
                "ns:", nslist[nitern],
                "encoding_fidelity:", encoding_fidelity_list[nitern],
                "entropy_SQ2:", entropy_SQ2.detach().item(),
                "learning rate:", optimizer.param_groups[0]['lr'],
                "running time:", datetime.now() - time1,
                "remaining time:", (datetime.now() - time1) * (N - nitern) / nitern)

    return fidelitylist, inputenergylist, inputstatelist, xlist


minimize_target()
