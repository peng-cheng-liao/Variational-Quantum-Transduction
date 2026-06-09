import numpy as np
import torch

from main import *

statetype= "_cat_code_states_"
Nt = 20
state_signals = generate_cat_code(2, Nt)

eta = 0.1
theta = np.arcsin(np.sqrt(eta))
depth = 20
N = 10000  # number of training steps
randomization = 0
disp = True
disp_frequency = 100
save = True
save_frequency = 1
n_constraint = 4
n_coefficient = 10


def penalty(n, n_constraint, ncoefficient):
    n_constraint = torch.tensor(n_constraint, dtype=torch.float64)
    ncoefficient = torch.tensor(ncoefficient)
    penalty1 = ncoefficient * (n - n_constraint) ** 2
    # penalty2 = ncoefficient / (1 + torch.exp(-5 * (n - n_constraint)))
    return penalty1


def target(x):
    ef, n1_input0, state_in0 = transduction_protocol_ECD_M_EF(state_signals, theta, x, depth, Nt)
    fidelity = ef
    return -fidelity + penalty(n1_input0, n_constraint, n_coefficient), n1_input0, state_in0


def minimize_target(N=N, disp=disp, save=save):
    time1 = datetime.now()
    np.random.seed()
    std_dev = np.sqrt(n_constraint / (2 * depth) / 2)
    alphas = np.random.randn(4 * depth) * std_dev
    betas = np.random.uniform(-np.pi, np.pi, 4 * depth)
    x = np.concatenate((alphas, betas))
    x = torch.tensor(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.05)
    fidelitylist = []
    inputenergylist = []
    inputstatelist = []
    xlist = []
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    n_flag = 0
    for nitern in range(N):
        loss, inputenergy, inputstate = target(x)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(x, 0.01)
        optimizer.step()
        optimizer.zero_grad()

        # save the training data
        fidelity = loss.clone()
        inputenergylist.append(inputenergy.detach().item())
        fidelitylist.append(-(fidelity.detach().item() -
                              penalty(inputenergylist[nitern], n_constraint, n_coefficient).detach().item()))

        inputstatelist.append(inputstate.detach().numpy())
        xlist.append(x.detach().numpy())

        if (nitern % save_frequency == 0 or nitern == N - 1) and save:
            filepath = f"Data/Transduction" + statetype + f"Nt={Nt}_depth={depth}_eta={eta}_energy={n_constraint}_N={N}_randomization={randomization}_"
            indexmax = np.argmax(fidelitylist)
            np.save(filepath + "fidelity", fidelitylist)
            np.save(filepath + "energy", inputenergylist)
            np.save(filepath + "inputstate", inputstatelist[indexmax])
            np.save(filepath + "xlist", xlist[indexmax])

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
                "fidelity:", fidelitylist[nitern],
                "input energy:", inputenergylist[nitern],
                "learning rate:", optimizer.param_groups[0]['lr'],
                "running time:", datetime.now() - time1,
                "remaining time:", (datetime.now() - time1) * (N - nitern) / nitern)

    return fidelitylist, inputenergylist, inputstatelist, xlist


minimize_target()
