import numpy as np
import torch

from main import *

statetype= "_cat_code_states_"
Nt = 20
state_signals = generate_cat_code(1,Nt)
probabilities = [0.5, 0.5]
theta = np.pi / 10
depth = 20
N = 10000  # number of training steps
randomization = 0
disp = True
disp_frequency = 10
save = True
save_frequency = 10
n_constraint = 1
n_coefficient = 10


def penalty(n, n_constraint, ncoefficient):
    n_constraint = torch.tensor(n_constraint, dtype=torch.float64)
    ncoefficient = torch.tensor(ncoefficient)
    penalty1 = ncoefficient * (n - n_constraint) ** 2
    # penalty2 = ncoefficient / (1 + torch.exp(-5 * (n - n_constraint)))
    return penalty1


def target(x):
    rho_out0, n1_input0, state_in0 = transduction_protocol_ECD_QMM(state_signals[0], theta, x, depth, Nt)
    rho_out1, n1_input1, state_in1 = transduction_protocol_ECD_QMM(state_signals[1], theta, x, depth, Nt)
    f0 = state_fidelity(rho_out0, state_signals[0])
    f1 = state_fidelity(rho_out1, state_signals[1])
    fidelity = (f0+f1)/2
    return -fidelity + penalty(n1_input0, n_constraint, n_coefficient), n1_input0, state_in0, f0, f1


def minimize_target(N=N, disp=disp, save=save):
    time1 = datetime.now()
    np.random.seed()
    std_dev = np.sqrt(n_constraint / (2 * depth) / 2)
    alphas = np.random.randn(8 * depth) * std_dev
    betas = np.random.uniform(-np.pi, np.pi, 8 * depth)
    x = np.concatenate((alphas, betas))
    x = torch.tensor(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.05)
    fidelitylist = []
    f0list = []
    f1list = []
    inputenergylist = []
    inputstatelist = []
    xlist = []
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    n_flag = 0
    for nitern in range(N):
        loss, inputenergy, inputstate,f0,f1 = target(x)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(x, 0.01)
        optimizer.step()
        optimizer.zero_grad()

        # save the training data
        fidelity = loss.clone()
        inputenergylist.append(inputenergy.detach().item())
        fidelitylist.append(-(fidelity.detach().item() -
                              penalty(inputenergylist[nitern], n_constraint, n_coefficient).detach().item()))
        f0list.append(f0.detach().numpy())
        f1list.append(f1.detach().numpy())
        inputstatelist.append(inputstate.detach().numpy())
        xlist.append(x.detach().numpy())

        if (nitern % save_frequency == 0 or nitern == N - 1) and save:
            filepath = f"Data/Transduction" + statetype + f"Nt={Nt}_depth={depth}+energy={n_constraint}_N={N}_randomization={randomization}_"
            indexmax = np.argmax(fidelitylist)
            np.save(filepath + "fidelity", fidelitylist)
            np.save(filepath + "f0", f0list)
            np.save(filepath + "f1", f1list)
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
                "fidelity:", (fidelitylist[nitern],f0.detach().item(),f1.detach().item()),
                "input energy:", inputenergylist[nitern],
                "learning rate:", optimizer.param_groups[0]['lr'],
                "running time:", datetime.now() - time1,
                "remaining time:", (datetime.now() - time1) * (N - nitern) / nitern)

    return fidelitylist, inputenergylist, inputstatelist, xlist


minimize_target()
