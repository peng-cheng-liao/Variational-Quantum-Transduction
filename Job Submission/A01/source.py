from QTorch.Transduction import *
from torch.optim.lr_scheduler import StepLR
from  datetime import  datetime

statetype= "_TMSVEF_MM"
Nt = 20



eta = 0.1
theta = np.arcsin(np.sqrt(eta))
depth = 20
N = 10000  # number of training steps
randomization = 0
disp = True
disp_frequency = 100
save = True
save_frequency = 10

np_constraint = 4
n_coefficient = 10
n_tmsv = torch.tensor(4)
r = torch.arcsinh(torch.sqrt(n_tmsv))


def penalty(n, n_constraint, ncoefficient):
    n_constraint = torch.tensor(n_constraint, dtype=torch.float64)
    ncoefficient = torch.tensor(ncoefficient)
    penalty1 = ncoefficient * (n - n_constraint) ** 2
    return penalty1


def target(x):
    ef, ns_input, np_input, state_RS, state_PA = transduction_protocol_TMSVEF_ECD_MM(r, theta, x, depth, Nt)
    loss = -ef + penalty(np_input, np_constraint, n_coefficient)
    return loss, ef, ns_input, np_input, state_RS, state_PA


def minimize_target(N=N, disp=disp, save=save):
    time1 = datetime.now()
    np.random.seed()
    std_dev = np.sqrt(np_constraint / (2 * depth) / 2)
    alphas = np.random.randn(8 * depth) * std_dev
    betas = np.random.uniform(-np.pi, np.pi, 8 * depth)
    x = np.concatenate((alphas, betas))
    x = torch.tensor(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.05)
    eflist = []
    nplist = []
    statePA_list = []
    xlist = []
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    n_flag = 0
    for nitern in range(N):
        loss, ef, ns_input, np_input, state_RS, state_PA  = target(x)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(x, 0.01)
        optimizer.step()
        optimizer.zero_grad()

        # save the training data
        eflist.append(ef.detach().item())
        nplist.append(np_input.detach().item())


        filepath = f"Data/Transduction" + statetype + f"Nt={Nt}_depth={depth}_eta={eta}_np={np_constraint}_N={N}_randomization={randomization}_"
        if (np.argmax(eflist) == nitern - 1 and nplist[nitern] <= np_constraint + 0.01) and save:
            x_save = x.detach().numpy()
            state_PA_save = state_PA.detach().numpy()
            print("save this data")
            np.save(filepath + "state_PA", state_PA_save)
            np.save(filepath + "parameters", x_save)

        if (nitern % save_frequency == 0 or nitern == N - 1) and save:
            indexmax = np.argmax(eflist)
            np.save(filepath + "ef", eflist)
            np.save(filepath + "np", nplist)


        # adjust the learning rate
        if optimizer.param_groups[0]['lr'] > 1e-6:
            scheduler.step()

        # display the data
        if nitern > 1 and nitern % disp_frequency == 0 and disp:
            print(
                "Nt:", Nt,
                "depth:", depth,
                "np_constraint:", np_constraint,
                "iteration:", nitern,
                "en:", eflist[nitern],
                "np energy:", nplist[nitern],
                "ns energy:", ns_input.detach().item(),
                "learning rate:", optimizer.param_groups[0]['lr'],
                "running time:", datetime.now() - time1,
                "remaining time:", (datetime.now() - time1) * (N - nitern) / nitern)

    return eflist, nplist, statePA_list, xlist


minimize_target()
