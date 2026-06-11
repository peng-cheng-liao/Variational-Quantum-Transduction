import matplotlib.pyplot as plt
import os

import numpy as np

from main import *
from matplotlib import rc
from Quantum_Plotting import *

rc('text', usetex=True)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
n_s = 3
n_p = 3
energy_tol = 0.05


# ECD MM

def plot_ECD_MM():
    Nr = 20
    etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
    CI_matrix = np.zeros((len(etalist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                ci_history = np.load(parent_dir + f"/Data_HPC/68/Transduction_CoherentInfo_EA_Nt=30_depth=20_eta={eta}_ns=3_np=3_N=10000_randomization={j}_ci_list.npy")
                ns_history = np.load(
                    parent_dir + f"/Data_HPC/68/Transduction_CoherentInfo_EA_Nt=30_depth=20_eta={eta}_ns=3_np=3_N=10000_randomization={j}_ns_list.npy")
                np_history = np.load(
                    parent_dir + f"/Data_HPC/68/Transduction_CoherentInfo_EA_Nt=30_depth=20_eta={eta}_ns=3_np=3_N=10000_randomization={j}_np_list.npy")
                ci_history[np.isnan(ci_history)] = 0
                mask = np.logical_or(ns_history >= n_s+energy_tol, np_history >= n_s+energy_tol)
                ci_history[mask] = -10
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("ECD MM EA", eta, j, "fail")
    ci_list1 = np.max(CI_matrix, 1)


    etalist_str = [f"{eta:.2f}" for eta in etalist]
    ci_matrix = np.zeros((len(etalist_str), 400))
    for i, eta in enumerate(etalist_str):
        for r in range(400):
            try:
                CI_list = np.load(parent_dir + "/Data_HPC/82/eta=" + eta + f"_depth=20_Nt=30/seed_{r}/data_ci_list.npy")
                ns_list = np.load(parent_dir + "/Data_HPC/82/eta=" + eta + f"_depth=20_Nt=30/seed_{r}/data_ns_list.npy")
                np_list = np.load(parent_dir + "/Data_HPC/82/eta=" + eta + f"_depth=20_Nt=30/seed_{r}/data_np_list.npy")
                mask = np.logical_or(np_list >= n_s+energy_tol, ns_list >= n_s+energy_tol)
                CI_list[mask] = -10
                ci_matrix[i, r] = np.max(CI_list)
            except:
                ci_matrix[i, r] = -10


    ci_list2 = np.max(ci_matrix, axis=1)
    #CI_list = np.max(np.vstack((ci_list1, ci_list2)),1)
    CI_list = np.maximum(ci_list1, ci_list2)
    print("ECD-MM", CI_list)

    #plt.plot(etalist, ci_list1, label="VQT-EA", marker='o', color=default_colors[0])
    plt.plot(etalist, ci_list2, label="VQT-EA", marker='o', color=default_colors[0])




def plot_ECD_M():
    Nr = 20
    etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
    f_matrix_ECD = np.zeros((len(etalist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                fidelity_history = np.load(
                    parent_dir + f"/Data_HPC/69/Transduction_CoherentInfo_EA_M_Nt=30_depth=20_eta={eta}_ns=3_np=3_N=10000_randomization={j}_ci_list.npy")
                ns_history = np.load(
                    parent_dir + f"/Data_HPC/69/Transduction_CoherentInfo_EA_M_Nt=30_depth=20_eta={eta}_ns=3_np=3_N=10000_randomization={j}_ns_list.npy")
                np_history = np.load(
                    parent_dir + f"/Data_HPC/69/Transduction_CoherentInfo_EA_M_Nt=30_depth=20_eta={eta}_ns=3_np=3_N=10000_randomization={j}_np_list.npy")
                mask = np.logical_or(ns_history >= n_s+energy_tol, np_history >= n_s+energy_tol)
                fidelity_history[mask] = 0
                fidelity_history[np.isnan(fidelity_history)] = 0
                f_matrix_ECD[i, j] = np.max(fidelity_history)
            except:
                print("ECD M EA", eta, j, "fail")

    f_list_ECD = np.max(f_matrix_ECD, 1)
    print("ECD M EA", f_list_ECD, np.argmax(f_matrix_ECD, 1))

    etalist_str = [f"{eta:.2f}" for eta in etalist]
    ci_matrix = np.zeros((len(etalist_str), 200))
    for i, eta in enumerate(etalist_str):
        for r in range(200):
            try:
                CI_list = np.load(parent_dir + "/Data_HPC/83/eta=" + eta + f"_depth=20_Nt=30/seed_{r}/data_ci_list.npy")
                ns_list = np.load(parent_dir + "/Data_HPC/83/eta=" + eta + f"_depth=20_Nt=30/seed_{r}/data_ns_list.npy")
                np_list = np.load(parent_dir + "/Data_HPC/83/eta=" + eta + f"_depth=20_Nt=30/seed_{r}/data_np_list.npy")
                mask = np.logical_or(np_list >= n_s + energy_tol, ns_list >= n_s + energy_tol)
                CI_list[mask] = 0
                ci_matrix[i, r] = np.max(CI_list)
            except:
                ci_matrix[i, r] = 0

    ci_list2 = np.max(ci_matrix, axis=1)

    plt.plot(etalist, f_list_ECD, label="VQT", marker="s", ls="--", color=default_colors[0])
    plt.plot(etalist, ci_list2, label="VQT", marker="s", ls="--", color=default_colors[1])


def plot_ECD_MM_fixedinput():
    Nr = 20

    etalist1 = np.around(np.arange(0.02, 0.4, 0.02), 2)
    etalist2 = np.around(np.arange(0.4, 1.0, 0.1), 1)
    etalist = np.concatenate((etalist1, etalist2))
    etalist = np.sort(etalist)
    dlist = [3, 4]
    f_matrix_ECD = np.zeros((len(etalist), len(dlist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            for k, d in enumerate(dlist):
                try:
                    fidelity_history = np.load(
                        parent_dir + f"/Data_HPC/54/Transduction_CoherentInfo_EA_fixedinput_d={d}_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                    fidelity_history[np.isnan(fidelity_history)] = 0
                    f_matrix_ECD[i, k, j] = np.max(fidelity_history)
                except:
                    print("ECD MM EA fixedinput", eta, j, d, "fail")

    f_list_ECD = np.max(f_matrix_ECD, axis=(1, 2))
    print("ECD MM EA fixedinput", f_list_ECD)

    plt.plot(etalist, f_list_ECD, label="GKP-EA", marker="D", color=default_colors[1])


def h(x):
    x = np.asarray(x, dtype=float)
    xp = (x + 1.0) / 2.0
    xm = (x - 1.0) / 2.0
    return xp * np.log2(xp) - xm * np.log2(xm)


def g(x):
    return (x + 1) * np.log2(x + 1) - x * np.log2(x)


# EA TMS
def plot_EA_TMS():
    QCP_list_ec = []
    QCP_list_nec = []
    r = np.arcsinh(np.sqrt(n_p+energy_tol))
    G = np.cosh(r) ** 2
    Q_twoway_list = []
    for eta in etalist:
        eta_EA = 1 / (1 + (1 - eta) / (eta * G))
        QCP_list_ec.append(max(0, g(eta_EA * n_s) - g((1 - eta_EA) * n_s)))
        #QCP_list_nec.append(max(0, np.log2(eta_EA / (1 - eta_EA))))
        # Q_twoway_list.append(-np.log2(1 - eta_EA))
    plt.plot(etalist, QCP_list_ec, label="TMS-EA", marker="^", color=default_colors[2])
    print("TMS-EC:", QCP_list_ec)
    # plt.plot(etalist, Q_twoway_list, label="Two-way",color = default_colors[3])
    # plt.plot(etalist, QCP_list_nec, label="TMS")


def plot_GKP_n2_Nt30():
    etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)
    randomizationlist = np.arange(20)
    d1_list = np.arange(2, 6, 1)
    d2_list = np.arange(1, 5, 1)
    ci_matrix = np.zeros((len(etalist), len(randomizationlist), len(d1_list), len(d2_list)))
    for i, eta in enumerate(etalist):
        for j, r in enumerate(randomizationlist):
            for k, d1 in enumerate(d1_list):
                for l, d2 in enumerate(d2_list):
                    try:
                        ci_data = np.load(parent_dir +
                                          f"/Data_HPC/64/Transduction_CoherentInfo_GKP_Nt=30_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_ci_list.npy")
                        ns_data = np.load(
                            parent_dir + f"/Data_HPC/64/Transduction_CoherentInfo_GKP_Nt=30_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_ns_list.npy")
                        np_data = np.load(
                            parent_dir + f"/Data_HPC/64/Transduction_CoherentInfo_GKP_Nt=30_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_np_list.npy")
                        #state_RS = np.load(
                        #    parent_dir + f"/Data_HPC/64/Transduction_CoherentInfo_GKP_Nt=30_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_state_RS.npy")
                        #state_P = np.load(
                        #    parent_dir + f"/Data_HPC/64/Transduction_CoherentInfo_GKP_Nt=30_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_state_P.npy")
                        mask = np.logical_or((np.array(ns_data) >= 2.05), (np.array(np_data) >= 2.05))
                        ci_data[mask] = 0
                        ci_matrix[i, j, k, l] = np.max(ci_data)
                    except:
                        ci_matrix[i, j, k, l] = 0

    ci_list = np.max(ci_matrix, axis=(1, 2, 3))
    print("GKP", ci_list)
    plt.plot(etalist, ci_list, label="GKP", ls="--", marker="*", color=default_colors[1])


def plot_pure_loss_capacity(n_s):
    etalist = np.arange(0.05, 1.0, 0.05)
    cilist = []
    for eta in etalist:
        ci = max(0, g(eta * n_s) - g((1 - eta) * n_s))
        cilist.append(ci)
    plt.plot(etalist, cilist, label="Pure-Loss", ls="--", marker="v", color=default_colors[3])


fig = plt.figure(figsize=(8, 6))
fs = 20
plt.rcParams.update({
    'font.size': fs,
    'axes.labelsize': fs,
    'legend.fontsize': fs - 6,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'lines.linewidth': 1.5,  # Make lines slightly thicker for visibility
    'lines.markersize': 5  # Adjust markers to match the scale
})

plot_ECD_MM()
plot_ECD_M()
#plot_ECD_MM_fixedinput()
#plot_GKP_n2_Nt30()
plot_EA_TMS()
plot_pure_loss_capacity(3)

plt.legend(loc="upper left", frameon=False)
#plt.title(f"Non-Adaptive Protocols")
plt.tick_params(axis='both', which='major', )
plt.xlabel("Transmissivity" + r" $\eta$", )
plt.ylabel("Coherent Information(CI)")

plt.tight_layout()
plt.savefig(parent_dir + "/Figs/CI_ns=np=3_Non-Adaptive.jpg", dpi=500)
plt.show()
