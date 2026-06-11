import matplotlib.pyplot as plt
import os

import numpy as np

from main import *
from matplotlib import rc
from Quantum_Plotting import *

rc('text', usetex=True)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)
n_s = 2
n_p = 2
energy_tol = 0.05




def plot_ECD_MM_Tele():
    Nr = 20
    etalist1 = np.around(np.arange(0.01, 0.1, 0.01), 2)
    etalist2 = np.around(np.arange(0.1, 1.0, 0.1), 1)
    etalist3 = np.concatenate((etalist1, etalist2))
    CI_matrix = np.zeros((len(etalist3), Nr))
    for i, eta in enumerate(etalist3):
        for j in range(Nr):
            try:
                ci_history = np.load(parent_dir +
                                     f"/Data_HPC/45/Transduction_CoherentInfo_EATele_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("MM EATele", eta, j, "fail")

    CI_list = np.max(CI_matrix, 1)
    print("ECD-MM-Tele", CI_list)
    plt.plot(etalist3, CI_list, label="VQT",marker="^")
    """
    RandomIndex = np.argmax(CI_matrix, 1)
    alphalist = []
    for j, eta in enumerate(etalist3):
        x = np.load(
            parent_dir + f"/Data_HPC/45/Transduction_CoherentInfo_EATele_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={RandomIndex[j]}_parameters.npy")
        alphalist.append(x[-1])
    print("ECD-MM-Tele Alpha list", alphalist)
    """


def plot_ECD_MM_Tele_v2():
    etalist2 = np.around(np.arange(0.1, 1.0, 0.1), 2)
    etalist2_str = [f"{eta:.2f}" for eta in etalist2]
    ci_matrix = np.zeros((len(etalist2_str), 200))
    for i, eta in enumerate(etalist2_str):
        for r in range(200):
            try:
                CI_list = np.load(parent_dir + f"/Data_HPC/88/eta={eta}_depth=20_Nt=30/seed_{r}/data_ci_list.npy")
                ns_list = np.load(parent_dir + f"/Data_HPC/88/eta={eta}_depth=20_Nt=30/seed_{r}/data_ns_list.npy")
                np_list = np.load(parent_dir + f"/Data_HPC/88/eta={eta}_depth=20_Nt=30/seed_{r}/data_np_list.npy")
                mask = np.logical_or(np_list >= n_s + energy_tol, ns_list >= n_s + energy_tol)
                CI_list[mask] = -10
                ci_matrix[i, r] = np.max(CI_list)
            except:
                ci_matrix[i, r] = -10

    ci_list2 = np.max(ci_matrix, axis=1)
    # plt.scatter(etalist, f_list_ECD, label="ECD-MM")
    # plt.plot(np.delete(etalist, 9), np.delete(f_list_ECD, 9), label="VQT-EA", marker='o',color = default_colors[0])
    plt.plot(etalist2, ci_list2, label="VQT-EA", marker='o', color=default_colors[3])


def plot_ECD_M_Tele():
    Nr = 20
    etalist1 = np.around(np.arange(0.01, 0.1, 0.01), 2)
    etalist2 = np.around(np.arange(0.1, 1.0, 0.1), 1)
    etalist = np.concatenate((etalist1, etalist2))
    CI_matrix = np.zeros((len(etalist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                ci_history = np.load(parent_dir +
                                     f"/Data_HPC/58/Transduction_CoherentInfo_EATele_M_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("M EATele", eta, j, "fail")

    CI_list = np.max(CI_matrix, 1)
    print("ECD-M-Tele", CI_list)
    plt.plot(etalist, CI_list, label="VQT-without-EA", marker="s")

    """
    RandomIndex = np.argmax(CI_matrix,1)
    alphalist = []
    for j, eta in enumerate(etalist):
        x = np.load(parent_dir + f"/Data_HPC/45/Transduction_CoherentInfo_EATele_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={RandomIndex[j]}_parameters.npy")
        alphalist.append(x[-1])
    print("ECD-MM-Tele Alpha list", alphalist)
    """


def plot_ECD_MM_Tele_fixedinput():
    Nr = 20
    etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)
    CI_matrix = np.zeros((len(etalist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                ci_history = np.load(
                    parent_dir + f"/Data_HPC/57/Transduction_CoherentInfo_EATele_fixedinputNt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                ns_history = np.load(
                    parent_dir + f"/Data_HPC/57/Transduction_CoherentInfo_EATele_fixedinputNt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ns_list.npy")
                np_history = np.load(
                    parent_dir + f"/Data_HPC/57/Transduction_CoherentInfo_EATele_fixedinputNt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_np_list.npy")
                ci_history[np.isnan(ci_history)] = 0
                # mask = np.logical_or((ns_history > 2.05) , (np_history > 2.05))
                # ci_history[mask] = 0
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("MM EATele fixed input", eta, j, "fail")

    CI_list = np.max(CI_matrix, 1)
    print("ECD-MM-Tele fixed input", CI_list)
    plt.plot(etalist, CI_list, label="ECD-MM-Adaptive-fi",marker="o")
    """
    RandomIndex = np.argmax(CI_matrix,1)
    alphalist = []
    for j, eta in enumerate(etalist3):
        x = np.load(parent_dir + f"/Data_HPC/45/Transduction_CoherentInfo_EATele_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={RandomIndex[j]}_parameters.npy")
        alphalist.append(x[-1])
    print("ECD-MM-Tele Alpha list", alphalist)
    """


def plot_ECD_MM_Tele_fixedinput_verification():
    etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)
    CI_Matrix = np.load(
        parent_dir + f"/Data_Local/53_Transduction_CoherentInfo_EATele_fixedinputNt=30_depth=20_Verification_CIMatrix.npy")
    CI_list = np.max(CI_Matrix, 1)
    plt.plot(etalist, CI_list, label="ECD-MM-Tele-fi-vi")


def plot_ECD_MM_Tele_IteTrain():
    Nr = 20
    etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)
    CI_matrix = np.zeros((len(etalist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                ci_history = np.load(
                    parent_dir + f"/Data_HPC/56/Transduction_CoherentInfo_EATele_IteTrainNt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("MM EATele IteTrain", eta, j, "fail")

    CI_list = np.max(CI_matrix, 1)
    print("ECD-MM-Tele IteTrain", CI_list)
    plt.scatter(etalist, CI_list, label="ECD-MM-Tele-IteTrain")


def plot_ECD_MM_Tele_UM():
    Nr = 20
    CI_matrix = np.zeros((9, Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                ci_history = np.load(parent_dir +
                                     f"/Data_HPC/47/Transduction_CoherentInfo_EATele_UM_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("MM EATele UM", eta, j, "fail")

    CI_list = np.max(CI_matrix, 1)
    print("ECD-MM-Tele-UM", CI_list)
    plt.scatter(etalist, CI_list, label="ECD-MM-Tele-UM")
    RandomIndex = np.argmax(CI_matrix, 1)
    """
    alphalist = []
    for j,eta in enumerate(etalist):
        x = np.load(parent_dir + f"/Data_HPC/47/Transduction_CoherentInfo_EATele_UM_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={RandomIndex[j]}_parameters.npy")
        alphalist.append(x[-1])
    print("ECD-MM-Tele-UM Alpha list", alphalist)
    """


def plot_ECD_MM_Tele_UM2():
    Nr = 20
    CI_matrix = np.zeros((9, Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                ci_history = np.load(parent_dir +
                                     f"/Data_HPC/49/Transduction_CoherentInfo_EATele_UM_Nt=30_depth=15_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("MM EATele UM 2", eta, j, "fail")

    CI_list = np.max(CI_matrix, 1)
    print("ECD-MM-Tele-UM-2", CI_list)
    plt.scatter(etalist, CI_list, label="ECD-MM-Tele-UM-2")


def plot_ECD_MM_AQT():
    Nr = 20
    CI_matrix = np.zeros((9, Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                ci_history = np.load(parent_dir +
                                     f"/Data_HPC/47/Transduction_CoherentInfo_EATele_UM_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                CI_matrix[i, j] = np.max(ci_history)
            except:
                print("MM EATele UM", eta, j, "fail")

    CI_list = np.max(CI_matrix, 1)
    print("ECD-MM-Tele-UM", CI_list)
    plt.scatter(etalist, CI_list, label="ECD-MM-Tele-UM")


def plot_Tele_numerical():
    Nr = 20
    f_matrix_ECD = np.zeros((9, Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                fidelity_history = np.load(
                    parent_dir + f"/Data_HPC/46/Transduction_CoherentInfo_Tele_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                fidelity_history[np.isnan(fidelity_history)] = 0
                f_matrix_ECD[i, j] = np.max(fidelity_history)
            except:
                print("Tele Numerical", eta, j, "fail")

    f_list_ECD = np.max(f_matrix_ECD, 1)
    print("Tele Numerical", f_list_ECD)
    plt.scatter(etalist, f_list_ECD, label="Tele-Nu")


def plot_AQT_numerical():
    Nr = 20
    etalist1 = np.around(np.arange(0.01, 0.1, 0.01), 2)
    etalist2 = np.around(np.arange(0.1, 1.0, 0.1), 1)
    etalist = np.concatenate((etalist1, etalist2))
    f_matrix_ECD = np.zeros((len(etalist), Nr))


    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                fidelity_history = np.load(
                    parent_dir + f"/Data_HPC/48/Transduction_CoherentInfo_AQT_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                fidelity_history[np.isnan(fidelity_history)] = 0
                f_matrix_ECD[i, j] = np.max(fidelity_history)
            except:
                print("AQT Numerical", eta, j, "fail")

    f_list_ECD = np.max(f_matrix_ECD, 1)
    print("AQT Numerical", f_list_ECD)
    #plt.plot(etalist, f_list_ECD, label="AQT",ls="--",marker='D')


    path = parent_dir+ f"/Data/cohinfo_curve_vs_eta.csv"



    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    x = (1 + 10000*np.arange(100)) / 1_000_000

    y = data[:, 1]
    print(x)
    print(y)

    """
    ci_list = []
    print(v1.shape, v2.shape)
    for i, eta in enumerate(etalist):
        CI = transduction_protocol_CoherentInfo_AQT_v2(eta, n_s, n_p, 30)[0]
        ci_list.append(CI.detach().item())
    """

    plt.plot(x[1::5], y[0::5], label="AQT",ls="--",marker='D')








def plot_Tele_thermal():
    """
    Nt = 30
    n_s = 2
    n_p = 2
    depth = 0
    parameters = torch.rand(depth * Nt)
    etalist = np.arange(0.1, 1.0, 0.1)
    print(etalist)
    CI_list = []

    for eta in etalist:
        CI = transduction_protocol_CoherentInfo_Tele(n_p, eta, parameters, depth, Nt)[0]
        CI_list.append(CI.detach().item())

    """
    CI_list = np.load(
        parent_dir + "/Data_Local/Transduction_CoherentInfo_Tele_Thermal_Nt=30_ns=2_np=2_N=10000_ci_list.npy")
    plt.scatter(etalist, CI_list, label="Tele-Thermal")


import pandas as pd


def plot_AQT_analytical():
    df = pd.read_csv("cohinfo_curve_vs_eta.csv", usecols=[0, 1])
    y = df.iloc[:, 1].tolist()
    plt.plot(np.linspace(0, 1, len(y)), y, label="AQT-Analytical")


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
    r = np.arcsinh(np.sqrt(n_p))
    G = np.cosh(r) ** 2
    Q_twoway_list = []
    for eta in etalist:
        eta_EA = 1 / (1 + (1 - eta) / (eta * G))
        QCP_list_ec.append(max(0, g(eta_EA * n_s) - g((1 - eta_EA) * n_s)))
        #QCP_list_nec.append(max(0, np.log2(eta_EA / (1 - eta_EA))))
        # Q_twoway_list.append(-np.log2(1 - eta_EA))
    plt.plot(etalist, QCP_list_ec, label="TMS-EC", ls="--")
    print("TMS-EC:", QCP_list_ec)
    # plt.plot(etalist, Q_twoway_list, label="Two-way",color = default_colors[3])
    # plt.plot(etalist, QCP_list_nec, label="TMS")


def plot_Tele_Bound():
    etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
    Gs = 2 * n_p + 1 + 2 * np.sqrt(n_p * (n_p + 1))  # squeezing gain
    LB_list = []
    UB_list = []
    UB2_list = []
    LB_list_EC = []
    UB_list_EC = []
    UB_list_EC2 = []
    for eta in etalist:
        v = 1 / (np.sqrt(eta * (1 - eta)) * Gs)
        sigma_square = v / 2
        # print(eta, sigma_square)
        Dprime = np.sqrt((2 * n_s + sigma_square + 1) ** 2 - 4 * n_s * (n_s + 1))
        lb_ec = g(n_s + sigma_square) - g((Dprime + sigma_square - 1) / 2) - g((Dprime - sigma_square - 1) / 2)
        ub_ec = g(2 * n_s / (sigma_square + 2)) - g(n_s * sigma_square / (sigma_square + 2))
        ub_ec2 = g(n_s + sigma_square / 2) - g(((sigma_square / 2) * (n_s + sigma_square / 2)) / (1 - sigma_square / 2))

        lb = np.maximum(0, -np.log2(v / 2) - 1 / np.log(2))  # without energy constraint
        ub = np.maximum(0,
                        -np.log2(v / 2) - 1 / np.log(2) + 2 * h(
                            np.sqrt(1.0 + (v ** 2) / 4.0)))  # without energy constraint
        ub2 = np.log2((1 - sigma_square) / sigma_square)
        LB_list_EC.append(lb_ec)
        UB_list_EC.append(ub_ec)
        UB_list_EC2.append(ub_ec2)
        LB_list.append(lb)
        UB_list.append(ub)
        UB2_list.append(ub2)

    plt.plot(etalist, LB_list_EC, label="Tele-LB-EC", ls="--", marker= "*")
    plt.plot(etalist, UB_list_EC, label="Tele-UB-EC", ls="--")
    plt.plot(etalist, LB_list, label="Tele-LB", ls="--")
    plt.plot(etalist, UB_list, label="Tele-UB", ls="--")


def plot_AQT_bound():
    r = np.arcsinh(np.sqrt(n_p))
    LB_list = []
    for eta in etalist:
        sigma2_q = ((1 - eta) / eta) * np.exp(-2 * r)
        sigma2 = sigma2_q / 2
        Dprime = np.sqrt((2 * n_s + sigma2 + 1) ** 2 - 4 * n_s * (n_s + 1))
        lb_ec = g(n_s + sigma2_q) - g(sigma2_q)
        lb_ec2 = g(np.sqrt((n_s + 1 / 2) * (n_s + 1 / 2 + sigma2_q)) - 1 / 2) - g(np.sqrt(sigma2_q * (n_s + 1)) - 1 / 2)
        LB_list.append(lb_ec)
    plt.plot(etalist, LB_list, label="Tele-AQT", ls="--")


def plot_Q2_bound():
    r = np.arcsinh(np.sqrt(n_p))
    n_s_eq = n_s * np.cosh(r) * np.cosh(r) + np.sinh(r) * np.sinh(r)
    print(n_s_eq)
    LB_list = []
    UB_list = []
    Q2_list = []
    for eta in etalist:
        lb = g(n_s_eq) - g((1 - eta) * n_s_eq)
        ub = g((1 + eta) * n_s_eq / 2) - g((1 - eta) * n_s_eq / 2)
        LB_list.append(lb)
        UB_list.append(ub)
        Q2_list.append(-np.log2(1 - eta))
    #plt.plot(etalist, LB_list, label="Q2-LB", ls="--")
    #plt.plot(etalist, UB_list, label="Q2-UB", ls="--")
    plt.plot(etalist, Q2_list, label="PLOB", ls="--")


def plot_GKP():
    etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)
    randomizationlist = np.arange(20)
    d1_list = np.arange(2, 8, 1)
    d2_list = np.arange(1, 8, 1)
    ci_matrix = np.zeros((len(etalist), len(randomizationlist), len(d1_list), len(d2_list)))
    for i, eta in enumerate(etalist):
        for j, r in enumerate(randomizationlist):
            for k, d1 in enumerate(d1_list):
                for l, d2 in enumerate(d2_list):
                    try:
                        ci_data = np.load(
                            parent_dir + f"/Data_HPC/61/Transduction_CoherentInfo_GKP_Nt=40_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_ci_list.npy")
                        ns_data = np.load(
                            parent_dir + f"/Data_HPC/61/Transduction_CoherentInfo_GKP_Nt=40_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_ns_list.npy")
                        np_data = np.load(
                            parent_dir + f"/Data_HPC/61/Transduction_CoherentInfo_GKP_Nt=40_d1={d1}_d2={d2}_j2=0_eta={eta}_ns=2_np=2_N=2000_randomization={r}_np_list.npy")
                        mask = np.logical_or((np.array(ns_data) >= 2.00), (np.array(np_data) >= 2.00))
                        ci_data[mask] = 0
                        ci_matrix[i, j, k, l] = np.max(ci_data)
                        #print(i, j, k, l, np.max(ci_data))
                    except:
                        ci_matrix[i, j, k, l] = 0

    ci_list = np.max(ci_matrix, axis=(1, 2, 3))
    for i, eta in enumerate(etalist):
        ci_matrix_i = ci_matrix[i, :, :, :]

        print(eta, np.unravel_index(np.argmax(ci_matrix_i), ci_matrix_i.shape))
    print("GKP", ci_list)
    plt.scatter(etalist, ci_list, label="GKP", )

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
    print("GKP-Nt=30-n=2", ci_list)
    plt.scatter(etalist, ci_list, label="GKP-Nt=30-n=2")

fig = plt.figure(figsize=(8,6))
fs = 20
plt.rcParams.update({
    'font.size': fs,
    'axes.labelsize': fs,
    'legend.fontsize': fs-6,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'lines.linewidth': 1.5,   # Make lines slightly thicker for visibility
    'lines.markersize': 5     # Adjust markers to match the scale
})



plot_ECD_MM_Tele()
#plot_ECD_MM_Tele_v2()
plot_ECD_M_Tele()
# plot_ECD_MM_Tele_fixedinput()
plot_AQT_numerical()
#plot_Tele_Bound()
#plot_Q2_bound()

#plot_Tele_thermal()
#plot_Tele_numerical()

#plot_ECD_MM()
#plot_ECD_M()
#plot_EA_TMS()
#plot_ECD_MM_fixedinput()
#plot_ECD_MM_IteTrain()

#
#plot_GKP_n2_Nt30()

# plot_ECD_MM_Tele_fixedinput_verification()
#plot_ECD_MM_Tele_IteTrain()
#plot_ECD_MM_Tele_UM()
#plot_AQT_analytical()
#plot_ECD_MM_Tele_UM2()
plt.legend(loc = "upper left", frameon=False)
#plt.title(f"Non-Adaptive Protocols")
plt.tick_params(axis='both', which='major',)
plt.xlabel("Transmissivity"+r" $\eta$",)
plt.ylabel("Coherent Information(CI)")





plt.tight_layout()
plt.savefig(parent_dir + "/Figs/VQT_ns=np=2_Adaptive.jpg", dpi=500)
plt.show()
