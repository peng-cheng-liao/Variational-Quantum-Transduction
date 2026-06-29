import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
os.environ.setdefault("MPLCONFIGDIR", str(script_dir / ".mplconfig"))
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

rc('text', usetex=True)

parent_dir = str(repo_dir)
data_dir = repo_dir / "Data_HPC"
fig_dir = repo_dir / "Figs"

etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
n_s = 3
n_p = 3

VQT_RUN_ID = 82
VQT_WITHOUT_EA_RUN_ID = 69
GKP_RUN_ID = 63
GKP_PREFIX = "Transduction_CoherentInfo_GKP_Nt=30_ns=3_np=3_N=2000_processed"

default_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'  # Cyan
]


def load_best_feasible_ci(run_id, etas=etalist):
    path = data_dir / str(run_id) / f"best_feasible_ci_list_{run_id}.npy"
    ci_list = np.load(path)
    if len(ci_list) != len(etas):
        raise ValueError(f"{path} has {len(ci_list)} entries, expected {len(etas)}")
    return ci_list


def load_gkp_processed_ci():
    run_dir = data_dir / str(GKP_RUN_ID)
    gkp_etas = np.load(run_dir / f"{GKP_PREFIX}_eta_list.npy")
    ci_list = np.load(run_dir / f"{GKP_PREFIX}_ci_list.npy")
    if len(ci_list) != len(gkp_etas):
        raise ValueError("GKP processed CI and eta arrays have different lengths")
    return gkp_etas, ci_list


# ECD MM

def plot_ECD_MM():
    ci_list = load_best_feasible_ci(VQT_RUN_ID)
    print("ECD-MM", ci_list)
    plt.plot(etalist, ci_list, label="VQT", marker='o', color=default_colors[0])


def plot_ECD_M():
    ci_list = load_best_feasible_ci(VQT_WITHOUT_EA_RUN_ID)
    print("ECD M EA", ci_list)
    plt.plot(etalist, ci_list, label="VQT without EA", marker="s", ls="--", color=default_colors[0])


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
    etalist = np.arange(0.05, 1.0, 0.05)
    for eta in etalist:
        eta_EA = 1 / (1 + (1 - eta) / (eta * G))
        QCP_list_ec.append(max(0, g(eta_EA * n_s) - g((1 - eta_EA) * n_s)))
        #QCP_list_nec.append(max(0, np.log2(eta_EA / (1 - eta_EA))))
        # Q_twoway_list.append(-np.log2(1 - eta_EA))
    plt.plot(etalist, QCP_list_ec, label="TMS-EA", marker="^", color=default_colors[2])
    print("TMS-EC:", QCP_list_ec)
    # plt.plot(etalist, Q_twoway_list, label="Two-way",color = default_colors[3])
    # plt.plot(etalist, QCP_list_nec, label="TMS")


def plot_GKP_n3_Nt30():
    gkp_etas, ci_list = load_gkp_processed_ci()
    selected_etas = np.around(np.arange(0.05, 0.70, 0.05), 2)
    mask = np.isin(np.around(gkp_etas, 2), selected_etas)
    gkp_etas = gkp_etas[mask]
    ci_list = ci_list[mask]
    print("GKP", ci_list)
    plt.plot(gkp_etas, ci_list, label="GKP-QT", ls="--", marker="*", color=default_colors[1])


def plot_pure_loss_capacity(n_s):
    etalist = np.arange(0.05, 1.0, 0.05)
    cilist = []
    for eta in etalist:
        ci = max(0, g(eta * n_s) - g((1 - eta) * n_s))
        cilist.append(ci)
    plt.plot(etalist, cilist, label="QT", ls="--", marker="v", color=default_colors[3])


def main():
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
    plot_GKP_n3_Nt30()
    plot_EA_TMS()
    plot_pure_loss_capacity(3)

    plt.legend(loc="upper left", frameon=False)
    #plt.title(f"Non-Adaptive Protocols")
    plt.tick_params(axis='both', which='major', )
    plt.xlabel("Transmissivity" + r" $\eta$", )
    plt.ylabel("Coherent Information(CI)")

    plt.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "CI_ns=np=3_Non-Adaptive.jpg", dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
