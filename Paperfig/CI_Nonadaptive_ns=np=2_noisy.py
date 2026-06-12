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
from Quantum_Plotting import *

rc('text', usetex=True)

data_dir = repo_dir / "Data_HPC"
fig_dir = repo_dir / "Figs"

etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
VQT_RUN_ID = 84
VQT_NOISE_RUN_ID = 92
VQT_RUN92_NOISY_SETUPS = [
    {
        "folder": "noisy_nPth=0p1_kS=0p99_kP=0p99",
        "label": r"VQT-noise ($n_P^{\rm th}=0.1$, $\kappa_S=\kappa_P=0.99$)",
        "marker": "x",
        "ls": "-.",
    },
    {
        "folder": "noisy_nPth=0p01_kS=0p99_kP=0p99",
        "label": r"VQT-noise ($n_P^{\rm th}=0.01$, $\kappa_S=\kappa_P=0.99$)",
        "marker": "P",
        "ls": "-.",
    },
    {
        "folder": "noisy_nPth=0p001_kS=0p99_kP=0p99",
        "label": r"VQT-noise ($n_P^{\rm th}=0.001$, $\kappa_S=\kappa_P=0.99$)",
        "marker": "X",
        "ls": "-.",
    },
]


def eta_folder(eta):
    return f"eta={eta:.2f}"


def load_best_feasible_ci(run_id, etas=etalist):
    ci_list = []
    for eta in etas:
        path = data_dir / str(run_id) / eta_folder(eta) / "best_feasible_ci.txt"
        try:
            ci_list.append(float(path.read_text().strip()))
        except OSError:
            print(f"Missing best feasible CI: {path}")
            ci_list.append(np.nan)
    return np.array(ci_list)


def load_vqt_run92_ci(setup_name, etas=etalist):
    result_folder = data_dir / str(VQT_NOISE_RUN_ID) / setup_name
    ci_list = []
    for eta in etas:
        path = result_folder / eta_folder(eta) / "best_feasible_ci.txt"
        if path.exists():
            ci_list.append(float(path.read_text().strip()))
        else:
            print(f"Missing VQT run-{VQT_NOISE_RUN_ID} CI: {path}")
            ci_list.append(np.nan)
    return np.array(ci_list)


def plot_vqt():
    ci_list = load_best_feasible_ci(VQT_RUN_ID)
    print("VQT", ci_list)
    plt.plot(etalist, ci_list, label="VQT", marker="o", color=default_colors[0])


def plot_noisy_vqt_setups():
    for i, setup in enumerate(VQT_RUN92_NOISY_SETUPS):
        ci_values = load_vqt_run92_ci(setup["folder"])
        print(setup["label"], ci_values)
        color_index = 4 + i
        plt.plot(
            etalist,
            ci_values,
            label=setup["label"],
            marker=setup["marker"],
            ls=setup["ls"],
            color=default_colors[color_index] if len(default_colors) > color_index else None,
        )


def main():
    fig = plt.figure(figsize=(8, 6))
    fs = 20
    plt.rcParams.update({
        'font.size': fs,
        'axes.labelsize': fs,
        'legend.fontsize': fs - 6,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })

    plot_vqt()
    plot_noisy_vqt_setups()

    plt.legend(loc="upper left", frameon=False)
    plt.tick_params(axis='both', which='major')
    plt.xlabel("Transmissivity" + r" $\eta$")
    plt.ylabel("Coherent Information(CI)")

    plt.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "CI_ns=np=2_Non-Adaptive_noisy.jpg", dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
