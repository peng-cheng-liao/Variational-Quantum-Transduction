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
n_s = 2
n_p = 2
VQT_NOISE_RUN_ID = 92
GKP_NOISE_RUN_ID = 93
VQT_RUN92_NOISY_SETUPS = [
    {
        "folder": "noisy_nPth=0p1_kS=0p99_kP=0p99",
        "title": r"$n_P^{\rm th}=0.1,\ \kappa_S=\kappa_P=0.99$",
        "nbar_p": 0.1,
        "kappa_s": 0.99,
        "kappa_p": 0.99,
        "label": "VQT",
    },
    {
        "folder": "noisy_nPth=0p01_kS=0p99_kP=0p99",
        "title": r"$n_P^{\rm th}=0.01,\ \kappa_S=\kappa_P=0.99$",
        "nbar_p": 0.01,
        "kappa_s": 0.99,
        "kappa_p": 0.99,
        "label": "VQT",
    },
    {
        "folder": "noisy_nPth=0p001_kS=0p99_kP=0p99",
        "title": r"$n_P^{\rm th}=0.001,\ \kappa_S=\kappa_P=0.99$",
        "nbar_p": 0.001,
        "kappa_s": 0.99,
        "kappa_p": 0.99,
        "label": "VQT",
    },
]


def eta_folder(eta):
    return f"eta={eta:.2f}"


def load_vqt_run92_ci(setup_name, etas=etalist):
    result_folder = data_dir / str(VQT_NOISE_RUN_ID) / setup_name
    ci_list = []
    for eta in etas:
        path = result_folder / eta_folder(eta) / "best_feasible_ci.txt"
        try:
            ci_list.append(float(path.read_text().strip()))
        except (OSError, ValueError):
            print(f"Missing VQT run-{VQT_NOISE_RUN_ID} CI: {path}")
            ci_list.append(np.nan)
    return np.array(ci_list)


def load_gkp_run93_ci(setup_name, etas=etalist):
    result_folder = data_dir / str(GKP_NOISE_RUN_ID) / setup_name
    ci_list = []
    for eta in etas:
        path = result_folder / eta_folder(eta) / "best_feasible_ci.txt"
        try:
            ci_list.append(float(path.read_text().strip()))
        except (OSError, ValueError):
            print(f"Missing GKP run-{GKP_NOISE_RUN_ID} CI: {path}")
            ci_list.append(np.nan)
    return np.array(ci_list)


def g(x):
    scalar_input = np.isscalar(x)
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = (x[mask] + 1.0) * np.log2(x[mask] + 1.0) - x[mask] * np.log2(x[mask])
    if scalar_input:
        return float(out)
    return out


def gaussian_thermal_loss_ci(N, eta_channel, nbar_p, kappa_p):
    T = np.clip(kappa_p * eta_channel, 0.0, 1.0)
    denom = 1.0 - T
    if denom <= 1e-12:
        denom = 1e-12
    N_B = kappa_p * (1.0 - eta_channel) * nbar_p / denom

    a = N + 0.5
    b = T * N + (1.0 - T) * N_B + 0.5
    c = np.sqrt(max(T * N * (N + 1.0), 0.0))

    Delta = a * a + b * b - 2.0 * c * c
    D = (a * b - c * c) ** 2
    disc = max(Delta * Delta - 4.0 * D, 0.0)

    nu_plus = np.sqrt(max((Delta + np.sqrt(disc)) / 2.0, 0.0))
    nu_minus = np.sqrt(max((Delta - np.sqrt(disc)) / 2.0, 0.0))

    n_plus = max(nu_plus - 0.5, 0.0)
    n_minus = max(nu_minus - 0.5, 0.0)

    return g(T * N + (1.0 - T) * N_B) - g(n_plus) - g(n_minus)


def gaussian_thermal_loss_ci_bound(eta_channel, nbar_p, kappa_p, n_s=2, num_grid=1001):
    n_grid = np.linspace(0.0, n_s, num_grid)
    ci_values = np.array([
        gaussian_thermal_loss_ci(N, eta_channel, nbar_p, kappa_p)
        for N in n_grid
    ])
    return max(0.0, float(np.nanmax(ci_values)))


def gaussian_qt_curve(setup, etas=etalist):
    values = []
    for eta in etas:
        values.append(
            gaussian_thermal_loss_ci_bound(
                eta,
                setup["nbar_p"],
                setup["kappa_p"],
                n_s=n_s,
            )
        )
    return np.array(values)


def gaussian_tms_ea_curve(setup, etas=etalist):
    values = []
    r = np.arcsinh(np.sqrt(n_p))
    G = np.cosh(r) ** 2
    for eta in etas:
        eta_EA = 1 / (1 + (1 - eta) / (eta * G))
        values.append(
            gaussian_thermal_loss_ci_bound(
                eta_EA,
                setup["nbar_p"],
                setup["kappa_p"],
                n_s=n_s,
            )
        )
    return np.array(values)


def plot_noisy_vqt_setup(ax, setup):
    ci_values = load_vqt_run92_ci(setup["folder"])
    print(f"{setup['label']} {setup['folder']}", ci_values)
    ax.plot(
        etalist,
        ci_values,
        label=setup["label"],
        marker="o",
        color=default_colors[0],
    )


def plot_noisy_gkp_setup(ax, setup):
    ci_values = load_gkp_run93_ci(setup["folder"])
    print(f"GKP {setup['folder']}", ci_values)
    ax.plot(
        etalist,
        ci_values,
        label="GKP",
        marker="*",
        ls="--",
        color=default_colors[1] if len(default_colors) > 1 else None,
    )


def plot_gaussian_benchmarks(ax, setup):
    # kappa_s is tracked in setup metadata for parity with VQT-noise, but the
    # single-output nonadaptive Gaussian benchmark only uses the retained P mode.
    qt_values = gaussian_qt_curve(setup)
    tms_values = gaussian_tms_ea_curve(setup)
    print(f"TMS-EA {setup['folder']}", tms_values)
    ax.plot(etalist, tms_values, label="TMS-EA", marker="^", color=default_colors[2])
    print(f"QT {setup['folder']}", qt_values)
    ax.plot(etalist, qt_values, label="QT", ls="--", marker="v", color=default_colors[3])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
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

    for ax, setup in zip(axes, VQT_RUN92_NOISY_SETUPS):
        plot_noisy_vqt_setup(ax, setup)
        plot_noisy_gkp_setup(ax, setup)
        plot_gaussian_benchmarks(ax, setup)
        ax.set_title(setup["title"])
        ax.set_xlabel(r"Transmissivity $\eta$")
        ax.tick_params(axis='both', which='major')
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Coherent Information (CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "CI_ns=np=2_Non-Adaptive_noisy_three_panel.jpg", dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
