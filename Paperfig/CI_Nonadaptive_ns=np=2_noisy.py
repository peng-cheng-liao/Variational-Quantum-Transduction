import os
import sys
import json
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
os.environ.setdefault("MPLBACKEND", "Agg")
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
FIGSIZE = (11.0, 4.0)
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 12
TITLE_SIZE = 16
LEGEND_SIZE = 13
LINE_WIDTH = 1.9
MARKER_SIZE = 5.5
VQT_NOISE_RUN_ID = 92
GKP_NOISE_RUN_ID = 93
VQT_A_LOSS_RUN_ID = 95
VQT_A_LOSS_CURVES = [
    {
        "folder": "noisy_nPth=0p1_kS=0p99_kP=0p99_kA=1p00",
        "label": "VQT",
        "marker": "o",
        "ls": "-",
        "color_index": 0,
    },
    {
        "folder": "noisy_nPth=0p1_kS=0p99_kP=0p99_kA=0p95",
        "label": r"VQT $(\kappa_A=0.95)$",
        "marker": "s",
        "ls": ":",
        "color_index": 4,
    },
    {
        "folder": "noisy_nPth=0p1_kS=0p99_kP=0p99_kA=0p90",
        "label": r"VQT $(\kappa_A=0.90)$",
        "marker": "D",
        "ls": "-.",
        "color_index": 5,
    },
]
NOISY_SETUPS = [
    {
        "folder": "noisy_nPth=0p1_kS=0p99_kP=0p99",
        "title": r"$n_P^{\rm th}=0.1$" + "\n" + r"$\kappa_S=\kappa_P=0.99$",
        "nbar_p": 0.1,
        "kappa_s": 0.99,
        "kappa_p": 0.99,
        "vqt_curves": VQT_A_LOSS_CURVES,
        "gkp_curve": {
            "run_id": GKP_NOISE_RUN_ID,
            "folder": "noisy_nPth=0p1_kS=0p99_kP=0p99",
            "label": "GKP",
        },
    },
    {
        "folder": "noisy_nPth=0p01_kS=0p99_kP=0p99",
        "title": r"$n_P^{\rm th}=0.01$" + "\n" + r"$\kappa_S=\kappa_P=0.99$",
        "nbar_p": 0.01,
        "kappa_s": 0.99,
        "kappa_p": 0.99,
        "vqt_curves": [
            {
                "run_id": VQT_NOISE_RUN_ID,
                "folder": "noisy_nPth=0p01_kS=0p99_kP=0p99",
                "label": "VQT",
                "marker": "o",
                "ls": "-",
                "color_index": 0,
            },
        ],
    },
    {
        "folder": "noisy_nPth=0p001_kS=0p99_kP=0p99",
        "title": r"$n_P^{\rm th}=0.001$" + "\n" + r"$\kappa_S=\kappa_P=0.99$",
        "nbar_p": 0.001,
        "kappa_s": 0.99,
        "kappa_p": 0.99,
        "vqt_curves": [
            {
                "run_id": VQT_NOISE_RUN_ID,
                "folder": "noisy_nPth=0p001_kS=0p99_kP=0p99",
                "label": "VQT",
                "marker": "o",
                "ls": "-",
                "color_index": 0,
            },
        ],
    },
]


def eta_folder(eta):
    return f"eta={eta:.2f}"


def _format_eta_list(etas):
    return ", ".join(f"{eta:.2f}" for eta in etas) if etas else "none"


def _curve_color(curve):
    index = curve.get("color_index")
    if index is None or index >= len(default_colors):
        return None
    return default_colors[index]


def load_noisy_ci_curve(run_id, setup_name, label, etas=etalist):
    # Each eta folder contains the processed best-feasible CI selected by the
    # corresponding HPC processing job, plus diagnostics in noise_config.json.
    result_folder = data_dir / str(run_id) / setup_name
    rows = []
    missing_etas = []
    invalid_etas = []
    over_constraint = []
    for eta in sorted(float(item) for item in etas):
        eta_dir = result_folder / eta_folder(eta)
        ci_path = eta_dir / "best_feasible_ci.txt"
        config_path = eta_dir / "noise_config.json"
        try:
            ci_value = float(ci_path.read_text().strip())
        except (OSError, ValueError):
            missing_etas.append(eta)
            rows.append((eta, np.nan))
            continue

        if not np.isfinite(ci_value):
            invalid_etas.append(eta)

        if config_path.exists():
            config = json.loads(config_path.read_text())
            ns_input = config.get("ns_input")
            np_input = config.get("np_input")
            if ns_input is not None and np_input is not None:
                ns_input = float(ns_input)
                np_input = float(np_input)
                if ns_input > n_s + 1e-6 or np_input > n_p + 1e-6:
                    over_constraint.append((eta, ns_input, np_input))

        rows.append((eta, ci_value))

    if missing_etas:
        print(
            f"Warning: missing {label} run-{run_id} eta values: "
            f"{_format_eta_list(missing_etas)}"
        )
    if invalid_etas:
        print(
            f"Warning: invalid {label} run-{run_id} CI values at eta: "
            f"{_format_eta_list(invalid_etas)}"
        )
    if over_constraint:
        eta_values = [item[0] for item in over_constraint]
        print(
            f"Warning: {label} run-{run_id} noisy input diagnostics exceed "
            f"n_S=n_P=2 at eta: {_format_eta_list(eta_values)}"
        )

    eta_values, ci_values = zip(*rows)
    return np.array(eta_values), np.array(ci_values)


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


def plot_noisy_vqt_curve(ax, curve):
    run_id = curve.get("run_id", VQT_A_LOSS_RUN_ID)
    eta_values, ci_values = load_noisy_ci_curve(run_id, curve["folder"], curve["label"])
    print(f"{curve['label']} run-{run_id} {curve['folder']}", ci_values)
    ax.plot(
        eta_values,
        ci_values,
        label=curve["label"],
        marker=curve["marker"],
        ls=curve["ls"],
        color=_curve_color(curve),
    )


def plot_noisy_gkp_curve(ax, curve):
    eta_values, ci_values = load_noisy_ci_curve(curve["run_id"], curve["folder"], curve["label"])
    print(f"{curve['label']} run-{curve['run_id']} {curve['folder']}", ci_values)
    ax.plot(
        eta_values,
        ci_values,
        label=curve["label"],
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
    plt.rcParams.update({
        'font.size': TICK_LABEL_SIZE,
        'axes.labelsize': AXIS_LABEL_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'xtick.labelsize': TICK_LABEL_SIZE,
        'ytick.labelsize': TICK_LABEL_SIZE,
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
    })
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)

    for ax, setup in zip(axes, NOISY_SETUPS):
        for curve in setup["vqt_curves"]:
            plot_noisy_vqt_curve(ax, curve)
        if setup.get("gkp_curve") is not None:
            plot_noisy_gkp_curve(ax, setup["gkp_curve"])
        plot_gaussian_benchmarks(ax, setup)
        ax.set_title(setup["title"], fontsize=TITLE_SIZE, pad=5)
        ax.set_xlabel(r"Transmissivity $\eta$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE, width=1.0, length=3.5)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Coherent Information (CI)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.80], w_pad=0.5)
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "CI_ns=np=2_Non-Adaptive_noisy_three_panel.jpg", dpi=500, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
