import json
import os
import sys
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

rc("text", usetex=True)

data_dir = repo_dir / "Data_HPC"
fig_dir = repo_dir / "Figs"

VQT_NOISE_RUN_ID = 99
GKP_NOISE_RUN_ID = 93
ETA_SCAN_CASE = "case2_eta_scan_nthP_0p1_nthA_0p1_tauA_0p90"
TAUA_SCAN_CASE = "case4_tauA_scan_eta_0p60_nthP_0p1_nthA_0p1"
GKP_ETA_SCAN_CASE = "noisy_nPth=0p1_kS=0p99_kP=0p99"

ETA_FIXED_FOR_TAUA_SCAN = 0.60
KAPPA_S = 0.99
KAPPA_P = 0.99
KAPPA_A_FOR_ETA_SCAN = 0.90
N_TH_P = 0.1
N_TH_A = 0.1

etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
tau_a_values = np.array([round(1.00 - 0.01 * i, 2) for i in range(21)])
n_s = 2
n_p = 2
FIGSIZE = (8.5, 3.8)
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 12
TITLE_SIZE = 13
LEGEND_SIZE = 13
LINE_WIDTH = 1.9
MARKER_SIZE = 5.5


def eta_folder(eta):
    return f"eta={float(eta):.2f}"


def tau_a_folder(tau_a):
    return f"tauA={float(tau_a):.2f}"


def _format_values(values):
    return ", ".join(f"{value:.2f}" for value in values) if values else "none"


def _metadata_close(config, keys, expected, tol=1e-9):
    for key in keys:
        if key in config:
            try:
                return abs(float(config[key]) - expected) <= tol
            except (TypeError, ValueError):
                return False
    return True


def _check_noise_config(config_path, scan_type, scan_value, expected, label):
    if not config_path.exists():
        return

    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not parse {label} metadata {config_path}: {exc}")
        return

    mismatches = []
    if config.get("scan_type") is not None and config["scan_type"] != scan_type:
        mismatches.append(f"scan_type={config['scan_type']}")
    if not _metadata_close(config, ["scan_value"], scan_value):
        mismatches.append(f"scan_value={config.get('scan_value')}")
    if "eta" in expected and not _metadata_close(config, ["eta"], expected["eta"]):
        mismatches.append(f"eta={config.get('eta')}")
    if "kappa_a" in expected and not _metadata_close(config, ["kappa_a"], expected["kappa_a"]):
        mismatches.append(f"kappa_a={config.get('kappa_a')}")
    if "kappa_s" in expected and not _metadata_close(config, ["kappa_s", "kappa_m"], expected["kappa_s"]):
        mismatches.append(f"kappa_s/kappa_m={config.get('kappa_s', config.get('kappa_m'))}")
    if "kappa_p" in expected and not _metadata_close(config, ["kappa_p", "kappa_o"], expected["kappa_p"]):
        mismatches.append(f"kappa_p/kappa_o={config.get('kappa_p', config.get('kappa_o'))}")
    if "n_th_p" in expected and not _metadata_close(config, ["initial_p_thermal_nbar"], expected["n_th_p"]):
        mismatches.append(f"initial_p_thermal_nbar={config.get('initial_p_thermal_nbar')}")
    if "n_th_a" in expected and not _metadata_close(config, ["initial_a_thermal_nbar"], expected["n_th_a"]):
        mismatches.append(f"initial_a_thermal_nbar={config.get('initial_a_thermal_nbar')}")

    if mismatches:
        print(
            f"Warning: metadata mismatch for {label} at {scan_type}={scan_value:.2f}: "
            + "; ".join(mismatches)
        )


def load_ci_scan(run_id, case_folder, scan_values, scan_type, label, expected_metadata=None):
    result_folder = data_dir / str(run_id) / case_folder
    expected_metadata = expected_metadata or {}
    x_values = []
    ci_values = []
    missing_values = []
    invalid_values = []

    for value in [float(item) for item in scan_values]:
        if scan_type == "eta":
            scan_dir = result_folder / eta_folder(value)
        elif scan_type == "tau_a":
            scan_dir = result_folder / tau_a_folder(value)
        else:
            raise ValueError(f"Unsupported scan_type: {scan_type}")

        ci_path = scan_dir / "best_feasible_ci.txt"
        config_path = scan_dir / "noise_config.json"
        x_values.append(value)

        try:
            ci_value = float(ci_path.read_text().strip())
        except (OSError, ValueError):
            missing_values.append(value)
            ci_values.append(np.nan)
            continue

        if not np.isfinite(ci_value):
            invalid_values.append(value)

        _check_noise_config(config_path, scan_type, value, expected_metadata, label)
        ci_values.append(ci_value)

    if missing_values:
        print(
            f"Warning: missing {label} run-{run_id} {scan_type} values: "
            f"{_format_values(missing_values)}"
        )
    if invalid_values:
        print(
            f"Warning: invalid {label} run-{run_id} CI values at {scan_type}: "
            f"{_format_values(invalid_values)}"
        )

    ci_values = np.array(ci_values)
    found_count = int(np.count_nonzero(np.isfinite(ci_values)))
    print(
        f"{label}: run={run_id}, folder={case_folder}, "
        f"{scan_type}_points={found_count}/{len(scan_values)}"
    )
    return np.array(x_values), ci_values


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


def thermal_loss_ci(N, tau, n_env):
    T = float(tau)
    N_B = float(n_env)
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


def thermal_loss_ci_bound(tau, n_env, n_s=2, num_grid=1001):
    if not np.isfinite(tau) or not np.isfinite(n_env):
        return np.nan
    if tau < -1e-12 or tau >= 1.0 or n_env < -1e-12:
        return np.nan
    tau = max(float(tau), 0.0)
    n_env = max(float(n_env), 0.0)
    n_grid = np.linspace(0.0, n_s, num_grid)
    ci_values = np.array([
        thermal_loss_ci(N, tau, n_env)
        for N in n_grid
    ])
    return max(0.0, float(np.nanmax(ci_values)))


def direct_qt_effective_params(eta):
    tau = KAPPA_P * eta
    denom = 1.0 - tau
    if denom <= 1e-12:
        print(f"Warning: invalid QT effective denominator at eta={eta:.2f}")
        return np.nan, np.nan
    n_eff = KAPPA_P * (1.0 - eta) * N_TH_P / denom
    return tau, n_eff


def gaussian_qt_curve(etas=etalist):
    values = []
    for eta in etas:
        tau, n_eff = direct_qt_effective_params(float(eta))
        values.append(thermal_loss_ci_bound(tau, n_eff, n_s=n_s))
    print(f"QT: direct thermal-loss baseline, eta_points={len(etas)}")
    return np.array(values)


def tms_ea_effective_params(eta, kappa_a):
    G = n_p + 1.0
    denom = kappa_a * G - KAPPA_P * (1.0 - eta) * (G - 1.0)
    if denom <= 0.0:
        print(
            f"Warning: invalid TMS-EA anti-squeezer denominator at "
            f"eta={eta:.2f}, tau_A={kappa_a:.2f}"
        )
        return np.nan, np.nan

    G_star = kappa_a * G / denom
    tau_ea_noisy = KAPPA_P * eta * G_star
    if tau_ea_noisy >= 1.0:
        print(
            f"Warning: TMS-EA point is outside the thermal-loss branch at "
            f"eta={eta:.2f}, tau_A={kappa_a:.2f}, tau={tau_ea_noisy:.6g}"
        )
        return np.nan, np.nan

    n_eff = (
        (G_star * KAPPA_P * (1.0 - eta) / G) * N_TH_P
        + (G_star - 1.0) * (1.0 - kappa_a)
    ) / (1.0 - KAPPA_P * eta * G_star)
    if n_eff < -1e-12:
        print(
            f"Warning: invalid negative TMS-EA n_eff at "
            f"eta={eta:.2f}, tau_A={kappa_a:.2f}, n_eff={n_eff:.6g}"
        )
        return np.nan, np.nan
    return tau_ea_noisy, max(n_eff, 0.0)


def _summarize_tms_params(label, tau_values, n_eff_values, invalid_x):
    if invalid_x:
        print(f"Warning: invalid TMS-EA {label} values: {_format_values(invalid_x)}")
    if tau_values:
        print(
            f"TMS-EA {label}: valid_points={len(tau_values)}, "
            f"tau_EA_noisy=[{min(tau_values):.6g}, {max(tau_values):.6g}], "
            f"n_eff=[{min(n_eff_values):.6g}, {max(n_eff_values):.6g}]"
        )
    else:
        print(f"Warning: no valid TMS-EA {label} points")


def gaussian_tms_ea_eta_curve(etas, kappa_a):
    values = []
    tau_values = []
    n_eff_values = []
    invalid_etas = []
    for eta in etas:
        tau, n_eff = tms_ea_effective_params(float(eta), float(kappa_a))
        if not np.isfinite(tau) or not np.isfinite(n_eff):
            values.append(np.nan)
            invalid_etas.append(float(eta))
            continue
        tau_values.append(tau)
        n_eff_values.append(n_eff)
        values.append(thermal_loss_ci_bound(tau, n_eff, n_s=n_s))

    _summarize_tms_params(f"eta scan tau_A={kappa_a:.2f}", tau_values, n_eff_values, invalid_etas)
    return np.array(values)


def gaussian_tms_ea_tau_a_curve(tau_a_scan_values, eta):
    values = []
    tau_values = []
    n_eff_values = []
    invalid_tau_a = []
    for tau_a in tau_a_scan_values:
        tau, n_eff = tms_ea_effective_params(float(eta), float(tau_a))
        if not np.isfinite(tau) or not np.isfinite(n_eff):
            values.append(np.nan)
            invalid_tau_a.append(float(tau_a))
            continue
        tau_values.append(tau)
        n_eff_values.append(n_eff)
        values.append(thermal_loss_ci_bound(tau, n_eff, n_s=n_s))

    _summarize_tms_params(f"tau_A scan eta={eta:.2f}", tau_values, n_eff_values, invalid_tau_a)
    return np.array(values)


def plot_eta_panel(ax):
    vqt_eta, vqt_ci = load_ci_scan(
        VQT_NOISE_RUN_ID,
        ETA_SCAN_CASE,
        etalist,
        "eta",
        "VQT eta scan",
        expected_metadata={
            "kappa_s": KAPPA_S,
            "kappa_p": KAPPA_P,
            "kappa_a": KAPPA_A_FOR_ETA_SCAN,
            "n_th_p": N_TH_P,
            "n_th_a": N_TH_A,
        },
    )
    gkp_eta, gkp_ci = load_ci_scan(
        GKP_NOISE_RUN_ID,
        GKP_ETA_SCAN_CASE,
        etalist,
        "eta",
        "GKP eta scan",
        expected_metadata={
            "kappa_s": KAPPA_S,
            "kappa_p": KAPPA_P,
            "n_th_p": N_TH_P,
        },
    )
    tms_ci = gaussian_tms_ea_eta_curve(etalist, KAPPA_A_FOR_ETA_SCAN)
    qt_ci = gaussian_qt_curve(etalist)

    ax.plot(vqt_eta, vqt_ci, label="VQT", marker="o", color=default_colors[0])
    ax.plot(gkp_eta, gkp_ci, label="GKP", marker="*", ls="--", color=default_colors[1])
    ax.plot(etalist, tms_ci, label="TMS-EA", marker="^", color=default_colors[2])
    ax.plot(etalist, qt_ci, label="QT", marker="v", ls="--", color=default_colors[3])
    ax.set_title(
        r"$n_P^{\rm th}=n_A^{\rm th}=0.1,\ \tau_S=\tau_P=0.99,\ \tau_A=0.90$",
        fontsize=TITLE_SIZE,
        pad=5,
    )
    ax.set_xlabel(r"Transduction efficiency $\eta$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
    return vqt_eta, vqt_ci


def plot_tau_a_panel(ax):
    vqt_tau_a, vqt_ci = load_ci_scan(
        VQT_NOISE_RUN_ID,
        TAUA_SCAN_CASE,
        tau_a_values,
        "tau_a",
        "VQT tau_A scan",
        expected_metadata={
            "eta": ETA_FIXED_FOR_TAUA_SCAN,
            "kappa_s": KAPPA_S,
            "kappa_p": KAPPA_P,
            "n_th_p": N_TH_P,
            "n_th_a": N_TH_A,
        },
    )
    tms_ci = gaussian_tms_ea_tau_a_curve(tau_a_values, ETA_FIXED_FOR_TAUA_SCAN)

    ax.plot(vqt_tau_a, vqt_ci, label="VQT", marker="o", color=default_colors[0])
    ax.plot(tau_a_values, tms_ci, label="TMS-EA", marker="^", color=default_colors[2])
    ax.set_title(
        r"$\eta=0.60,\ n_P^{\rm th}=n_A^{\rm th}=0.1,\ \tau_S=\tau_P=0.99$",
        fontsize=TITLE_SIZE,
        pad=5,
    )
    ax.set_xlabel(r"Auxiliary transmissivity $\tau_A$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
    return vqt_tau_a, vqt_ci


def _check_eta_tau_a_consistency(eta_values, eta_ci, tau_a_scan_values, tau_a_ci):
    eta_idx = np.where(np.isclose(eta_values, ETA_FIXED_FOR_TAUA_SCAN))[0]
    tau_a_idx = np.where(np.isclose(tau_a_scan_values, KAPPA_A_FOR_ETA_SCAN))[0]
    if len(eta_idx) == 0 or len(tau_a_idx) == 0:
        print("Warning: could not compare eta=0.60 and tau_A=0.90 VQT values")
        return

    eta_value = eta_ci[eta_idx[0]]
    tau_a_value = tau_a_ci[tau_a_idx[0]]
    diff = abs(eta_value - tau_a_value)
    print(
        f"VQT consistency eta=0.60 vs tau_A=0.90: "
        f"{eta_value:.16g} vs {tau_a_value:.16g}, diff={diff:.3g}"
    )
    if not np.isclose(eta_value, tau_a_value, rtol=1e-10, atol=1e-12):
        print("Warning: VQT eta/tau_A scan consistency check failed")


def _unique_legend_handles(axes):
    unique = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    order = ["VQT", "GKP", "TMS-EA", "QT"]
    return [unique[label] for label in order if label in unique], [label for label in order if label in unique]


def main():
    plt.rcParams.update({
        "font.size": TICK_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
    })
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    eta_values, eta_ci = plot_eta_panel(axes[0])
    tau_a_scan_values, tau_a_ci = plot_tau_a_panel(axes[1])
    _check_eta_tau_a_consistency(eta_values, eta_ci, tau_a_scan_values, tau_a_ci)
    print(f"tau_A scan endpoints: {tau_a_scan_values[0]:.2f}, {tau_a_scan_values[-1]:.2f}")

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=1.0, length=3.5)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Coherent Information (CI)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    handles, labels = _unique_legend_handles(axes)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=0.8)
    fig_dir.mkdir(exist_ok=True)
    output_paths = [
        fig_dir / "CI_ns=np=2_Non-Adaptive_noisy.jpg",
        fig_dir / "CI_ns=np=2_Non-Adaptive_noisy_three_panel.jpg",
    ]
    for output_path in output_paths:
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
