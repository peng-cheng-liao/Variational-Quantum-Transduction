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
GKP_ETA_SCAN_CASES = {
    0.1: {
        "run_id": GKP_NOISE_RUN_ID,
        "case_folder": "noisy_nPth=0p1_kS=0p99_kP=0p99",
    },
}

ETA_FIXED_FOR_TAUA_SCAN = 0.60
KAPPA_S = 0.99
KAPPA_P = 0.99
KAPPA_A_FOR_ETA_SCAN = 0.90

ETA_SCAN_CASES = [
    {
        "case_id": "case1",
        "case_folder": "case1_eta_scan_nthP_0_nthA_0_tauA_0p90",
        "n_th_p": 0.0,
        "n_th_a": 0.0,
        "kappa_a": KAPPA_A_FOR_ETA_SCAN,
    },
    {
        "case_id": "case2",
        "case_folder": "case2_eta_scan_nthP_0p1_nthA_0p1_tauA_0p90",
        "n_th_p": 0.1,
        "n_th_a": 0.1,
        "kappa_a": KAPPA_A_FOR_ETA_SCAN,
    },
]

TAUA_SCAN_CASES = [
    {
        "case_id": "case3",
        "case_folder": "case3_tauA_scan_eta_0p60_nthP_0_nthA_0",
        "n_th_p": 0.0,
        "n_th_a": 0.0,
        "eta": ETA_FIXED_FOR_TAUA_SCAN,
    },
    {
        "case_id": "case4",
        "case_folder": "case4_tauA_scan_eta_0p60_nthP_0p1_nthA_0p1",
        "n_th_p": 0.1,
        "n_th_a": 0.1,
        "eta": ETA_FIXED_FOR_TAUA_SCAN,
    },
]

etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
tau_a_values = np.array([round(1.00 - 0.01 * i, 2) for i in range(21)])
n_s = 2
n_p = 2
FIGSIZE = (10.8, 7.2)
AXIS_LABEL_SIZE = 14
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


def _format_nth(value):
    return "0" if abs(float(value)) < 1e-12 else f"{float(value):.1f}"


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


def direct_qt_effective_params(eta, n_th_p):
    tau = KAPPA_P * eta
    denom = 1.0 - tau
    if denom <= 1e-12:
        print(f"Warning: invalid QT effective denominator at eta={eta:.2f}")
        return np.nan, np.nan
    n_eff = KAPPA_P * (1.0 - eta) * n_th_p / denom
    return tau, n_eff


def gaussian_qt_curve(etas, n_th_p):
    values = []
    for eta in etas:
        tau, n_eff = direct_qt_effective_params(float(eta), float(n_th_p))
        values.append(thermal_loss_ci_bound(tau, n_eff, n_s=n_s))
    print(f"QT: direct thermal-loss baseline, n_P^th={n_th_p:.1f}, eta_points={len(etas)}")
    return np.array(values)


def tms_ea_effective_params(eta, kappa_a, n_th_p):
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
        (G_star * KAPPA_P * (1.0 - eta) / G) * n_th_p
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


def gaussian_tms_ea_eta_curve(etas, kappa_a, n_th_p):
    values = []
    tau_values = []
    n_eff_values = []
    invalid_etas = []
    for eta in etas:
        tau, n_eff = tms_ea_effective_params(float(eta), float(kappa_a), float(n_th_p))
        if not np.isfinite(tau) or not np.isfinite(n_eff):
            values.append(np.nan)
            invalid_etas.append(float(eta))
            continue
        tau_values.append(tau)
        n_eff_values.append(n_eff)
        values.append(thermal_loss_ci_bound(tau, n_eff, n_s=n_s))

    _summarize_tms_params(
        f"eta scan tau_A={kappa_a:.2f}, n_P^th={n_th_p:.1f}",
        tau_values,
        n_eff_values,
        invalid_etas,
    )
    return np.array(values)


def gaussian_tms_ea_tau_a_curve(tau_a_scan_values, eta, n_th_p):
    values = []
    tau_values = []
    n_eff_values = []
    invalid_tau_a = []
    for tau_a in tau_a_scan_values:
        tau, n_eff = tms_ea_effective_params(float(eta), float(tau_a), float(n_th_p))
        if not np.isfinite(tau) or not np.isfinite(n_eff):
            values.append(np.nan)
            invalid_tau_a.append(float(tau_a))
            continue
        tau_values.append(tau)
        n_eff_values.append(n_eff)
        values.append(thermal_loss_ci_bound(tau, n_eff, n_s=n_s))

    _summarize_tms_params(
        f"tau_A scan eta={eta:.2f}, n_P^th={n_th_p:.1f}",
        tau_values,
        n_eff_values,
        invalid_tau_a,
    )
    return np.array(values)


def _scan_has_all_ci(run_id, case_folder, scan_values, scan_type):
    result_folder = data_dir / str(run_id) / case_folder
    for value in [float(item) for item in scan_values]:
        if scan_type == "eta":
            ci_path = result_folder / eta_folder(value) / "best_feasible_ci.txt"
        elif scan_type == "tau_a":
            ci_path = result_folder / tau_a_folder(value) / "best_feasible_ci.txt"
        else:
            raise ValueError(f"Unsupported scan_type: {scan_type}")
        if not ci_path.exists():
            return False
    return True


def _gkp_case_for_nth(n_th_p):
    gkp_case = GKP_ETA_SCAN_CASES.get(round(float(n_th_p), 10))
    if gkp_case is None:
        return None
    if not _scan_has_all_ci(gkp_case["run_id"], gkp_case["case_folder"], etalist, "eta"):
        print(
            f"Skipping GKP for n_P^th={n_th_p:.1f}: "
            f"{gkp_case['case_folder']} is incomplete"
        )
        return None
    return gkp_case


def plot_eta_panel(ax, case):
    n_th_p = case["n_th_p"]
    n_th_a = case["n_th_a"]
    kappa_a = case["kappa_a"]
    vqt_eta, vqt_ci = load_ci_scan(
        VQT_NOISE_RUN_ID,
        case["case_folder"],
        etalist,
        "eta",
        f"VQT {case['case_id']} eta scan",
        expected_metadata={
            "kappa_s": KAPPA_S,
            "kappa_p": KAPPA_P,
            "kappa_a": kappa_a,
            "n_th_p": n_th_p,
            "n_th_a": n_th_a,
        },
    )
    tms_ci = gaussian_tms_ea_eta_curve(etalist, kappa_a, n_th_p)
    qt_ci = gaussian_qt_curve(etalist, n_th_p)

    ax.plot(vqt_eta, vqt_ci, label="VQT", marker="o", color=default_colors[0])
    gkp_case = _gkp_case_for_nth(n_th_p)
    if gkp_case is not None:
        gkp_eta, gkp_ci = load_ci_scan(
            gkp_case["run_id"],
            gkp_case["case_folder"],
            etalist,
            "eta",
            f"GKP n_P^th={n_th_p:.1f} eta scan",
            expected_metadata={
                "kappa_s": KAPPA_S,
                "kappa_p": KAPPA_P,
                "n_th_p": n_th_p,
            },
        )
        ax.plot(gkp_eta, gkp_ci, label="GKP", marker="*", ls="--", color=default_colors[1])
    else:
        print(f"Skipping GKP for {case['case_id']}: no corresponding n_P^th={n_th_p:.1f} data")

    ax.plot(etalist, tms_ci, label="TMS-EA", marker="^", color=default_colors[2])
    ax.plot(etalist, qt_ci, label="QT", marker="v", ls="--", color=default_colors[3])
    ax.set_title(
        rf"$n_P^{{\rm th}}=n_A^{{\rm th}}={_format_nth(n_th_p)},\ \tau_A={kappa_a:.2f}$",
        fontsize=TITLE_SIZE,
        pad=5,
    )
    ax.set_xlabel(r"Transduction efficiency $\eta$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
    return vqt_eta, vqt_ci


def plot_tau_a_panel(ax, case):
    n_th_p = case["n_th_p"]
    n_th_a = case["n_th_a"]
    eta = case["eta"]
    vqt_tau_a, vqt_ci = load_ci_scan(
        VQT_NOISE_RUN_ID,
        case["case_folder"],
        tau_a_values,
        "tau_a",
        f"VQT {case['case_id']} tau_A scan",
        expected_metadata={
            "eta": eta,
            "kappa_s": KAPPA_S,
            "kappa_p": KAPPA_P,
            "n_th_p": n_th_p,
            "n_th_a": n_th_a,
        },
    )
    tms_ci = gaussian_tms_ea_tau_a_curve(tau_a_values, eta, n_th_p)

    ax.plot(vqt_tau_a, vqt_ci, label="VQT", marker="o", color=default_colors[0])
    ax.plot(tau_a_values, tms_ci, label="TMS-EA", marker="^", color=default_colors[2])
    ax.set_title(
        rf"$\eta={eta:.2f},\ n_P^{{\rm th}}=n_A^{{\rm th}}={_format_nth(n_th_p)}$",
        fontsize=TITLE_SIZE,
        pad=5,
    )
    ax.set_xlabel(r"Ancillary transmissivity $\tau_A$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
    ax.set_xticks(np.arange(0.80, 1.001, 0.05))
    ax.set_xlim(1.01, 0.79)
    return vqt_tau_a, vqt_ci


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
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharey=True)

    plot_eta_panel(axes[0, 0], ETA_SCAN_CASES[0])
    plot_eta_panel(axes[0, 1], ETA_SCAN_CASES[1])
    tau_a_scan_values_0, _ = plot_tau_a_panel(axes[1, 0], TAUA_SCAN_CASES[0])
    tau_a_scan_values_1, _ = plot_tau_a_panel(axes[1, 1], TAUA_SCAN_CASES[1])
    print(
        f"tau_A scan endpoints: "
        f"{tau_a_scan_values_0[0]:.2f}-{tau_a_scan_values_0[-1]:.2f}, "
        f"{tau_a_scan_values_1[0]:.2f}-{tau_a_scan_values_1[-1]:.2f}"
    )

    for ax in axes.flat:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=1.0, length=3.5)
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("Coherent Information (CI)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    axes[1, 0].set_ylabel("Coherent Information (CI)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    handles, labels = _unique_legend_handles(axes.flat)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=0.8, h_pad=1.2)
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
