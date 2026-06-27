import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DATA_ROOT_CANDIDATES = [
    REPO_DIR / "Data_HPC" / "99" / "Data",
    REPO_DIR / "Data_HPC" / "99" / "data",
]
GKP_ROOT_CANDIDATES = [
    REPO_DIR / "Data_HPC" / "93" / "Data",
    REPO_DIR / "Data_HPC" / "93",
]
FIG_DIR = REPO_DIR / "Figs"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vqt_mplconfig_job99")

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc("text", usetex=True)

ETA_VALUES = np.around(np.arange(0.05, 1.0, 0.05), 2)
KAPPA_P = 0.99
KAPPA_A = 0.99
KAPPA_S = 0.99
N_S = 2.0
N_P = 2.0

ETA_SCAN_CASES = [
    {
        "case_id": "nthP_0p1_nthA_0p1_tauAll_0p99",
        "n_th": 0.1,
    },
    {
        "case_id": "nthP_0p05_nthA_0p05_tauAll_0p99",
        "n_th": 0.05,
    },
    {
        "case_id": "nthP_0p01_nthA_0p01_tauAll_0p99",
        "n_th": 0.01,
    },
]

TAU_A_VALUES = np.around(np.arange(0.80, 1.01, 0.01), 2)
TAU_A_SCAN_CASES = [
    {
        "case_id": "eta_0p30_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan",
        "eta": 0.30,
        "n_th": 0.01,
    },
    {
        "case_id": "eta_0p50_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan",
        "eta": 0.50,
        "n_th": 0.01,
    },
    {
        "case_id": "eta_0p70_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan",
        "eta": 0.70,
        "n_th": 0.01,
    },
]

COLORS = {
    "VQT": "#1f77b4",
    "GKP": "#ff7f0e",
    "TMS-EA": "#2ca02c",
}

FIGSIZE = (13.2, 7.4)
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
TITLE_SIZE = 13
LEGEND_SIZE = 13
LINE_WIDTH = 1.9
MARKER_SIZE = 5.5


def find_data_root():
    for path in DATA_ROOT_CANDIDATES:
        if path.is_dir():
            return path
    return DATA_ROOT_CANDIDATES[0]


def find_gkp_root():
    for path in GKP_ROOT_CANDIDATES:
        if path.is_dir():
            return path
    return None


def eta_folder(eta):
    return f"eta={float(eta):.2f}"


def value_tag(value):
    text = f"{float(value):.6g}"
    return text.replace(".", "p").replace("-", "m")


def format_nth(value):
    return f"{float(value):.3g}"


def eta_scan_title(case):
    return (
        rf"$n_P^{{\rm th}}=n_A^{{\rm th}}={format_nth(case['n_th'])}$"
        "\n"
        rf"$\kappa_S=\kappa_P=\kappa_A={KAPPA_S:.2f}$"
    )


def tau_a_scan_title(case):
    return (
        rf"$n_P^{{\rm th}}=n_A^{{\rm th}}={format_nth(case['n_th'])}$"
        "\n"
        rf"$\kappa_S=\kappa_P={KAPPA_S:.2f},\ \eta={case['eta']:.2f}$"
    )


def bosonic_entropy(nbar):
    scalar_input = np.isscalar(nbar)
    values = np.asarray(nbar, dtype=float)
    values = np.maximum(values, 0.0)
    out = np.zeros_like(values, dtype=float)
    mask = values > 0.0
    out[mask] = (
        (values[mask] + 1.0) * np.log2(values[mask] + 1.0)
        - values[mask] * np.log2(values[mask])
    )
    if scalar_input:
        return float(out)
    return out


def thermal_loss_ci(input_energy, transmissivity, env_nbar):
    t = float(transmissivity)
    n_b = float(env_nbar)
    n = float(input_energy)

    a = n + 0.5
    b = t * n + (1.0 - t) * n_b + 0.5
    c = np.sqrt(max(t * n * (n + 1.0), 0.0))

    delta = a * a + b * b - 2.0 * c * c
    det = (a * b - c * c) ** 2
    disc = max(delta * delta - 4.0 * det, 0.0)
    nu_plus = np.sqrt(max((delta + np.sqrt(disc)) / 2.0, 0.0))
    nu_minus = np.sqrt(max((delta - np.sqrt(disc)) / 2.0, 0.0))

    n_plus = max(nu_plus - 0.5, 0.0)
    n_minus = max(nu_minus - 0.5, 0.0)
    return bosonic_entropy(t * n + (1.0 - t) * n_b) - bosonic_entropy(n_plus) - bosonic_entropy(n_minus)


def thermal_loss_ci_bound(transmissivity, env_nbar, input_energy_limit=N_S, num_grid=1001):
    if not np.isfinite(transmissivity) or not np.isfinite(env_nbar):
        return np.nan
    if transmissivity < -1e-12 or transmissivity >= 1.0 or env_nbar < -1e-12:
        return np.nan

    t = max(float(transmissivity), 0.0)
    n_b = max(float(env_nbar), 0.0)
    grid = np.linspace(0.0, float(input_energy_limit), num_grid)
    values = np.array([thermal_loss_ci(n, t, n_b) for n in grid])
    return max(0.0, float(np.nanmax(values)))


def tms_ea_effective_params(eta, n_th, kappa_a=KAPPA_A):
    gain = N_P + 1.0
    denom = kappa_a * gain - KAPPA_P * (1.0 - eta) * (gain - 1.0)
    if denom <= 0.0:
        return np.nan, np.nan

    gain_star = kappa_a * gain / denom
    tau_eff = KAPPA_P * eta * gain_star
    if tau_eff >= 1.0:
        return np.nan, np.nan

    n_eff = (
        (gain_star * KAPPA_P * (1.0 - eta) / gain) * n_th
        + (gain_star - 1.0) * (1.0 - kappa_a)
    ) / (1.0 - KAPPA_P * eta * gain_star)
    return tau_eff, max(float(n_eff), 0.0)


def tms_ea_curve(etas, n_th, kappa_a=KAPPA_A):
    values = []
    for eta in etas:
        tau_eff, n_eff = tms_ea_effective_params(float(eta), float(n_th), float(kappa_a))
        values.append(thermal_loss_ci_bound(tau_eff, n_eff))
    return np.array(values)


def tms_ea_tau_a_curve(tau_a_values, eta, n_th):
    values = []
    for tau_a in tau_a_values:
        tau_eff, n_eff = tms_ea_effective_params(float(eta), float(n_th), float(tau_a))
        values.append(thermal_loss_ci_bound(tau_eff, n_eff))
    return np.array(values)


def load_vqt_case(data_root, case):
    case_dir = data_root / case["case_id"]
    etas = []
    ci_values = []
    missing = []
    metadata_warnings = []

    for eta in ETA_VALUES:
        eta_dir = case_dir / eta_folder(eta)
        ci_path = eta_dir / "best_feasible_ci.txt"
        config_path = eta_dir / "noise_config.json"
        etas.append(float(eta))
        try:
            ci_values.append(float(ci_path.read_text().strip()))
        except (OSError, ValueError):
            ci_values.append(np.nan)
            missing.append(float(eta))
            continue

        if config_path.exists():
            config = json.loads(config_path.read_text())
            checks = [
                ("eta", float(eta)),
                ("initial_p_thermal_nbar", case["n_th"]),
                ("initial_a_thermal_nbar", case["n_th"]),
                ("kappa_o", 0.99),
                ("kappa_m", 0.99),
                ("kappa_a", 0.99),
                ("n_o", 0.0),
                ("n_m", 0.0),
                ("n_a", 0.0),
            ]
            for key, expected in checks:
                if key in config and abs(float(config[key]) - expected) > 1e-9:
                    metadata_warnings.append(
                        f"{case['case_id']} {eta_folder(eta)} {key}={config[key]}"
                    )

    if missing:
        print(f"Warning: missing {case['case_id']} eta values: {missing}")
    for warning in metadata_warnings:
        print(f"Warning: metadata mismatch: {warning}")

    ci_array = np.array(ci_values)
    print(f"{case['case_id']}: {np.count_nonzero(np.isfinite(ci_array))}/{len(ETA_VALUES)} points")
    return np.array(etas), ci_array


def gkp_case_dirs(gkp_root, n_th):
    tag = value_tag(n_th)
    return [
        gkp_root / f"nPth_{tag}_kS_0p99_kP_0p99",
        gkp_root / f"noisy_nPth={tag}_kS=0p99_kP=0p99",
    ]


def load_gkp_case(gkp_root, case):
    if gkp_root is None:
        print(f"Skipping GKP {case['case_id']}: Data_HPC/93 folder not found")
        return None, None

    case_dirs = [path for path in gkp_case_dirs(gkp_root, case["n_th"]) if path.is_dir()]
    if not case_dirs:
        print(f"Skipping GKP nPth={case['n_th']}: no corresponding folder in {gkp_root}")
        return None, None

    case_dir = case_dirs[0]
    etas = []
    ci_values = []
    missing = []
    for eta in ETA_VALUES:
        ci_path = case_dir / eta_folder(eta) / "best_feasible_ci.txt"
        etas.append(float(eta))
        try:
            ci_values.append(float(ci_path.read_text().strip()))
        except (OSError, ValueError):
            ci_values.append(np.nan)
            missing.append(float(eta))

    if missing:
        print(f"Warning: missing GKP {case_dir.name} eta values: {missing}")

    ci_array = np.array(ci_values)
    count = int(np.count_nonzero(np.isfinite(ci_array)))
    if count == 0:
        print(f"Skipping GKP {case_dir.name}: no finite CI points")
        return None, None

    print(f"GKP {case_dir.name}: {count}/{len(ETA_VALUES)} points")
    return np.array(etas), ci_array


def load_gkp_eta_value(gkp_root, case):
    if gkp_root is None:
        print(f"Skipping GKP eta={case['eta']:.2f}: Data_HPC/93 folder not found")
        return np.nan

    case_dirs = [path for path in gkp_case_dirs(gkp_root, case["n_th"]) if path.is_dir()]
    if not case_dirs:
        print(f"Skipping GKP eta={case['eta']:.2f}, nPth={case['n_th']}: no corresponding folder")
        return np.nan

    case_dir = case_dirs[0]
    ci_path = case_dir / eta_folder(case["eta"]) / "best_feasible_ci.txt"
    try:
        value = float(ci_path.read_text().strip())
    except (OSError, ValueError):
        print(f"Skipping GKP {case_dir.name} eta={case['eta']:.2f}: missing CI")
        return np.nan

    print(f"GKP {case_dir.name} eta={case['eta']:.2f}: {value:.12g}")
    return value


def tau_a_folder(tau_a):
    return f"tauA={float(tau_a):.2f}"


def load_tau_a_case(data_root, case):
    case_dir = data_root / case["case_id"]
    tau_a_values = []
    ci_values = []
    missing = []
    metadata_warnings = []

    for tau_a in TAU_A_VALUES:
        tau_a_dir = case_dir / tau_a_folder(tau_a)
        ci_path = tau_a_dir / "best_feasible_ci.txt"
        config_path = tau_a_dir / "noise_config.json"
        tau_a_values.append(float(tau_a))
        try:
            ci_values.append(float(ci_path.read_text().strip()))
        except (OSError, ValueError):
            ci_values.append(np.nan)
            missing.append(float(tau_a))
            continue

        if config_path.exists():
            config = json.loads(config_path.read_text())
            checks = [
                ("eta", case["eta"]),
                ("scan_value", float(tau_a)),
                ("initial_p_thermal_nbar", case["n_th"]),
                ("initial_a_thermal_nbar", case["n_th"]),
                ("kappa_o", 0.99),
                ("kappa_m", 0.99),
                ("kappa_a", float(tau_a)),
                ("n_o", 0.0),
                ("n_m", 0.0),
                ("n_a", 0.0),
            ]
            for key, expected in checks:
                if key in config and abs(float(config[key]) - expected) > 1e-9:
                    metadata_warnings.append(
                        f"{case['case_id']} {tau_a_folder(tau_a)} {key}={config[key]}"
                    )

    if missing:
        print(f"Warning: missing {case['case_id']} tau_A values: {missing}")
    for warning in metadata_warnings:
        print(f"Warning: metadata mismatch: {warning}")

    ci_array = np.array(ci_values)
    print(f"{case['case_id']}: {np.count_nonzero(np.isfinite(ci_array))}/{len(TAU_A_VALUES)} points")
    return np.array(tau_a_values), ci_array


def unique_legend_handles(axes):
    unique = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    order = ["VQT", "GKP", "TMS-EA"]
    return [unique[label] for label in order if label in unique], [label for label in order if label in unique]


def main():
    data_root = find_data_root()
    if not data_root.is_dir():
        candidates = "\n".join(f"  {path}" for path in DATA_ROOT_CANDIDATES)
        raise FileNotFoundError(f"Missing data folder. Checked:\n{candidates}")
    print(f"Using data root: {data_root}")
    gkp_root = find_gkp_root()
    if gkp_root is None:
        print("GKP root not found; top-row GKP lines will be skipped")
    else:
        print(f"Using GKP root: {gkp_root}")

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

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharey=True)
    all_values = []

    for ax, case in zip(axes[0], ETA_SCAN_CASES):
        etas, vqt_ci = load_vqt_case(data_root, case)
        gkp_etas, gkp_ci = load_gkp_case(gkp_root, case)
        tms_ci = tms_ea_curve(ETA_VALUES, case["n_th"])
        all_values.extend(vqt_ci[np.isfinite(vqt_ci)])
        if gkp_ci is not None:
            all_values.extend(gkp_ci[np.isfinite(gkp_ci)])
        all_values.extend(tms_ci[np.isfinite(tms_ci)])

        ax.plot(etas, vqt_ci, marker="o", color=COLORS["VQT"], label="VQT")
        if gkp_ci is not None:
            ax.plot(gkp_etas, gkp_ci, marker="*", ls="--", color=COLORS["GKP"], label="GKP")
        ax.plot(ETA_VALUES, tms_ci, marker="^", color=COLORS["TMS-EA"], label="TMS-EA")
        ax.set_title(eta_scan_title(case), fontsize=TITLE_SIZE, pad=5)
        ax.set_xlabel(r"Transduction efficiency $\eta$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        ax.set_xlim(0.03, 0.97)
        ax.set_xticks(np.arange(0.1, 1.0, 0.2))

    for ax, case in zip(axes[1], TAU_A_SCAN_CASES):
        tau_a_values, vqt_ci = load_tau_a_case(data_root, case)
        gkp_ci = load_gkp_eta_value(gkp_root, case)
        tms_ci = tms_ea_tau_a_curve(TAU_A_VALUES, case["eta"], case["n_th"])
        all_values.extend(vqt_ci[np.isfinite(vqt_ci)])
        if np.isfinite(gkp_ci):
            all_values.append(gkp_ci)
        all_values.extend(tms_ci[np.isfinite(tms_ci)])

        ax.plot(tau_a_values, vqt_ci, marker="o", color=COLORS["VQT"], label="VQT")
        if np.isfinite(gkp_ci):
            ax.axhline(gkp_ci, ls="--", color=COLORS["GKP"], label="GKP")
        ax.plot(TAU_A_VALUES, tms_ci, marker="^", color=COLORS["TMS-EA"], label="TMS-EA")
        ax.set_title(tau_a_scan_title(case), fontsize=TITLE_SIZE, pad=5)
        ax.set_xlabel(r"Ancillary transmissivity $\kappa_A$", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        ax.set_xlim(1.01, 0.79)
        ax.set_xticks(np.arange(1.00, 0.79, -0.05))

    for ax in axes.flat:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, width=1.0, length=3.5)
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("Coherent Information (CI)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    axes[1, 0].set_ylabel("Coherent Information (CI)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    if all_values:
        ymin = min(all_values)
        ymax = max(all_values)
        pad = 0.06 * max(ymax - ymin, 1.0)
        axes[0, 0].set_ylim(ymin - pad, ymax + pad)

    handles, labels = unique_legend_handles(axes.flat)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=0.8, h_pad=1.2)

    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("jpg", "pdf"):
        output_path = FIG_DIR / f"CI_ns=np=2_Non-Adaptive_noisy.{suffix}"
        fig.savefig(output_path, dpi=500, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
