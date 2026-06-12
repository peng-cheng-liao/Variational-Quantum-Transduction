import argparse
import csv
import json
import os
import re
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
import torch

from matplotlib import rc
from Quantum_Plotting import *
from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise

rc('text', usetex=True)

parent_dir = str(repo_dir)
data_dir = repo_dir / "Data_HPC"
fig_dir = repo_dir / "Figs"

etalist = np.around(np.arange(0.05, 1.0, 0.05), 2)
n_s = 2
n_p = 2
energy_tol = 0.01

VQT_RUN_ID = 84
VQT_NOISE_RUN_ID = "84_noise"
VQT_NOISE_INITIAL_P_NBAR = 0.1
VQT_NOISE_KAPPA_O = 0.99
VQT_NOISE_KAPPA_M = 0.99
VQT_NOISE_N_O = 0.0
VQT_NOISE_N_M = 0.0


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


def load_selection_summary(run_id):
    summary_path = data_dir / str(run_id) / "selection_summary.tsv"
    if not summary_path.exists():
        return {}

    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {row["eta_folder"]: row for row in rows}


def infer_depth_nt(metadata, parameters):
    source_eta_folder = metadata.get("source_eta_folder", "")
    match = re.search(r"depth=(\d+)_Nt=(\d+)", source_eta_folder)
    if match:
        return int(match.group(1)), int(match.group(2))

    if len(parameters) % 24 != 0:
        raise ValueError(f"Cannot infer depth from parameter length {len(parameters)}")
    return len(parameters) // 24, 30


def load_vqt_best_parameters(run_id, eta):
    folder = data_dir / str(run_id) / eta_folder(eta)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing eta folder: {folder}")

    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix == ".npy"
        and "param" in p.name.lower()
        and "best" in p.name.lower()
        and "feasible" in p.name.lower()
    )
    if not candidates:
        candidates = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix == ".npy" and "param" in p.name.lower()
        )
    if len(candidates) != 1:
        candidate_list = "\n".join(str(p) for p in candidates) or "(none)"
        raise RuntimeError(
            f"Expected one best/feasible parameter file in {folder}; found {len(candidates)}:\n"
            f"{candidate_list}"
        )

    summary = load_selection_summary(run_id)
    metadata = summary.get(eta_folder(eta), {})
    if metadata and Path(metadata.get("parameter_source", "")).name != candidates[0].name:
        raise RuntimeError(
            f"Parameter file mismatch for {eta_folder(eta)}: local {candidates[0].name}, "
            f"summary source {metadata.get('parameter_source')}"
        )

    parameters_np = np.load(candidates[0])
    parameters = torch.as_tensor(parameters_np, dtype=torch.float64)
    return parameters, candidates[0], metadata


def compute_vqt_noise_ci_for_eta(eta, run_id=VQT_RUN_ID, noise_run_id=VQT_NOISE_RUN_ID):
    parameters, parameter_path, metadata = load_vqt_best_parameters(run_id, eta)
    depth, Nt = infer_depth_nt(metadata, parameters)
    source_ci_path = data_dir / str(run_id) / eta_folder(eta) / "best_feasible_ci.txt"
    source_ci = float(source_ci_path.read_text().strip())

    with torch.no_grad():
        ci, ns_input, np_input, _, _ = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
            eta,
            parameters,
            depth,
            Nt,
            initial_p_thermal_nbar=VQT_NOISE_INITIAL_P_NBAR,
            kappa_o=VQT_NOISE_KAPPA_O,
            n_o=VQT_NOISE_N_O,
            kappa_m=VQT_NOISE_KAPPA_M,
            n_m=VQT_NOISE_N_M,
        )

    ci_value = float(ci.detach().cpu())
    output_dir = data_dir / str(noise_run_id) / eta_folder(eta)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_feasible_ci.txt").write_text(f"{ci_value}\n")
    (output_dir / "source_parameter_file.txt").write_text(f"{parameter_path.relative_to(repo_dir)}\n")
    (output_dir / "source_best_feasible_ci.txt").write_text(f"{source_ci}\n")

    config = {
        "source_run_id": run_id,
        "eta": float(eta),
        "depth": depth,
        "Nt": Nt,
        "initial_p_thermal_nbar": VQT_NOISE_INITIAL_P_NBAR,
        "kappa_o": VQT_NOISE_KAPPA_O,
        "n_o": VQT_NOISE_N_O,
        "kappa_m": VQT_NOISE_KAPPA_M,
        "n_m": VQT_NOISE_N_M,
        "function": "transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise",
        "source_parameter_file": str(parameter_path.relative_to(repo_dir)),
        "source_best_feasible_ci": source_ci,
        "source_eta_folder": metadata.get("source_eta_folder"),
        "best_seed": metadata.get("best_seed"),
        "ns_input": float(ns_input.detach().cpu()),
        "np_input": float(np_input.detach().cpu()),
    }
    (output_dir / "noise_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return ci_value


def load_or_compute_vqt_noise_ci(etas=etalist, recompute=False):
    ci_list = []
    for eta in etas:
        path = data_dir / VQT_NOISE_RUN_ID / eta_folder(eta) / "best_feasible_ci.txt"
        if path.exists() and not recompute:
            ci_list.append(float(path.read_text().strip()))
        else:
            ci_list.append(compute_vqt_noise_ci_for_eta(eta))
    return np.array(ci_list)


def load_gkp_selected_ci(etas=etalist):
    summary_path = data_dir / "64_v2" / "selection_summary.tsv"
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    score_by_eta = {round(float(row["eta"]), 2): float(row["score"]) for row in rows}
    return np.array([score_by_eta.get(round(float(eta), 2), np.nan) for eta in etas])


# ECD MM

def plot_ECD_MM():
    """

    Nr = 20

    etalist1 = np.around(np.arange(0.1, 1.0, 0.1), 1)
    etalist2 = np.around(np.arange(0.02, 0.4, 0.02), 2)
    etalist = np.concatenate((etalist1, etalist2))
    etalist = np.sort(etalist)
    f_matrix_ECD = np.zeros((len(etalist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                fidelity_history = np.load(
                    parent_dir + f"/Data_HPC/43/Transduction_CoherentInfo_EA_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                fidelity_history[np.isnan(fidelity_history)] = 0
                f_matrix_ECD[i, j] = np.max(fidelity_history)
            except:
                try:
                    fidelity_history = np.load(
                        parent_dir + f"/Data_HPC/52/Transduction_CoherentInfo_EA_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                    fidelity_history[np.isnan(fidelity_history)] = 0
                    f_matrix_ECD[i, j] = np.max(fidelity_history)
                except:
                    print("ECD MM EA", eta, j, "fail")

    f_list_ECD = np.max(f_matrix_ECD, 1)
    print("ECD-MM", f_list_ECD)
    etalist = np.delete(etalist, 9)
    f_list_ECD = np.delete(f_list_ECD, 9)
    """

    # New Training 84
    ci_list2 = load_best_feasible_ci(VQT_RUN_ID)
    # plt.scatter(etalist, f_list_ECD, label="ECD-MM")
    #plt.plot(np.delete(etalist, 9), np.delete(f_list_ECD, 9), label="VQT-EA", marker='o',color = default_colors[0])
    plt.plot(etalist, ci_list2, label="VQT", marker='o', color=default_colors[0])


def plot_ECD_MM_noise(recompute=False):
    ci_noise = load_or_compute_vqt_noise_ci(etalist, recompute=recompute)
    print("VQT-noise", ci_noise)
    plt.plot(
        etalist,
        ci_noise,
        label=r"VQT-noise",
        marker="x",
        ls="-.",
        color=default_colors[4] if len(default_colors) > 4 else None,
    )


def plot_ECD_M():
    """


    Nr = 20
    etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)
    f_matrix_ECD = np.zeros((len(etalist), Nr))
    for i, eta in enumerate(etalist):
        for j in range(Nr):
            try:
                fidelity_history = np.load(
                    parent_dir + f"/Data_HPC/59/Transduction_CoherentInfo_EA_M_Nt=30_depth=20_eta={eta}_ns=2_np=2_N=10000_randomization={j}_ci_list.npy")
                fidelity_history[np.isnan(fidelity_history)] = 0
                f_matrix_ECD[i, j] = np.max(fidelity_history)
            except:
                print("ECD M EA", eta, j, "fail")

    f_list_ECD = np.max(f_matrix_ECD, 1)
    print("ECD M EA", f_list_ECD)
    """

    # New Training 87
    ci_list2 = load_best_feasible_ci(87)

    #plt.plot(etalist, f_list_ECD, label="VQT", marker="s",ls="--", color = default_colors[0])
    plt.plot(etalist, ci_list2, label="VQT without EA", marker="s", ls="--", color=default_colors[0])


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
                        mask = np.logical_or((np.array(ns_data) >= n_s+energy_tol), (np.array(np_data) >= n_p+energy_tol))
                        ci_data[mask] = 0
                        ci_matrix[i, j, k, l] = np.max(ci_data)
                    except:
                        ci_matrix[i, j, k, l] = 0

    ci_list = np.max(ci_matrix, axis=(1, 2, 3))
    print("GKP", ci_list)
    plt.plot(etalist, ci_list, label="GKP", ls="--", marker="*", color=default_colors[1])


def plot_GKP_n2_Nt30_v2():
    ci_list = load_gkp_selected_ci()
    print("GKP", ci_list)
    plt.plot(etalist, ci_list, label="GKP-QT", ls="--", marker="*", color=default_colors[1])


def plot_pure_loss_capacity(n_s):
    etalist = np.arange(0.05, 1.0, 0.05)
    cilist = []
    for eta in etalist:
        ci = max(0, g(eta * n_s) - g((1 - eta) * n_s))
        cilist.append(ci)
    plt.plot(etalist, cilist, label="QT", ls="--", marker="v", color=default_colors[3])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--recompute-noise", action="store_true")
    parser.add_argument("--only-noise-eta", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.only_noise_eta is not None:
        ci_noise = compute_vqt_noise_ci_for_eta(args.only_noise_eta)
        print(f"VQT-noise eta={args.only_noise_eta:.2f}: {ci_noise}")
        return

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
    if not args.skip_noise:
        plot_ECD_MM_noise(recompute=args.recompute_noise)
    plot_ECD_M()
    #plot_ECD_MM_fixedinput()
    # plot_GKP_n2_Nt30()
    plot_GKP_n2_Nt30_v2()
    plot_EA_TMS()
    plot_pure_loss_capacity(2)

    plt.legend(loc="upper left", frameon=False)
    #plt.title(f"Non-Adaptive Protocols")
    plt.tick_params(axis='both', which='major', )
    plt.xlabel("Transmissivity" + r" $\eta$", )
    plt.ylabel("Coherent Information(CI)")

    plt.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "CI_ns=np=2_Non-Adaptive.jpg", dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
