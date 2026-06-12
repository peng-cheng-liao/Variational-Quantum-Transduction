import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
os.environ.setdefault("MPLCONFIGDIR", str(script_dir / ".mplconfig"))
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise


data_dir = repo_dir / "Data_HPC"
source_run_id = 84
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)


def eta_folder(eta):
    return f"eta={eta:.2f}"


def format_float_for_path(x):
    text = f"{float(x):.12g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def noise_folder_name(output_prefix, initial_p_thermal_nbar, kappa_o, kappa_m):
    return (
        f"{output_prefix}"
        f"_nPth={format_float_for_path(initial_p_thermal_nbar)}"
        f"_kS={format_float_for_path(kappa_o)}"
        f"_kP={format_float_for_path(kappa_m)}"
    )


def relative_to_repo(path):
    try:
        return str(path.relative_to(repo_dir))
    except ValueError:
        return str(path)


def scalar_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def load_selection_summary(run_id):
    summary_path = data_dir / str(run_id) / "selection_summary.tsv"
    if not summary_path.exists():
        return {}

    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {row["eta_folder"]: row for row in rows}


def find_parameter_candidates(folder):
    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix in {".npy", ".pt", ".pth"}
        and any(token in p.name.lower() for token in ("param", "parameters"))
    )
    best_feasible = [
        p for p in candidates
        if "best" in p.name.lower() and "feasible" in p.name.lower()
    ]
    return best_feasible or candidates


def infer_depth_nt(metadata, parameters, parameter_path):
    search_text = " ".join(
        str(metadata.get(key, ""))
        for key in ("source_eta_folder", "parameter_source", "score_source")
    )
    search_text = f"{search_text} {parameter_path.name}"
    match = re.search(r"depth=(\d+)_Nt=(\d+)", search_text)
    if match:
        depth = int(match.group(1))
        Nt = int(match.group(2))
    else:
        if len(parameters) % 24 != 0:
            raise ValueError(f"Cannot infer depth from parameter length {len(parameters)}")
        depth = len(parameters) // 24
        Nt = 30
        print(
            f"Warning: could not infer Nt from metadata for {parameter_path}; "
            "defaulting to Nt=30 based on run-84 naming."
        )

    if len(parameters) != 24 * depth:
        raise ValueError(
            f"Parameter length {len(parameters)} is inconsistent with depth={depth}; "
            f"expected {24 * depth}."
        )
    return depth, Nt


def load_parameter_file(path, device):
    if path.suffix == ".npy":
        value = np.load(path)
        return torch.as_tensor(value, dtype=torch.float64, device=device)

    value = torch.load(path, map_location=device)
    if isinstance(value, dict):
        for key in ("x", "parameters", "params"):
            if key in value:
                value = value[key]
                break
        else:
            raise RuntimeError(f"No parameter tensor found in checkpoint keys: {sorted(value.keys())}")
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def load_vqt_best_parameters(run_id, eta, device="cpu"):
    folder = data_dir / str(run_id) / eta_folder(eta)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing eta folder: {folder}")

    candidates = find_parameter_candidates(folder)
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

    parameters = load_parameter_file(candidates[0], device=device)
    return parameters, candidates[0], metadata


def read_cached_result(eta_out_dir, eta, source_run_id, output_folder):
    ci_path = eta_out_dir / "best_feasible_ci.txt"
    config_path = eta_out_dir / "noise_config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    return {
        "eta": float(eta),
        "ci_noise": float(ci_path.read_text().strip()),
        "ns_input": config.get("ns_input", ""),
        "np_input": config.get("np_input", ""),
        "source_parameter_file": config.get("source_parameter_file", ""),
        "source_ci_file": config.get("source_best_feasible_ci_file", ""),
        "output_folder": relative_to_repo(output_folder),
        "source_run_id": source_run_id,
    }


def compute_noisy_ci_for_eta(
        eta,
        *,
        source_run_id=84,
        output_folder,
        initial_p_thermal_nbar,
        kappa_o,
        kappa_m,
        recompute=False,
        device="cpu",
):
    eta_out_dir = output_folder / eta_folder(eta)
    ci_path = eta_out_dir / "best_feasible_ci.txt"
    if ci_path.exists() and not recompute:
        return read_cached_result(eta_out_dir, eta, source_run_id, output_folder)

    parameters, parameter_path, metadata = load_vqt_best_parameters(source_run_id, eta, device=device)
    depth, Nt = infer_depth_nt(metadata, parameters, parameter_path)
    source_ci_path = data_dir / str(source_run_id) / eta_folder(eta) / "best_feasible_ci.txt"
    source_ci = float(source_ci_path.read_text().strip())

    with torch.no_grad():
        CI, ns_input, np_input, state_RS, state_PA_return = (
            transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
                eta,
                parameters,
                depth,
                Nt,
                initial_p_thermal_nbar=initial_p_thermal_nbar,
                kappa_o=kappa_o,
                n_o=0.0,
                kappa_m=kappa_m,
                n_m=0.0,
            )
        )

    ci_value = scalar_float(CI)
    ns_value = scalar_float(ns_input)
    np_value = scalar_float(np_input)

    eta_out_dir.mkdir(parents=True, exist_ok=True)
    ci_path.write_text(f"{ci_value}\n")
    (eta_out_dir / "source_parameter_file.txt").write_text(f"{relative_to_repo(parameter_path)}\n")

    config = {
        "source_run_id": source_run_id,
        "eta": float(eta),
        "initial_p_thermal_nbar": initial_p_thermal_nbar,
        "kappa_o": kappa_o,
        "n_o": 0.0,
        "kappa_m": kappa_m,
        "n_m": 0.0,
        "depth": depth,
        "Nt": Nt,
        "source_parameter_file": relative_to_repo(parameter_path),
        "source_best_feasible_ci_file": relative_to_repo(source_ci_path),
        "source_best_feasible_ci": source_ci,
        "function": "transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise",
        "model": "initial_P_thermal_branches_plus_output_pure_loss",
        "source_eta_folder": metadata.get("source_eta_folder"),
        "best_seed": metadata.get("best_seed"),
        "ns_input": ns_value,
        "np_input": np_value,
        "state_PA_return_is_none": state_PA_return is None,
        "state_RS_shape": list(state_RS.shape),
    }
    (eta_out_dir / "noise_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    return {
        "eta": float(eta),
        "ci_noise": ci_value,
        "ns_input": ns_value,
        "np_input": np_value,
        "source_parameter_file": relative_to_repo(parameter_path),
        "source_ci_file": relative_to_repo(source_ci_path),
        "output_folder": relative_to_repo(output_folder),
        "source_run_id": source_run_id,
    }


def write_summary(output_folder, rows):
    summary_path = output_folder / "noise_ci_summary.tsv"
    output_folder.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eta",
        "ci_noise",
        "ns_input",
        "np_input",
        "source_parameter_file",
        "source_ci_file",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", type=int, default=source_run_id)
    parser.add_argument("--output-root", type=Path, default=data_dir)
    parser.add_argument("--output-prefix", default="84_noise")
    parser.add_argument("--initial-p-nbar", type=float, default=0.1)
    parser.add_argument("--kappa-o", type=float, default=0.99)
    parser.add_argument("--kappa-m", type=float, default=0.99)
    parser.add_argument("--etas", type=float, nargs="*")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    etas = np.array(args.etas if args.etas else DEFAULT_ETAS, dtype=float)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_dir / output_root
    output_folder = output_root / noise_folder_name(
        args.output_prefix,
        args.initial_p_nbar,
        args.kappa_o,
        args.kappa_m,
    )

    print(f"Output folder: {relative_to_repo(output_folder)}")
    for eta in etas:
        parameters, parameter_path, metadata = load_vqt_best_parameters(
            args.source_run_id,
            eta,
            device=args.device,
        )
        depth, Nt = infer_depth_nt(metadata, parameters, parameter_path)
        print(
            f"{eta_folder(eta)}: parameter_file={relative_to_repo(parameter_path)} "
            f"depth={depth} Nt={Nt}"
        )

    if args.dry_run:
        print("Dry run only; no noisy CI values computed.")
        return

    rows = [
        compute_noisy_ci_for_eta(
            eta,
            source_run_id=args.source_run_id,
            output_folder=output_folder,
            initial_p_thermal_nbar=args.initial_p_nbar,
            kappa_o=args.kappa_o,
            kappa_m=args.kappa_m,
            recompute=args.recompute,
            device=args.device,
        )
        for eta in etas
    ]
    summary_path = write_summary(output_folder, rows)
    print(f"Wrote summary: {relative_to_repo(summary_path)}")


if __name__ == "__main__":
    main()
