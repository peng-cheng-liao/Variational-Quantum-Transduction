import argparse
import csv
import inspect
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


job_dir = Path(__file__).resolve().parent
repo_dir = job_dir.parents[1]

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(job_dir / ".mplconfig"))

if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))


SOURCE_RUN_ID = 84
REFERENCE_RUN_ID = 92
RUN_ID = 95
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)
DEFAULT_KAPPA_A_VALUES = (1.0, 0.95, 0.9)
DEFAULT_PARAMETER_ROOT = repo_dir / "Job Submission" / str(REFERENCE_RUN_ID) / "parameters"
DEFAULT_OUTPUT_ROOT = repo_dir / "Data_HPC" / str(RUN_ID)

NOISE_CONFIG = {
    "initial_p_thermal_nbar": 0.1,
    "kappa_o": 0.99,
    "kappa_m": 0.99,
    "n_o": 0.0,
    "n_m": 0.0,
    "n_a": 0.0,
}
FUNCTION_NAME = "transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise"


def eta_folder(eta):
    return f"eta={float(eta):.2f}"


def compact_float(value):
    return f"{float(value):.2f}".replace(".", "p")


def output_subdir(kappa_a):
    return (
        "noisy_nPth=0p1_kS=0p99_kP=0p99_"
        f"kA={compact_float(kappa_a)}"
    )


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(repo_dir))
    except ValueError:
        return str(path)


def scalar_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_thermal_noise_protocol():
    from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise

    protocol = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise
    signature = inspect.signature(protocol)
    missing = [name for name in ("kappa_a", "n_a") if name not in signature.parameters]
    if missing:
        raise RuntimeError(
            f"{FUNCTION_NAME} is missing required argument(s): {', '.join(missing)}. "
            "Update QTorch/Transduction.py before running Job 95."
        )
    return protocol


def read_selection_summary(path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["eta_folder"]: row for row in csv.DictReader(f, delimiter="\t")}


def load_source_info(args, eta):
    eta_name = eta_folder(eta)
    source_info_path = args.parameter_root / eta_name / "source_info.json"
    source_info = {}
    if source_info_path.exists():
        source_info = json.loads(source_info_path.read_text())

    selection = args.selection_summary_by_eta.get(eta_name)
    if selection:
        source_info.setdefault("selection_summary", selection)

    source_info.setdefault("source_run_id", SOURCE_RUN_ID)
    source_info.setdefault("reference_run_id", REFERENCE_RUN_ID)
    source_info.setdefault(
        "original_source_parameter_path",
        relative_to_repo(args.parameter_root / eta_name / "parameters_best_feasible.npy"),
    )
    return source_info


def infer_depth_nt(source_info, parameters):
    search_parts = []
    for key in (
        "original_source_parameter_path",
        "original_noiseless_best_feasible_ci_path",
    ):
        if source_info.get(key):
            search_parts.append(str(source_info[key]))

    selection = source_info.get("selection_summary", {})
    for key in ("source_eta_folder", "parameter_source", "score_source"):
        if selection.get(key):
            search_parts.append(str(selection[key]))

    match = re.search(r"depth=(\d+)_Nt=(\d+)", " ".join(search_parts))
    if match:
        depth = int(match.group(1))
        Nt = int(match.group(2))
    else:
        if len(parameters) % 24 != 0:
            raise ValueError(f"Cannot infer depth from parameter length {len(parameters)}")
        depth = len(parameters) // 24
        Nt = 30
        print("Warning: could not infer Nt from metadata; defaulting to Nt=30.", flush=True)

    if len(parameters) != 24 * depth:
        raise ValueError(
            f"Parameter length {len(parameters)} is inconsistent with depth={depth}; "
            f"expected {24 * depth}."
        )
    return depth, Nt


def load_parameters(args, eta):
    eta_dir = args.parameter_root / eta_folder(eta)
    path = eta_dir / "parameters_best_feasible.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing parameter file: {path}")

    values = np.load(path)
    parameters = torch.as_tensor(values, dtype=torch.float64, device=args.device)
    source_info = load_source_info(args, eta)
    depth, Nt = infer_depth_nt(source_info, parameters)

    if args.depth is not None:
        depth = args.depth
    if args.nt is not None:
        Nt = args.nt
    if len(parameters) < 24 * depth:
        raise ValueError(
            f"Parameter length {len(parameters)} is too short for depth={depth}; "
            f"expected at least {24 * depth}."
        )
    if len(parameters) != 24 * depth:
        parameters = parameters[: 24 * depth]
    return parameters, path, source_info, depth, Nt


def task_output_dir(args, kappa_a):
    if args.output_dir is not None:
        return args.output_dir
    return args.output_root / output_subdir(kappa_a)


def result_paths(args, eta, kappa_a):
    eta_dir = task_output_dir(args, kappa_a) / eta_folder(eta)
    suffix = f"kappaA_{compact_float(kappa_a)}"
    return {
        "eta_dir": eta_dir,
        "pt": eta_dir / f"result_{suffix}.pt",
        "ci": eta_dir / f"best_feasible_ci_{suffix}.txt",
        "config": eta_dir / f"noise_config_{suffix}.json",
        "source": eta_dir / "source_parameter_file.txt",
    }


def is_completed(paths):
    return paths["pt"].exists() and paths["ci"].exists() and paths["config"].exists()


def read_cached_result(paths):
    payload = safe_torch_load(paths["pt"])
    return summary_row_from_payload(payload)


def summary_row_from_payload(payload):
    return {
        "eta": payload.get("eta", ""),
        "eta_index": payload.get("eta_index", ""),
        "kappa_a": payload.get("kappa_a", ""),
        "ci": payload.get("CI", ""),
        "coherent_information": payload.get("coherent_information", payload.get("CI", "")),
        "ns_input": payload.get("ns_input", ""),
        "np_input": payload.get("np_input", ""),
        "initial_p_thermal_nbar": payload.get("initial_p_thermal_nbar", ""),
        "kappa_o": payload.get("kappa_o", ""),
        "kappa_m": payload.get("kappa_m", ""),
        "n_o": payload.get("n_o", ""),
        "n_m": payload.get("n_m", ""),
        "n_a": payload.get("n_a", ""),
        "depth": payload.get("depth", ""),
        "Nt": payload.get("Nt", ""),
        "parameter_source_path": payload.get("parameter_source_path", ""),
        "seed": payload.get("seed", ""),
        "parameter_index": payload.get("parameter_index", ""),
        "output_file": payload.get("output_file", ""),
        "runtime_seconds": payload.get("runtime_seconds", ""),
        "timestamp_utc": payload.get("timestamp_utc", ""),
        "function": payload.get("function", ""),
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def calculate_task(args, eta_index, eta, kappa_a):
    paths = result_paths(args, eta, kappa_a)
    if is_completed(paths) and not args.overwrite:
        print(
            f"{eta_folder(eta)} kappa_a={kappa_a:g}: cache hit "
            f"{relative_to_repo(paths['pt'])}",
            flush=True,
        )
        return read_cached_result(paths)

    parameters, parameter_path, source_info, depth, Nt = load_parameters(args, eta)
    selection = source_info.get("selection_summary", {})
    print(
        f"{eta_folder(eta)} kappa_a={kappa_a:g}: starting CI evaluation, "
        f"parameter_file={relative_to_repo(parameter_path)} depth={depth} Nt={Nt}",
        flush=True,
    )

    thermal_noise_protocol = load_thermal_noise_protocol()
    started = time.perf_counter()
    with torch.no_grad():
        result = thermal_noise_protocol(
            float(eta),
            parameters,
            depth,
            Nt,
            initial_p_thermal_nbar=NOISE_CONFIG["initial_p_thermal_nbar"],
            kappa_o=NOISE_CONFIG["kappa_o"],
            n_o=NOISE_CONFIG["n_o"],
            kappa_m=NOISE_CONFIG["kappa_m"],
            n_m=NOISE_CONFIG["n_m"],
            kappa_a=float(kappa_a),
            n_a=NOISE_CONFIG["n_a"],
            env_cutoff_a=args.env_cutoff_a,
            kraus_prob_tol=args.kraus_prob_tol,
            max_kraus_terms=args.max_kraus_terms,
            initial_thermal_prob_tol=args.initial_thermal_prob_tol,
            max_initial_thermal_branches=args.max_initial_thermal_branches,
            return_debug=args.return_debug,
        )
    elapsed = time.perf_counter() - started

    if args.return_debug:
        CI, ns_input, np_input, state_RS, state_PA_return, debug = result
    else:
        CI, ns_input, np_input, state_RS, state_PA_return = result
        debug = {}

    ci_value = scalar_float(CI)
    ns_value = scalar_float(ns_input)
    np_value = scalar_float(np_input)
    payload = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "reference_run_id": REFERENCE_RUN_ID,
        "eta": float(eta),
        "eta_index": eta_index,
        "kappa_a": float(kappa_a),
        "kappa_o": NOISE_CONFIG["kappa_o"],
        "kappa_m": NOISE_CONFIG["kappa_m"],
        "n_a": NOISE_CONFIG["n_a"],
        "n_o": NOISE_CONFIG["n_o"],
        "n_m": NOISE_CONFIG["n_m"],
        "initial_p_thermal_nbar": NOISE_CONFIG["initial_p_thermal_nbar"],
        "CI": ci_value,
        "coherent_information": ci_value,
        "ns_input": ns_value,
        "np_input": np_value,
        "depth": depth,
        "Nt": Nt,
        "parameter_source_path": relative_to_repo(parameter_path),
        "source_parameter_file": relative_to_repo(parameter_path),
        "seed": selection.get("best_seed", ""),
        "parameter_index": selection.get("best_parameter_index", ""),
        "source_info": source_info,
        "function": FUNCTION_NAME,
        "model": "initial_P_thermal_branches_plus_output_pure_loss_on_S_P_A",
        "env_cutoff_a": args.env_cutoff_a,
        "kraus_prob_tol": args.kraus_prob_tol,
        "max_kraus_terms": args.max_kraus_terms,
        "initial_thermal_prob_tol": args.initial_thermal_prob_tol,
        "max_initial_thermal_branches": args.max_initial_thermal_branches,
        "state_PA_return_is_none": state_PA_return is None,
        "state_RS_shape": list(state_RS.shape),
        "runtime_seconds": elapsed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": relative_to_repo(paths["pt"]),
    }
    if args.return_debug:
        payload["debug"] = {
            key: scalar_float(value) if torch.is_tensor(value) and value.numel() == 1 else value
            for key, value in debug.items()
            if key not in {"rho_P", "rho_RP"}
        }

    paths["eta_dir"].mkdir(parents=True, exist_ok=True)
    torch.save(payload, paths["pt"])
    paths["ci"].write_text(f"{payload['CI']}\n")
    paths["source"].write_text(f"{payload['parameter_source_path']}\n")
    write_json(paths["config"], {key: value for key, value in payload.items() if key != "debug"})

    print(
        f"{eta_folder(eta)} kappa_a={kappa_a:g}: finished CI={ci_value:.12g} "
        f"ns={ns_value:.12g} np={np_value:.12g} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary_row_from_payload(payload)


def select_etas(args):
    provided = [args.eta is not None, args.eta_index is not None]
    if sum(provided) > 1:
        raise SystemExit("Use only one of --eta or --eta-index.")

    if args.eta is not None:
        matches = np.where(np.isclose(DEFAULT_ETAS, args.eta))[0]
        eta_index = int(matches[0]) if len(matches) else None
        return [(eta_index, float(args.eta))]
    if args.eta_index is not None:
        if args.eta_index < 0 or args.eta_index >= len(DEFAULT_ETAS):
            raise SystemExit(
                f"--eta-index must be in [0, {len(DEFAULT_ETAS) - 1}], "
                f"got {args.eta_index}"
            )
        return [(args.eta_index, float(DEFAULT_ETAS[args.eta_index]))]
    return [(idx, float(eta)) for idx, eta in enumerate(DEFAULT_ETAS)]


def select_kappa_a_values(args):
    if args.kappa_a is None:
        return list(DEFAULT_KAPPA_A_VALUES)

    requested = float(args.kappa_a)
    for value in DEFAULT_KAPPA_A_VALUES:
        if np.isclose(requested, value):
            return [float(value)]
    allowed = ", ".join(f"{value:g}" for value in DEFAULT_KAPPA_A_VALUES)
    raise SystemExit(f"--kappa-a must be one of: {allowed}")


def write_summary(output_root, rows):
    if not rows:
        return None
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "noise_ci_summary.tsv"
    fieldnames = [
        "eta",
        "eta_index",
        "kappa_a",
        "ci",
        "coherent_information",
        "ns_input",
        "np_input",
        "initial_p_thermal_nbar",
        "kappa_o",
        "kappa_m",
        "n_a",
        "n_o",
        "n_m",
        "depth",
        "Nt",
        "parameter_source_path",
        "seed",
        "parameter_index",
        "output_file",
        "runtime_seconds",
        "timestamp_utc",
        "function",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Job 95: Job-92-style noisy VQT CI sweep over auxiliary loss kappa_A."
    )
    parser.add_argument("--eta", type=float)
    parser.add_argument("--eta-index", type=int)
    parser.add_argument("--kappa-a", type=float, help="Run only one kappa_A value from the default sweep.")
    parser.add_argument("--max-items", "--limit", type=int, dest="max_items")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parameter-root", type=Path, default=DEFAULT_PARAMETER_ROOT)
    parser.add_argument("--env-cutoff-a", type=int)
    parser.add_argument("--kraus-prob-tol", type=float, default=1e-12)
    parser.add_argument("--initial-thermal-prob-tol", type=float, default=1e-14)
    parser.add_argument("--max-kraus-terms", type=int)
    parser.add_argument("--max-initial-thermal-branches", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--return-debug", action="store_true")
    args = parser.parse_args()

    if args.num_threads is not None:
        if args.num_threads < 1:
            raise SystemExit("--num-threads must be positive.")
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)

    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items/--limit must be positive.")

    for attr in ("output_root", "output_dir", "parameter_root"):
        value = getattr(args, attr)
        if value is not None and not value.is_absolute():
            setattr(args, attr, (repo_dir / value).resolve())

    if not args.parameter_root.is_dir():
        raise SystemExit(f"Missing parameter root: {args.parameter_root}")

    args.selection_summary_by_eta = read_selection_summary(args.parameter_root / "selection_summary.tsv")
    return args


def main():
    args = parse_args()
    eta_tasks = select_etas(args)
    kappa_a_values = select_kappa_a_values(args)
    planned = [(eta_index, eta, kappa_a) for eta_index, eta in eta_tasks for kappa_a in kappa_a_values]
    if args.max_items is not None:
        planned = planned[: args.max_items]

    print(f"Job directory: {job_dir}", flush=True)
    print(f"Repository root: {repo_dir}", flush=True)
    print(f"Using repository QTorch: {repo_dir / 'QTorch'}", flush=True)
    print(f"Parameter root: {relative_to_repo(args.parameter_root)}", flush=True)
    print(f"Output root: {relative_to_repo(args.output_root)}", flush=True)
    print(
        "Noise parameters: "
        f"initial_p_thermal_nbar={NOISE_CONFIG['initial_p_thermal_nbar']} "
        f"kappa_o={NOISE_CONFIG['kappa_o']} kappa_m={NOISE_CONFIG['kappa_m']} "
        f"n_o={NOISE_CONFIG['n_o']} n_m={NOISE_CONFIG['n_m']} n_a={NOISE_CONFIG['n_a']}",
        flush=True,
    )
    print(f"kappa_a values: {' '.join(f'{value:g}' for value in kappa_a_values)}", flush=True)
    print(
        f"Branch controls: kraus_prob_tol={args.kraus_prob_tol} "
        f"max_kraus_terms={args.max_kraus_terms} "
        f"initial_thermal_prob_tol={args.initial_thermal_prob_tol} "
        f"max_initial_thermal_branches={args.max_initial_thermal_branches}",
        flush=True,
    )
    print(f"Etas: {' '.join(eta_folder(eta) for _, eta in eta_tasks)}", flush=True)
    print(f"Estimated task count: {len(planned)}", flush=True)

    if args.dry_run:
        for eta_index, eta, kappa_a in planned:
            parameters, parameter_path, source_info, depth, Nt = load_parameters(args, eta)
            selection = source_info.get("selection_summary", {})
            paths = result_paths(args, eta, kappa_a)
            print(
                f"{eta_folder(eta)} eta_index={eta_index} kappa_a={kappa_a:g}: "
                f"output_file={relative_to_repo(paths['pt'])} "
                f"parameter_file={relative_to_repo(parameter_path)} "
                f"parameter_count={len(parameters)} depth={depth} Nt={Nt} "
                f"best_seed={selection.get('best_seed', '')}",
                flush=True,
            )
        print("Dry run only; no CI values computed.", flush=True)
        return

    rows = [calculate_task(args, eta_index, eta, kappa_a) for eta_index, eta, kappa_a in planned]
    summary_path = write_summary(args.output_root, rows)
    if summary_path is not None:
        print(f"Wrote combined summary: {relative_to_repo(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
