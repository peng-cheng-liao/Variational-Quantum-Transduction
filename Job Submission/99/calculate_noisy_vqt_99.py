import argparse
import csv
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


job_dir = Path(__file__).resolve().parent
repo_dir = job_dir.parents[1]
parameter_dir = job_dir / "parameters"
local_qtorch_dir = job_dir / "QTorch"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(job_dir / ".mplconfig"))

if str(job_dir) not in sys.path:
    sys.path.insert(0, str(job_dir))


SOURCE_RUN_ID = 84
REFERENCE_RUN_ID = 92
RUN_ID = 99
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)
TAU_A_VALUES = np.array([round(1.00 - 0.01 * i, 2) for i in range(21)])
DEFAULT_OUTPUT_ROOT = job_dir / "Data"
FUNCTION_NAME = "transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise"
MODEL_NAME = "initial_P_and_A_thermal_branches_plus_output_pure_loss_on_S_P_A"

CASES = [
    {
        "case_id": "eta_0p40_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan",
        "scan_type": "tau_a",
        "output_subdir": "eta_0p40_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan",
        "description": "tau_A scan, eta=0.40, n_P^th=n_A^th=0.01, tau_S=tau_P=0.99",
        "eta": 0.40,
        "initial_p_thermal_nbar": 0.01,
        "initial_a_thermal_nbar": 0.01,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "tau_a_values": TAU_A_VALUES,
    },
]
CASE_BY_ID = {case["case_id"]: case for case in CASES}


def eta_folder(eta):
    return f"eta={float(eta):.2f}"


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(repo_dir))
    except ValueError:
        return str(path)


def scalar_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def scan_values_for_case(case):
    if case["scan_type"] == "tau_a":
        return np.array(case["tau_a_values"], dtype=float)
    return np.array(case["eta_values"], dtype=float)


def scan_folder(case, scan_value):
    if case["scan_type"] == "tau_a":
        return f"tauA={float(scan_value):.2f}"
    return eta_folder(scan_value)


def point_eta_and_kappa_a(case, scan_value):
    if case["scan_type"] == "tau_a":
        return float(case["eta"]), float(scan_value)
    return float(scan_value), float(case["kappa_a"])


def load_thermal_noise_protocol():
    from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise

    protocol = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise
    signature = inspect.signature(protocol)
    required = (
        "initial_p_thermal_nbar",
        "initial_a_thermal_nbar",
        "kappa_o",
        "kappa_m",
        "kappa_a",
        "n_o",
        "n_m",
        "n_a",
    )
    missing = [name for name in required if name not in signature.parameters]
    if missing:
        raise RuntimeError(
            f"{FUNCTION_NAME} is missing required argument(s): {', '.join(missing)}. "
            f"Update {local_qtorch_dir / 'Transduction.py'} before running Job 99."
        )
    return protocol, signature


def load_source_info(eta):
    path = parameter_dir / eta_folder(eta) / "source_info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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


def load_parameters(eta, device, depth_override=None, nt_override=None):
    eta_dir = parameter_dir / eta_folder(eta)
    path = eta_dir / "parameters_best_feasible.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing local parameter file: {path}")

    values = np.load(path)
    parameters = torch.as_tensor(values, dtype=torch.float64, device=device)
    source_info = load_source_info(eta)
    depth, Nt = infer_depth_nt(source_info, parameters)
    if depth_override is not None:
        depth = depth_override
    if nt_override is not None:
        Nt = nt_override
    if len(parameters) < 24 * depth:
        raise ValueError(
            f"Parameter length {len(parameters)} is too short for depth={depth}; "
            f"expected at least {24 * depth}."
        )
    if len(parameters) != 24 * depth:
        parameters = parameters[: 24 * depth]
    return parameters, path, source_info, depth, Nt


def read_cached_result(point_out_dir):
    ci_path = point_out_dir / "best_feasible_ci.txt"
    config_path = point_out_dir / "noise_config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    return {
        "case_id": config.get("case_id", ""),
        "scan_type": config.get("scan_type", ""),
        "scan_value": config.get("scan_value", ""),
        "eta": config.get("eta", ""),
        "kappa_a": config.get("kappa_a", ""),
        "ci_noise": float(ci_path.read_text().strip()),
        "ns_input": config.get("ns_input", ""),
        "np_input": config.get("np_input", ""),
        "initial_p_thermal_nbar": config.get("initial_p_thermal_nbar", ""),
        "initial_a_thermal_nbar": config.get("initial_a_thermal_nbar", ""),
        "kappa_o": config.get("kappa_o", ""),
        "kappa_m": config.get("kappa_m", ""),
        "n_o": config.get("n_o", ""),
        "n_m": config.get("n_m", ""),
        "n_a": config.get("n_a", ""),
        "output_folder": relative_to_repo(point_out_dir),
        "source_parameter_file": config.get("source_parameter_file", ""),
    }


def calculate_scan_point(args, case, scan_value):
    eta, kappa_a = point_eta_and_kappa_a(case, scan_value)
    case_out_dir = args.output_root / case["output_subdir"]
    point_out_dir = case_out_dir / scan_folder(case, scan_value)
    ci_path = point_out_dir / "best_feasible_ci.txt"
    if ci_path.exists() and not args.recompute:
        print(
            f"{case['case_id']} {scan_folder(case, scan_value)}: "
            f"cache hit {relative_to_repo(ci_path)}",
            flush=True,
        )
        return read_cached_result(point_out_dir)

    parameters, parameter_path, source_info, depth, Nt = load_parameters(
        eta,
        args.device,
        depth_override=args.depth,
        nt_override=args.nt,
    )
    print(
        f"{case['case_id']} {scan_folder(case, scan_value)}: starting CI evaluation, "
        f"eta={eta:.2f} tau_A={kappa_a:.2f} "
        f"parameter_file={relative_to_repo(parameter_path)} depth={depth} Nt={Nt}",
        flush=True,
    )

    thermal_noise_protocol, signature = load_thermal_noise_protocol()
    kwargs = {
        "initial_p_thermal_nbar": case["initial_p_thermal_nbar"],
        "initial_a_thermal_nbar": case["initial_a_thermal_nbar"],
        "kappa_o": case["kappa_o"],
        "n_o": 0.0,
        "kappa_m": case["kappa_m"],
        "n_m": 0.0,
        "kappa_a": kappa_a,
        "n_a": 0.0,
        "kraus_prob_tol": args.kraus_prob_tol,
        "max_kraus_terms": args.max_kraus_terms,
        "initial_thermal_prob_tol": args.initial_thermal_prob_tol,
        "max_initial_thermal_branches": args.max_initial_thermal_branches,
    }
    for name, value in (
        ("env_cutoff_o", args.env_cutoff_o),
        ("env_cutoff_m", args.env_cutoff_m),
        ("env_cutoff_a", args.env_cutoff_a),
    ):
        if name in signature.parameters:
            kwargs[name] = value

    started = time.perf_counter()
    with torch.no_grad():
        CI, ns_input, np_input, state_RS, state_PA_return = thermal_noise_protocol(
            eta,
            parameters,
            depth,
            Nt,
            **kwargs,
        )
    elapsed = time.perf_counter() - started

    ci_value = scalar_float(CI)
    ns_value = scalar_float(ns_input)
    np_value = scalar_float(np_input)

    point_out_dir.mkdir(parents=True, exist_ok=True)
    ci_path.write_text(f"{ci_value}\n")
    (point_out_dir / "source_parameter_file.txt").write_text(
        f"{relative_to_repo(parameter_path)}\n"
    )

    config = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "reference_run_id": REFERENCE_RUN_ID,
        "case_id": case["case_id"],
        "case_description": case["description"],
        "scan_type": case["scan_type"],
        "scan_value": float(scan_value),
        "eta": eta,
        "initial_p_thermal_nbar": case["initial_p_thermal_nbar"],
        "initial_a_thermal_nbar": case["initial_a_thermal_nbar"],
        "kappa_o": case["kappa_o"],
        "kappa_m": case["kappa_m"],
        "kappa_a": kappa_a,
        "CI": ci_value,
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
        "kraus_prob_tol": args.kraus_prob_tol,
        "max_kraus_terms": args.max_kraus_terms,
        "initial_thermal_prob_tol": args.initial_thermal_prob_tol,
        "max_initial_thermal_branches": args.max_initial_thermal_branches,
        "env_cutoff_o": args.env_cutoff_o,
        "env_cutoff_m": args.env_cutoff_m,
        "env_cutoff_a": args.env_cutoff_a,
        "depth": depth,
        "Nt": Nt,
        "source_parameter_file": relative_to_repo(parameter_path),
        "source_info": source_info,
        "function": FUNCTION_NAME,
        "model": MODEL_NAME,
        "ns_input": ns_value,
        "np_input": np_value,
        "state_PA_return_is_none": state_PA_return is None,
        "state_RS_shape": list(state_RS.shape),
        "runtime_seconds": elapsed,
        "elapsed_seconds": elapsed,
    }
    (point_out_dir / "noise_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"{case['case_id']} {scan_folder(case, scan_value)}: "
        f"finished CI={ci_value:.12g} ns={ns_value:.12g} "
        f"np={np_value:.12g} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "case_id": case["case_id"],
        "scan_type": case["scan_type"],
        "scan_value": float(scan_value),
        "eta": eta,
        "kappa_a": kappa_a,
        "ci_noise": ci_value,
        "ns_input": ns_value,
        "np_input": np_value,
        "initial_p_thermal_nbar": case["initial_p_thermal_nbar"],
        "initial_a_thermal_nbar": case["initial_a_thermal_nbar"],
        "kappa_o": case["kappa_o"],
        "kappa_m": case["kappa_m"],
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
        "output_folder": relative_to_repo(point_out_dir),
        "source_parameter_file": relative_to_repo(parameter_path),
    }


def select_case(args):
    if args.case is not None and args.case_index is not None:
        raise SystemExit("Use only one of --case or --case-index.")

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(CASES):
            raise SystemExit(
                f"--case-index must be in [0, {len(CASES) - 1}], got {args.case_index}"
            )
        return CASES[args.case_index]
    return CASE_BY_ID[args.case or CASES[0]["case_id"]]


def select_scan_values(args, case):
    provided = [
        args.scan_index is not None,
        args.eta is not None,
        args.all_scan_points,
    ]
    if sum(provided) > 1:
        raise SystemExit("Use only one of --scan-index, --eta, or --all-scan-points.")

    values = scan_values_for_case(case)
    if args.scan_index is not None:
        if args.scan_index < 0 or args.scan_index >= len(values):
            raise SystemExit(
                f"--scan-index must be in [0, {len(values) - 1}], got {args.scan_index}"
            )
        return np.array([values[args.scan_index]], dtype=float)

    if args.eta is not None:
        return np.array([round(args.eta, 2)], dtype=float)

    return values


def write_summary(output_dir, rows):
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "noise_ci_summary.tsv"
    fieldnames = [
        "case_id",
        "scan_type",
        "scan_value",
        "eta",
        "kappa_a",
        "ci_noise",
        "ns_input",
        "np_input",
        "initial_p_thermal_nbar",
        "initial_a_thermal_nbar",
        "kappa_o",
        "kappa_m",
        "n_o",
        "n_m",
        "n_a",
        "source_parameter_file",
        "output_folder",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate noisy CI for Job 99 initial-P/A thermal VQT cases."
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case", choices=sorted(CASE_BY_ID))
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--scan-index", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--all-scan-points", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kraus-prob-tol", type=float, default=1e-12)
    parser.add_argument("--initial-thermal-prob-tol", type=float, default=1e-14)
    parser.add_argument("--max-kraus-terms", type=int)
    parser.add_argument("--max-initial-thermal-branches", type=int)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--env-cutoff-o", type=int)
    parser.add_argument("--env-cutoff-m", type=int)
    parser.add_argument("--env-cutoff-a", type=int)
    args = parser.parse_args()

    if args.list_cases:
        for idx, case in enumerate(CASES):
            values = scan_values_for_case(case)
            eta, first_kappa_a = point_eta_and_kappa_a(case, values[0])
            _, last_kappa_a = point_eta_and_kappa_a(case, values[-1])
            print(
                f"{idx}: {case['case_id']} scan_type={case['scan_type']} "
                f"eta={eta:.2f} "
                f"points={len(values)} initial_p_thermal_nbar={case['initial_p_thermal_nbar']} "
                f"initial_a_thermal_nbar={case['initial_a_thermal_nbar']} "
                f"kappa_o={case['kappa_o']} kappa_m={case['kappa_m']} "
                f"kappa_a={first_kappa_a:.2f}..{last_kappa_a:.2f} "
                f"n_o=0.0 n_m=0.0 n_a=0.0 output_subdir={case['output_subdir']}",
                flush=True,
            )
        raise SystemExit(0)

    if not local_qtorch_dir.is_dir():
        raise SystemExit(f"Missing local QTorch copy: {local_qtorch_dir}")
    if not parameter_dir.is_dir():
        raise SystemExit(f"Missing local parameter folder: {parameter_dir}")

    if args.num_threads is not None:
        if args.num_threads < 1:
            raise SystemExit("--num-threads must be positive.")
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)

    if not args.output_root.is_absolute():
        args.output_root = (job_dir / args.output_root).resolve()

    return args


def main():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_cpu = int(os.environ["SLURM_CPUS_PER_TASK"])
        torch.set_num_threads(num_cpu)
        torch.set_num_interop_threads(1)

    args = parse_args()
    case = select_case(args)
    scan_values = select_scan_values(args, case)
    case_out_dir = args.output_root / case["output_subdir"]
    print(f"Job directory: {job_dir}", flush=True)
    print(f"Using local QTorch: {local_qtorch_dir}", flush=True)
    print(f"Parameter directory: {parameter_dir}", flush=True)
    print(f"Case: {case['case_id']}", flush=True)
    print(f"Description: {case['description']}", flush=True)
    print(
        f"Noise parameters: initial_p_thermal_nbar={case['initial_p_thermal_nbar']} "
        f"initial_a_thermal_nbar={case['initial_a_thermal_nbar']} "
        f"kappa_o={case['kappa_o']} kappa_m={case['kappa_m']} "
        f"n_o=0.0 n_m=0.0 n_a=0.0",
        flush=True,
    )
    print(f"Output directory: {relative_to_repo(case_out_dir)}", flush=True)
    print(
        f"Scan points: {' '.join(scan_folder(case, value) for value in scan_values)}",
        flush=True,
    )

    if args.dry_run:
        for scan_value in scan_values:
            eta, kappa_a = point_eta_and_kappa_a(case, scan_value)
            parameters, parameter_path, source_info, depth, Nt = load_parameters(
                eta,
                args.device,
                depth_override=args.depth,
                nt_override=args.nt,
            )
            selection = source_info.get("selection_summary", {})
            print(
                f"{scan_folder(case, scan_value)}: eta={eta:.2f} tau_A={kappa_a:.2f} "
                f"parameter_file={relative_to_repo(parameter_path)} "
                f"output_dir={relative_to_repo(case_out_dir / scan_folder(case, scan_value))} "
                f"depth={depth} Nt={Nt} best_seed={selection.get('best_seed', '')} "
                f"parameter_count={len(parameters)}",
                flush=True,
            )
        print("Dry run only; no CI values computed.", flush=True)
        return

    rows = [calculate_scan_point(args, case, scan_value) for scan_value in scan_values]
    if len(rows) > 1:
        summary_path = write_summary(case_out_dir, rows)
        print(f"Wrote summary: {relative_to_repo(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
