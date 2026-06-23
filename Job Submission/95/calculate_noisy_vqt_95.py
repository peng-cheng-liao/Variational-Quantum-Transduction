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
RUN_ID = 95
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)
DEFAULT_OUTPUT_ROOT = repo_dir / "Data_HPC" / str(RUN_ID)
FUNCTION_NAME = "transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise"
MODEL_NAME = "initial_P_thermal_branches_plus_output_pure_loss_on_S_P_A"

SETUP_PRESETS = [
    {
        "name": "kappaA_1p00",
        "output_subdir": "noisy_nPth=0p1_kS=0p99_kP=0p99_kA=1p00",
        "initial_p_nbar": 0.1,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "kappa_a": 1.0,
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
    },
    {
        "name": "kappaA_0p95",
        "output_subdir": "noisy_nPth=0p1_kS=0p99_kP=0p99_kA=0p95",
        "initial_p_nbar": 0.1,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "kappa_a": 0.95,
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
    },
    {
        "name": "kappaA_0p90",
        "output_subdir": "noisy_nPth=0p1_kS=0p99_kP=0p99_kA=0p90",
        "initial_p_nbar": 0.1,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "kappa_a": 0.9,
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
    },
]
SETUP_BY_NAME = {setup["name"]: setup for setup in SETUP_PRESETS}


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


def load_thermal_noise_protocol():
    from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise

    protocol = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise
    signature = inspect.signature(protocol)
    missing = [name for name in ("kappa_a", "n_a") if name not in signature.parameters]
    if missing:
        raise RuntimeError(
            f"{FUNCTION_NAME} is missing required argument(s): {', '.join(missing)}. "
            f"Update {local_qtorch_dir / 'Transduction.py'} before running Job 95."
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


def read_cached_result(eta_out_dir, eta):
    ci_path = eta_out_dir / "best_feasible_ci.txt"
    config_path = eta_out_dir / "noise_config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    return {
        "eta": float(eta),
        "setup_name": config.get("setup_name", ""),
        "kappa_a": config.get("kappa_a", ""),
        "ci_noise": float(ci_path.read_text().strip()),
        "ns_input": config.get("ns_input", ""),
        "np_input": config.get("np_input", ""),
        "initial_p_thermal_nbar": config.get("initial_p_thermal_nbar", ""),
        "kappa_o": config.get("kappa_o", ""),
        "kappa_m": config.get("kappa_m", ""),
        "n_o": config.get("n_o", ""),
        "n_m": config.get("n_m", ""),
        "n_a": config.get("n_a", ""),
        "output_folder": relative_to_repo(eta_out_dir),
        "source_parameter_file": config.get("source_parameter_file", ""),
    }


def calculate_eta(args, eta):
    eta_out_dir = args.output_dir / eta_folder(eta)
    ci_path = eta_out_dir / "best_feasible_ci.txt"
    if ci_path.exists() and not args.recompute:
        print(f"{eta_folder(eta)} [{args.setup_name}]: cache hit {relative_to_repo(ci_path)}", flush=True)
        return read_cached_result(eta_out_dir, eta)

    parameters, parameter_path, source_info, depth, Nt = load_parameters(
        eta,
        args.device,
        depth_override=args.depth,
        nt_override=args.nt,
    )
    print(
        f"{eta_folder(eta)} [{args.setup_name}]: starting CI evaluation, "
        f"parameter_file={relative_to_repo(parameter_path)} depth={depth} Nt={Nt}",
        flush=True,
    )

    thermal_noise_protocol, signature = load_thermal_noise_protocol()
    kwargs = {
        "initial_p_thermal_nbar": args.initial_p_nbar,
        "kappa_o": args.kappa_o,
        "n_o": args.n_o,
        "kappa_m": args.kappa_m,
        "n_m": args.n_m,
        "kappa_a": args.kappa_a,
        "n_a": args.n_a,
        "kraus_prob_tol": args.kraus_prob_tol,
        "max_kraus_terms": args.max_kraus_terms,
        "initial_thermal_prob_tol": args.initial_thermal_prob_tol,
        "max_initial_thermal_branches": args.max_initial_thermal_branches,
    }
    if "env_cutoff_a" in signature.parameters:
        kwargs["env_cutoff_a"] = args.env_cutoff_a

    started = time.perf_counter()
    with torch.no_grad():
        CI, ns_input, np_input, state_RS, state_PA_return = thermal_noise_protocol(
            float(eta),
            parameters,
            depth,
            Nt,
            **kwargs,
        )
    elapsed = time.perf_counter() - started

    ci_value = scalar_float(CI)
    ns_value = scalar_float(ns_input)
    np_value = scalar_float(np_input)

    eta_out_dir.mkdir(parents=True, exist_ok=True)
    ci_path.write_text(f"{ci_value}\n")
    (eta_out_dir / "source_parameter_file.txt").write_text(
        f"{relative_to_repo(parameter_path)}\n"
    )

    config = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "reference_run_id": REFERENCE_RUN_ID,
        "setup_name": args.setup_name,
        "eta": float(eta),
        "initial_p_thermal_nbar": args.initial_p_nbar,
        "kappa_o": args.kappa_o,
        "kappa_m": args.kappa_m,
        "kappa_a": args.kappa_a,
        "n_o": args.n_o,
        "n_m": args.n_m,
        "n_a": args.n_a,
        "kraus_prob_tol": args.kraus_prob_tol,
        "max_kraus_terms": args.max_kraus_terms,
        "initial_thermal_prob_tol": args.initial_thermal_prob_tol,
        "max_initial_thermal_branches": args.max_initial_thermal_branches,
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
    (eta_out_dir / "noise_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"{eta_folder(eta)} [{args.setup_name}]: finished CI={ci_value:.12g} "
        f"ns={ns_value:.12g} np={np_value:.12g} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "eta": float(eta),
        "setup_name": args.setup_name,
        "kappa_a": args.kappa_a,
        "ci_noise": ci_value,
        "ns_input": ns_value,
        "np_input": np_value,
        "initial_p_thermal_nbar": args.initial_p_nbar,
        "kappa_o": args.kappa_o,
        "kappa_m": args.kappa_m,
        "n_o": args.n_o,
        "n_m": args.n_m,
        "n_a": args.n_a,
        "output_folder": relative_to_repo(eta_out_dir),
        "source_parameter_file": relative_to_repo(parameter_path),
    }


def select_etas(args):
    provided = [args.eta is not None, args.eta_index is not None, args.all_etas]
    if sum(provided) > 1:
        raise SystemExit("Use only one of --eta, --eta-index, or --all-etas.")

    if args.eta is not None:
        return np.array([args.eta], dtype=float)
    if args.eta_index is not None:
        if args.eta_index < 0 or args.eta_index >= len(DEFAULT_ETAS):
            raise SystemExit(
                f"--eta-index must be in [0, {len(DEFAULT_ETAS) - 1}], "
                f"got {args.eta_index}"
            )
        return np.array([DEFAULT_ETAS[args.eta_index]], dtype=float)
    return DEFAULT_ETAS


def select_setup(args):
    if args.setup is not None and args.setup_index is not None:
        raise SystemExit("Use only one of --setup or --setup-index.")

    if args.setup_index is not None:
        if args.setup_index < 0 or args.setup_index >= len(SETUP_PRESETS):
            raise SystemExit(
                f"--setup-index must be in [0, {len(SETUP_PRESETS) - 1}], "
                f"got {args.setup_index}"
            )
        setup = SETUP_PRESETS[args.setup_index]
    else:
        setup = SETUP_BY_NAME[args.setup or "kappaA_1p00"]

    args.setup_name = setup["name"]
    for attr in ("initial_p_nbar", "kappa_o", "kappa_m", "kappa_a", "n_o", "n_m", "n_a"):
        if getattr(args, attr) is None:
            setattr(args, attr, setup[attr])

    if args.output_dir is None:
        args.output_dir = args.output_root / setup["output_subdir"]


def write_summary(output_dir, rows):
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "noise_ci_summary.tsv"
    fieldnames = [
        "eta",
        "setup_name",
        "kappa_a",
        "ci_noise",
        "ns_input",
        "np_input",
        "initial_p_thermal_nbar",
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
        description="Evaluate noisy CI for run-84 best VQT parameters as run 95."
    )
    parser.add_argument("--list-setups", action="store_true")
    parser.add_argument("--setup", choices=sorted(SETUP_BY_NAME))
    parser.add_argument("--setup-index", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--eta-index", type=int)
    parser.add_argument("--all-etas", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-p-nbar", type=float)
    parser.add_argument("--kappa-o", type=float)
    parser.add_argument("--kappa-m", type=float)
    parser.add_argument("--kappa-a", type=float)
    parser.add_argument("--n-o", type=float)
    parser.add_argument("--n-m", type=float)
    parser.add_argument("--n-a", type=float)
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
    parser.add_argument("--env-cutoff-a", type=int)
    args = parser.parse_args()

    if args.list_setups:
        for idx, setup in enumerate(SETUP_PRESETS):
            print(
                f"{idx}: {setup['name']} "
                f"initial_p_nbar={setup['initial_p_nbar']} "
                f"kappa_o={setup['kappa_o']} kappa_m={setup['kappa_m']} "
                f"kappa_a={setup['kappa_a']} "
                f"n_o={setup['n_o']} n_m={setup['n_m']} n_a={setup['n_a']} "
                f"output_subdir={setup['output_subdir']}",
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
        args.output_root = (repo_dir / args.output_root).resolve()

    select_setup(args)

    if not args.output_dir.is_absolute():
        args.output_dir = (repo_dir / args.output_dir).resolve()

    return args


def main():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_cpu = int(os.environ["SLURM_CPUS_PER_TASK"])
        torch.set_num_threads(num_cpu)
        torch.set_num_interop_threads(1)

    args = parse_args()
    etas = select_etas(args)
    print(f"Job directory: {job_dir}", flush=True)
    print(f"Using local QTorch: {local_qtorch_dir}", flush=True)
    print(f"Parameter directory: {parameter_dir}", flush=True)
    print(f"Setup: {args.setup_name}", flush=True)
    print(
        f"Noise parameters: initial_p_nbar={args.initial_p_nbar} "
        f"kappa_o={args.kappa_o} kappa_m={args.kappa_m} "
        f"kappa_a={args.kappa_a} n_o={args.n_o} n_m={args.n_m} n_a={args.n_a}",
        flush=True,
    )
    print(f"Output directory: {relative_to_repo(args.output_dir)}", flush=True)
    print(f"Etas: {' '.join(eta_folder(eta) for eta in etas)}", flush=True)

    if args.dry_run:
        for eta in etas:
            parameters, parameter_path, source_info, depth, Nt = load_parameters(
                eta,
                args.device,
                depth_override=args.depth,
                nt_override=args.nt,
            )
            selection = source_info.get("selection_summary", {})
            print(
                f"{eta_folder(eta)}: parameter_file={relative_to_repo(parameter_path)} "
                f"output_dir={relative_to_repo(args.output_dir / eta_folder(eta))} "
                f"depth={depth} Nt={Nt} best_seed={selection.get('best_seed', '')}",
                flush=True,
            )
        print("Dry run only; no CI values computed.", flush=True)
        return

    rows = [calculate_eta(args, eta) for eta in etas]
    if len(rows) > 1:
        summary_path = write_summary(args.output_dir, rows)
        print(f"Wrote summary: {relative_to_repo(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
