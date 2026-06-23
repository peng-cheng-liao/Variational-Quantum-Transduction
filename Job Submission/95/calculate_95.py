import argparse
import csv
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
RUN_ID = 95
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)
DEFAULT_PARAMETER_ROOT = repo_dir / "Data_HPC" / str(SOURCE_RUN_ID)
DEFAULT_OUTPUT_ROOT = repo_dir / "Data_HPC" / str(RUN_ID)
DEFAULT_SOURCE_INFO_ROOT = repo_dir / "Job Submission" / "92" / "parameters"

SETUP_PRESETS = [
    {
        "name": "noisy",
        "output_subdir": "noisy_nPth=0p1_kS=0p99_kP=0p99_kA=0p99",
        "initial_p_thermal_nbar": 0.1,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "kappa_a": 0.99,
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
    },
    {
        "name": "noisy_nPth_0p01",
        "output_subdir": "noisy_nPth=0p01_kS=0p99_kP=0p99_kA=0p99",
        "initial_p_thermal_nbar": 0.01,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "kappa_a": 0.99,
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
    },
    {
        "name": "noisy_nPth_0p001",
        "output_subdir": "noisy_nPth=0p001_kS=0p99_kP=0p99_kA=0p99",
        "initial_p_thermal_nbar": 0.001,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "kappa_a": 0.99,
        "n_o": 0.0,
        "n_m": 0.0,
        "n_a": 0.0,
    },
    {
        "name": "noiseless_reference",
        "output_subdir": "noiseless_nPth=0_kS=1_kP=1_kA=1",
        "initial_p_thermal_nbar": 0.0,
        "kappa_o": 1.0,
        "kappa_m": 1.0,
        "kappa_a": 1.0,
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


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_thermal_noise_protocol():
    from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise

    return transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise


def read_selection_summary(path):
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["eta_folder"]: row for row in csv.DictReader(f, delimiter="\t")}


def load_source_info(args, eta):
    eta_name = eta_folder(eta)
    source_info_path = args.source_info_root / eta_name / "source_info.json"
    source_info = {}
    if source_info_path.exists():
        source_info = json.loads(source_info_path.read_text())

    selection = args.selection_summary_by_eta.get(eta_name)
    if selection:
        source_info.setdefault("selection_summary", selection)

    source_info.setdefault("source_run_id", SOURCE_RUN_ID)
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


def result_paths(args, eta):
    eta_dir = args.output_dir / eta_folder(eta)
    return {
        "eta_dir": eta_dir,
        "pt": eta_dir / "result.pt",
        "ci": eta_dir / "best_feasible_ci.txt",
        "config": eta_dir / "noise_config.json",
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
        "setup_name": payload.get("setup_name", ""),
        "ci": payload.get("CI", ""),
        "ns_input": payload.get("ns_input", ""),
        "np_input": payload.get("np_input", ""),
        "initial_p_thermal_nbar": payload.get("initial_p_thermal_nbar", ""),
        "kappa_o": payload.get("kappa_o", ""),
        "kappa_m": payload.get("kappa_m", ""),
        "kappa_a": payload.get("kappa_a", ""),
        "n_o": payload.get("n_o", ""),
        "n_m": payload.get("n_m", ""),
        "n_a": payload.get("n_a", ""),
        "depth": payload.get("depth", ""),
        "Nt": payload.get("Nt", ""),
        "source_parameter_file": payload.get("source_parameter_file", ""),
        "output_file": payload.get("output_file", ""),
        "runtime_seconds": payload.get("runtime_seconds", ""),
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def calculate_eta(args, eta):
    paths = result_paths(args, eta)
    if is_completed(paths) and not args.overwrite:
        print(f"{eta_folder(eta)}: cache hit {relative_to_repo(paths['pt'])}", flush=True)
        return read_cached_result(paths)

    parameters, parameter_path, source_info, depth, Nt = load_parameters(args, eta)
    print(
        f"{eta_folder(eta)} [{args.setup_name}]: starting CI evaluation, "
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
            initial_p_thermal_nbar=args.initial_p_thermal_nbar,
            kappa_o=args.kappa_o,
            n_o=args.n_o,
            kappa_m=args.kappa_m,
            n_m=args.n_m,
            kappa_a=args.kappa_a,
            n_a=args.n_a,
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

    payload = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "setup_name": args.setup_name,
        "eta": float(eta),
        "CI": scalar_float(CI),
        "ns_input": scalar_float(ns_input),
        "np_input": scalar_float(np_input),
        "initial_p_thermal_nbar": args.initial_p_thermal_nbar,
        "kappa_o": args.kappa_o,
        "kappa_m": args.kappa_m,
        "kappa_a": args.kappa_a,
        "n_o": args.n_o,
        "n_m": args.n_m,
        "n_a": args.n_a,
        "env_cutoff_a": args.env_cutoff_a,
        "kraus_prob_tol": args.kraus_prob_tol,
        "max_kraus_terms": args.max_kraus_terms,
        "initial_thermal_prob_tol": args.initial_thermal_prob_tol,
        "max_initial_thermal_branches": args.max_initial_thermal_branches,
        "depth": depth,
        "Nt": Nt,
        "source_parameter_file": relative_to_repo(parameter_path),
        "source_info": source_info,
        "function": "transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise",
        "model": "initial_P_thermal_branches_plus_output_pure_loss_on_S_P_A",
        "state_PA_return_is_none": state_PA_return is None,
        "state_RS_shape": list(state_RS.shape),
        "debug": {
            key: scalar_float(value) if torch.is_tensor(value) and value.numel() == 1 else value
            for key, value in debug.items()
            if key not in {"rho_P", "rho_RP"}
        },
        "runtime_seconds": elapsed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": relative_to_repo(paths["pt"]),
    }

    paths["eta_dir"].mkdir(parents=True, exist_ok=True)
    torch.save(payload, paths["pt"])
    paths["ci"].write_text(f"{payload['CI']}\n")
    paths["source"].write_text(f"{payload['source_parameter_file']}\n")
    write_json(paths["config"], {key: value for key, value in payload.items() if key != "debug"})

    print(
        f"{eta_folder(eta)} [{args.setup_name}]: finished CI={payload['CI']:.12g} "
        f"ns={payload['ns_input']:.12g} np={payload['np_input']:.12g} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary_row_from_payload(payload)


def select_etas(args):
    provided = [args.eta is not None, args.eta_index is not None]
    if sum(provided) > 1:
        raise SystemExit("Use only one of --eta or --eta-index.")

    if args.eta is not None:
        etas = np.array([args.eta], dtype=float)
    elif args.eta_index is not None:
        if args.eta_index < 0 or args.eta_index >= len(DEFAULT_ETAS):
            raise SystemExit(
                f"--eta-index must be in [0, {len(DEFAULT_ETAS) - 1}], "
                f"got {args.eta_index}"
            )
        etas = np.array([DEFAULT_ETAS[args.eta_index]], dtype=float)
    else:
        etas = DEFAULT_ETAS
    return etas


def select_setups(args):
    if args.setup is not None and args.setup_index is not None:
        raise SystemExit("Use only one of --setup or --setup-index.")

    if args.setup_index is not None:
        if args.setup_index < 0 or args.setup_index >= len(SETUP_PRESETS):
            raise SystemExit(
                f"--setup-index must be in [0, {len(SETUP_PRESETS) - 1}], "
                f"got {args.setup_index}"
            )
        return [SETUP_PRESETS[args.setup_index]]
    if args.setup is not None:
        return [SETUP_BY_NAME[args.setup]]
    return SETUP_PRESETS


def task_args_for_setup(args, setup):
    task_args = argparse.Namespace(**vars(args))
    task_args.setup_name = setup["name"]
    for attr in (
        "initial_p_thermal_nbar",
        "kappa_o",
        "kappa_m",
        "kappa_a",
        "n_o",
        "n_m",
        "n_a",
    ):
        if getattr(args, attr) is None:
            setattr(task_args, attr, setup[attr])

    if task_args.output_dir is None:
        task_args.output_dir = task_args.output_root / setup["output_subdir"]
    return task_args


def write_summary(output_dir, rows):
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "noise_ci_summary.tsv"
    fieldnames = [
        "eta",
        "setup_name",
        "ci",
        "ns_input",
        "np_input",
        "initial_p_thermal_nbar",
        "kappa_o",
        "kappa_m",
        "kappa_a",
        "n_o",
        "n_m",
        "n_a",
        "depth",
        "Nt",
        "source_parameter_file",
        "output_file",
        "runtime_seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def print_setup_list():
    for idx, setup in enumerate(SETUP_PRESETS):
        print(
            f"{idx}: {setup['name']} "
            f"initial_p_thermal_nbar={setup['initial_p_thermal_nbar']} "
            f"kappa_o={setup['kappa_o']} kappa_m={setup['kappa_m']} "
            f"kappa_a={setup['kappa_a']} "
            f"n_o={setup['n_o']} n_m={setup['n_m']} n_a={setup['n_a']} "
            f"output_subdir={setup['output_subdir']}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local job 95: noisy VQT CI evaluation with auxiliary A pure loss."
    )
    parser.add_argument("--list-setups", action="store_true")
    parser.add_argument("--setup", choices=sorted(SETUP_BY_NAME), help="Restrict to one setup; default runs all job-92 presets.")
    parser.add_argument("--setup-index", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--eta-index", type=int)
    parser.add_argument("--max-items", "--limit", type=int, dest="max_items")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parameter-root", type=Path, default=DEFAULT_PARAMETER_ROOT)
    parser.add_argument("--source-info-root", type=Path, default=DEFAULT_SOURCE_INFO_ROOT)
    parser.add_argument("--initial-p-thermal-nbar", type=float)
    parser.add_argument("--initial-p-nbar", type=float, dest="initial_p_thermal_nbar")
    parser.add_argument("--kappa-o", type=float)
    parser.add_argument("--kappa-m", type=float)
    parser.add_argument("--kappa-a", type=float)
    parser.add_argument("--n-o", type=float)
    parser.add_argument("--n-m", type=float)
    parser.add_argument("--n-a", type=float)
    parser.add_argument("--env-cutoff-a", type=int)
    parser.add_argument("--kraus-prob-tol", type=float, default=1e-12)
    parser.add_argument("--initial-thermal-prob-tol", type=float, default=1e-14)
    parser.add_argument("--max-kraus-terms", type=int)
    parser.add_argument("--max-initial-thermal-branches", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--return-debug", action="store_true")
    args = parser.parse_args()

    if args.list_setups:
        print_setup_list()
        raise SystemExit(0)

    if args.num_threads is not None:
        if args.num_threads < 1:
            raise SystemExit("--num-threads must be positive.")
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)

    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items/--limit must be positive.")

    for attr in ("output_root", "output_dir", "parameter_root", "source_info_root"):
        value = getattr(args, attr)
        if value is not None and not value.is_absolute():
            setattr(args, attr, (repo_dir / value).resolve())

    if not args.parameter_root.is_dir():
        raise SystemExit(f"Missing parameter root: {args.parameter_root}")

    args.selection_summary_by_eta = read_selection_summary(args.parameter_root / "selection_summary.tsv")
    args.selected_setups = select_setups(args)
    if args.output_dir is not None and len(args.selected_setups) != 1:
        raise SystemExit("--output-dir is only valid when --setup or --setup-index selects one setup.")

    return args


def main():
    args = parse_args()
    etas = select_etas(args)
    planned = [(setup, eta) for setup in args.selected_setups for eta in etas]
    if args.max_items is not None:
        planned = planned[: args.max_items]

    print(f"Job directory: {job_dir}", flush=True)
    print(f"Repository root: {repo_dir}", flush=True)
    print(f"Using repository QTorch: {repo_dir / 'QTorch'}", flush=True)
    print(f"Parameter root: {relative_to_repo(args.parameter_root)}", flush=True)
    print(f"Output root: {relative_to_repo(args.output_root)}", flush=True)
    print(f"Setups: {' '.join(setup['name'] for setup in args.selected_setups)}", flush=True)
    print(
        f"Branch controls: kraus_prob_tol={args.kraus_prob_tol} "
        f"max_kraus_terms={args.max_kraus_terms} "
        f"initial_thermal_prob_tol={args.initial_thermal_prob_tol} "
        f"max_initial_thermal_branches={args.max_initial_thermal_branches}",
        flush=True,
    )
    print(f"Etas: {' '.join(eta_folder(eta) for eta in etas)}", flush=True)
    print(f"Estimated task count: {len(planned)}", flush=True)

    if args.dry_run:
        for setup, eta in planned:
            task_args = task_args_for_setup(args, setup)
            parameters, parameter_path, source_info, depth, Nt = load_parameters(task_args, eta)
            selection = source_info.get("selection_summary", {})
            print(
                f"{setup['name']} {eta_folder(eta)}: "
                f"output_dir={relative_to_repo(task_args.output_dir)} "
                f"parameter_file={relative_to_repo(parameter_path)} "
                f"parameter_count={len(parameters)} depth={depth} Nt={Nt} "
                f"initial_p_thermal_nbar={task_args.initial_p_thermal_nbar} "
                f"kappa_o={task_args.kappa_o} kappa_m={task_args.kappa_m} "
                f"kappa_a={task_args.kappa_a} n_o={task_args.n_o} "
                f"n_m={task_args.n_m} n_a={task_args.n_a} "
                f"best_seed={selection.get('best_seed', '')}",
                flush=True,
            )
        print("Dry run only; no CI values computed.", flush=True)
        return

    rows = []
    for setup, eta in planned:
        task_args = task_args_for_setup(args, setup)
        rows.append(calculate_eta(task_args, eta))
    summary_path = write_summary(args.output_root, rows)
    if summary_path is not None:
        print(f"Wrote combined summary: {relative_to_repo(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
