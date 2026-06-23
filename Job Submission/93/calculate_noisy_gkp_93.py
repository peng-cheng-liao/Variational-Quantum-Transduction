import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
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


from QTorch.Transduction import transduction_protocol_CoherentInfo_GKP2_thermal_noise


RUN_ID = 93
SOURCE_RUN_ID = 94
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)

DEFAULT_D1 = 2
DEFAULT_D2 = 1
DEFAULT_J2 = 0
DEFAULT_NT = 30
DEFAULT_NR = DEFAULT_D1

SETUP_PRESETS = [
    {
        "name": "noisy",
        "output_subdir": "noisy_nPth=0p1_kS=0p99_kP=0p99",
        "initial_p_nbar": 0.1,
        "kappa_o": 0.99,
        "kappa_m": 0.99,
        "n_o": 0.0,
        "n_m": 0.0,
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


def optional_float(value):
    if value in ("", None):
        return None
    return float(value)


def load_source_info(eta):
    path = parameter_dir / eta_folder(eta) / "source_info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _selection_int(source_info, key, default):
    selection = source_info.get("selection_summary", {})
    value = selection.get(key, "")
    if value in ("", None):
        value = source_info.get(key, "")
    if value in ("", None):
        return default
    return int(value)


def load_protocol_settings(source_info):
    source_d1 = _selection_int(source_info, "d1", DEFAULT_D1)
    source_d2 = _selection_int(source_info, "d2", DEFAULT_D2)
    source_j2 = _selection_int(source_info, "j2", DEFAULT_J2)
    Nt = int(source_info.get("gkp_constants", {}).get("Nt", DEFAULT_NT))
    d1 = source_d1
    d2 = source_d2
    j2 = source_j2
    NR = d1

    if not (0 <= j2 < d2):
        raise ValueError(f"Invalid GKP branch index j2={j2} for d2={d2}")

    return {
        "d1": d1,
        "d2": d2,
        "j2": j2,
        "Nt": Nt,
        "NR": NR,
        "source_d1": source_d1,
        "source_d2": source_d2,
        "source_j2": source_j2,
        "source_metadata_differs_from_default_protocol_settings": (source_d1, source_d2, source_j2) != (
            DEFAULT_D1,
            DEFAULT_D2,
            DEFAULT_J2,
        ),
        "uses_source_protocol_settings": True,
    }


def load_parameters(eta, device):
    eta_dir = parameter_dir / eta_folder(eta)
    path = eta_dir / "parameters_best_feasible.npy"
    if not path.exists():
        path = eta_dir / "parameters.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing local parameter file: {path}")

    values = np.load(path)
    if values.shape != (8,):
        raise ValueError(f"{path} has shape {values.shape}; expected (8,)")

    parameters = torch.as_tensor(values, dtype=torch.float64, device=device)
    source_info = load_source_info(eta)
    protocol_settings = load_protocol_settings(source_info)
    return parameters, path, source_info, protocol_settings


def source_score(source_info, *, required=False):
    selection = source_info.get("selection_summary", {})
    score = optional_float(selection.get("score", ""))
    if score is None:
        score = optional_float(source_info.get("score", ""))
    if required and score is None:
        raise ValueError("Missing source_info['selection_summary']['score']")
    return score


def validate_zero_noise_args(args):
    expected = {
        "initial_p_nbar": 0.0,
        "kappa_o": 1.0,
        "kappa_m": 1.0,
        "n_o": 0.0,
        "n_m": 0.0,
    }
    mismatches = []
    for attr, expected_value in expected.items():
        actual_value = getattr(args, attr)
        if actual_value != expected_value:
            mismatches.append(f"{attr}={actual_value} (expected {expected_value})")
    if mismatches:
        raise SystemExit(
            "--validate-zero-noise must be run with noiseless noise settings: "
            + ", ".join(mismatches)
        )


def read_cached_result(eta_out_dir, eta):
    ci_path = eta_out_dir / "best_feasible_ci.txt"
    config_path = eta_out_dir / "noise_config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    return {
        "eta": float(eta),
        "setup_name": config.get("setup_name", ""),
        "ci_noise": float(ci_path.read_text().strip()),
        "ns_input": config.get("ns_input", ""),
        "np_input": config.get("np_input", ""),
        "d1": config.get("d1", ""),
        "d2": config.get("d2", ""),
        "j2": config.get("j2", ""),
        "Nt": config.get("Nt", ""),
        "NR": config.get("NR", ""),
        "source_score": config.get("source_score", ""),
        "zero_noise_abs_error": config.get("zero_noise_abs_error", ""),
        "zero_noise_validation_passed": config.get("zero_noise_validation_passed", ""),
        "elapsed_seconds": config.get("elapsed_seconds", ""),
        "output_file": config.get("output_file", ""),
        "output_folder": relative_to_repo(eta_out_dir.parent),
        "source_parameter_file": config.get("source_parameter_file", ""),
    }


def debug_scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().reshape(()).item()
    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def calculate_eta(args, eta):
    eta_out_dir = args.output_dir / eta_folder(eta)
    ci_path = eta_out_dir / "best_feasible_ci.txt"
    if ci_path.exists() and not args.recompute:
        if args.validate_zero_noise:
            raise SystemExit(
                f"{eta_folder(eta)}: existing output requires --recompute for "
                "--validate-zero-noise so noise_config.json records fresh diagnostics"
            )
        print(f"{eta_folder(eta)}: cache hit {relative_to_repo(ci_path)}", flush=True)
        return read_cached_result(eta_out_dir, eta)

    parameters, parameter_path, source_info, protocol = load_parameters(eta, args.device)
    selected_source_score = source_score(source_info, required=args.validate_zero_noise)
    print(
        f"{eta_folder(eta)} [{args.setup_name}]: starting CI evaluation, "
        f"parameter_file={relative_to_repo(parameter_path)} "
        f"d1={protocol['d1']} d2={protocol['d2']} j2={protocol['j2']} "
        f"Nt={protocol['Nt']} NR={protocol['NR']} "
        f"source_score={selected_source_score if selected_source_score is not None else ''} "
        f"output_file={relative_to_repo(ci_path)}",
        flush=True,
    )

    start_time = datetime.now()
    started = time.perf_counter()
    with torch.no_grad():
        result = transduction_protocol_CoherentInfo_GKP2_thermal_noise(
            float(eta),
            protocol["d1"],
            protocol["d2"],
            protocol["j2"],
            parameters,
            protocol["Nt"],
            NR=protocol["NR"],
            initial_p_thermal_nbar=args.initial_p_nbar,
            kappa_o=args.kappa_o,
            n_o=args.n_o,
            kappa_m=args.kappa_m,
            n_m=args.n_m,
            kraus_prob_tol=args.kraus_prob_tol,
            max_kraus_terms=args.max_kraus_terms,
            initial_thermal_prob_tol=args.initial_thermal_prob_tol,
            max_initial_thermal_branches=args.max_initial_thermal_branches,
            return_debug=args.return_debug,
        )
    if args.return_debug:
        CI, ns_input, np_input, state_RS, state_P_return, debug = result
    else:
        CI, ns_input, np_input, state_RS, state_P_return = result
        debug = {}
    elapsed = time.perf_counter() - started
    finish_time = datetime.now()

    ci_value = scalar_float(CI)
    ns_value = scalar_float(ns_input)
    np_value = scalar_float(np_input)
    zero_noise_abs_error = None
    zero_noise_validation_passed = None
    if args.validate_zero_noise:
        zero_noise_abs_error = abs(ci_value - selected_source_score)
        zero_noise_validation_passed = zero_noise_abs_error <= args.zero_noise_tolerance

    eta_out_dir.mkdir(parents=True, exist_ok=True)
    ci_path.write_text(f"{ci_value}\n")
    (eta_out_dir / "source_parameter_file.txt").write_text(
        f"{relative_to_repo(parameter_path)}\n"
    )

    config = {
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "setup_name": args.setup_name,
        "eta": float(eta),
        "initial_p_thermal_nbar": args.initial_p_nbar,
        "kappa_o": args.kappa_o,
        "n_o": args.n_o,
        "kappa_m": args.kappa_m,
        "n_m": args.n_m,
        "kraus_prob_tol": args.kraus_prob_tol,
        "max_kraus_terms": args.max_kraus_terms,
        "initial_thermal_prob_tol": args.initial_thermal_prob_tol,
        "max_initial_thermal_branches": args.max_initial_thermal_branches,
        "d1": protocol["d1"],
        "d2": protocol["d2"],
        "j2": protocol["j2"],
        "Nt": protocol["Nt"],
        "NR": protocol["NR"],
        "source_d1": protocol["source_d1"],
        "source_d2": protocol["source_d2"],
        "source_j2": protocol["source_j2"],
        "source_score": selected_source_score,
        "uses_source_protocol_settings": protocol["uses_source_protocol_settings"],
        "source_metadata_differs_from_default_protocol_settings": protocol[
            "source_metadata_differs_from_default_protocol_settings"
        ],
        "source_parameter_file": relative_to_repo(parameter_path),
        "source_info": source_info,
        "output_file": relative_to_repo(ci_path),
        "function": "transduction_protocol_CoherentInfo_GKP2_thermal_noise",
        "model": "canonical_unitary_completion_GKP_preparation_plus_initial_P_thermal_branches_plus_output_pure_loss",
        "ns_input": ns_value,
        "np_input": np_value,
        "state_RS_shape": list(state_RS.shape),
        "state_P_return_shape": list(state_P_return.shape),
        "start_time": start_time.isoformat(),
        "finish_time": finish_time.isoformat(),
        "elapsed_seconds": elapsed,
    }
    if args.validate_zero_noise:
        config.update(
            {
                "recomputed_zero_noise_ci": ci_value,
                "zero_noise_abs_error": zero_noise_abs_error,
                "zero_noise_validation_passed": zero_noise_validation_passed,
                "zero_noise_tolerance": args.zero_noise_tolerance,
            }
        )
    for key in (
        "prep_state_error",
        "trace_rho_P",
        "trace_rho_RP",
        "initial_thermal_branch_count",
        "initial_thermal_prob_sum",
        "pure_loss_terms_s",
        "pure_loss_terms_p",
        "branch_count",
        "skipped_branch_count",
    ):
        if key in debug:
            config[key] = debug_scalar(debug[key])
    (eta_out_dir / "noise_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"{eta_folder(eta)} [{args.setup_name}]: finished CI={ci_value:.12g} "
        f"ns={ns_value:.12g} np={np_value:.12g} "
        f"start={start_time.isoformat()} finish={finish_time.isoformat()} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    if args.validate_zero_noise:
        status = "PASS" if zero_noise_validation_passed else "FAIL"
        print(
            f"{eta_folder(eta)} zero-noise validation {status}: "
            f"eta={float(eta):.2f} source_score={selected_source_score:.12g} "
            f"recomputed_score={ci_value:.12g} abs_error={zero_noise_abs_error:.3g} "
            f"(d1,d2,j2,NR)=({protocol['d1']},{protocol['d2']},"
            f"{protocol['j2']},{protocol['NR']})",
            flush=True,
        )
    return {
        "eta": float(eta),
        "setup_name": args.setup_name,
        "ci_noise": ci_value,
        "ns_input": ns_value,
        "np_input": np_value,
        "d1": protocol["d1"],
        "d2": protocol["d2"],
        "j2": protocol["j2"],
        "Nt": protocol["Nt"],
        "NR": protocol["NR"],
        "source_score": selected_source_score,
        "zero_noise_abs_error": zero_noise_abs_error,
        "zero_noise_validation_passed": zero_noise_validation_passed,
        "elapsed_seconds": elapsed,
        "output_file": relative_to_repo(ci_path),
        "output_folder": relative_to_repo(args.output_dir),
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
        setup = SETUP_BY_NAME[args.setup or "noisy"]

    args.setup_name = setup["name"]
    for attr in ("initial_p_nbar", "kappa_o", "kappa_m", "n_o", "n_m"):
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
        "ci_noise",
        "ns_input",
        "np_input",
        "d1",
        "d2",
        "j2",
        "Nt",
        "NR",
        "source_score",
        "zero_noise_abs_error",
        "zero_noise_validation_passed",
        "source_parameter_file",
        "elapsed_seconds",
        "output_file",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate noisy CI for corrected GKP parameters from run 94 as job 93."
    )
    parser.add_argument("--list-setups", action="store_true")
    parser.add_argument("--setup", choices=sorted(SETUP_BY_NAME))
    parser.add_argument("--setup-index", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--eta-index", type=int)
    parser.add_argument("--all-etas", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-p-nbar", type=float)
    parser.add_argument("--kappa-o", type=float)
    parser.add_argument("--kappa-m", type=float)
    parser.add_argument("--n-o", type=float)
    parser.add_argument("--n-m", type=float)
    parser.add_argument("--kraus-prob-tol", type=float, default=1e-12)
    parser.add_argument("--initial-thermal-prob-tol", type=float, default=1e-14)
    parser.add_argument("--max-kraus-terms", type=int)
    parser.add_argument("--max-initial-thermal-branches", type=int)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--validate-zero-noise", action="store_true")
    parser.add_argument("--zero-noise-tolerance", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--return-debug", action="store_true")
    args = parser.parse_args()

    if args.list_setups:
        for idx, setup in enumerate(SETUP_PRESETS):
            print(
                f"{idx}: {setup['name']} "
                f"initial_p_nbar={setup['initial_p_nbar']} "
                f"kappa_o={setup['kappa_o']} kappa_m={setup['kappa_m']} "
                f"n_o={setup['n_o']} n_m={setup['n_m']} "
                f"output_subdir={setup['output_subdir']}",
                flush=True,
            )
        raise SystemExit(0)

    if not local_qtorch_dir.is_dir():
        raise SystemExit(f"Missing local QTorch copy: {local_qtorch_dir}")

    if not args.output_root.is_absolute():
        args.output_root = (job_dir / args.output_root).resolve()

    select_setup(args)
    if args.validate_zero_noise:
        validate_zero_noise_args(args)

    if not args.output_dir.is_absolute():
        args.output_dir = (job_dir / args.output_dir).resolve()

    return args


def main():
    # Parse args before setting PyTorch threads so CLI overrides SLURM defaults.
    args = parse_args()
    if args.num_threads is not None:
        num_cpu = args.num_threads
    else:
        num_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    torch.set_num_threads(num_cpu)
    torch.set_num_interop_threads(1)

    etas = select_etas(args)
    print(f"Job directory: {job_dir}", flush=True)
    print(f"Using local QTorch: {local_qtorch_dir}", flush=True)
    print(f"Setup: {args.setup_name}", flush=True)
    print(f"PyTorch threads: {torch.get_num_threads()}", flush=True)
    print(
        f"Noise parameters: initial_p_nbar={args.initial_p_nbar} "
        f"kappa_o={args.kappa_o} kappa_m={args.kappa_m} "
        f"n_o={args.n_o} n_m={args.n_m}",
        flush=True,
    )
    print(f"Output directory: {relative_to_repo(args.output_dir)}", flush=True)
    print(f"Etas: {' '.join(eta_folder(eta) for eta in etas)}", flush=True)

    if args.dry_run:
        for eta in etas:
            parameters, parameter_path, source_info, protocol = load_parameters(eta, args.device)
            selection = source_info.get("selection_summary", {})
            print(
                f"{eta_folder(eta)}: parameter_file={relative_to_repo(parameter_path)} "
                f"shape={tuple(parameters.shape)} d1={protocol['d1']} "
                f"d2={protocol['d2']} j2={protocol['j2']} "
                f"Nt={protocol['Nt']} NR={protocol['NR']} "
                f"uses_source_protocol_settings={protocol['uses_source_protocol_settings']} "
                f"source_score={selection.get('score', '')}",
                flush=True,
            )
        print("Dry run only; no CI values computed.", flush=True)
        return

    rows = [calculate_eta(args, eta) for eta in etas]
    if args.validate_zero_noise:
        failed = [
            row
            for row in rows
            if row.get("zero_noise_validation_passed") is not True
        ]
        if failed:
            raise SystemExit(
                "Zero-noise validation failed for etas: "
                + ", ".join(f"{row['eta']:.2f}" for row in failed)
            )
    if len(rows) > 1:
        summary_path = write_summary(args.output_dir, rows)
        print(f"Wrote summary: {relative_to_repo(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
