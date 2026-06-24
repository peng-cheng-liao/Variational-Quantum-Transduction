#!/usr/bin/env python3
"""Evaluate fixed-parameter VQT coherent information around eta0=0.30."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "Data_HPC"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

ETA0 = 0.30
DEFAULT_DELTAS = [round(x, 2) for x in np.arange(0.01, 0.101, 0.01)]
DEFAULT_OUTPUT_STEM = "vqt_eta_uncertainty_fixed_eta0_0p30"
VQT_RUN_ID = 84
VQT_DEPTH = 20
VQT_NT = 30
NS_CONSTRAINT = 2.0
NP_CONSTRAINT = 2.0

OUTPUT_FIELDS = [
    "eta0",
    "delta",
    "eta_eval_minus",
    "eta_eval_plus",
    "CI_minus",
    "CI_plus",
    "source_parameter_eta",
    "source_parameter_file_or_seed",
    "source_parameter_file",
    "source_seed",
    "nominal_stored_CI",
    "ns_minus",
    "np_minus",
    "ns_plus",
    "np_plus",
]


def eta_label(eta):
    return f"{float(eta):.2f}"


def eta_folder(eta):
    return f"eta={eta_label(eta)}"


def format_float(value):
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.17g}"


def read_float(path):
    return float(path.read_text().strip())


def relpath(path):
    return str(path.relative_to(REPO_ROOT))


def tensor_scalar(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def import_torch_protocol():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import torch
    from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA

    torch.set_num_threads(int(os.environ.get("JOB98_TORCH_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch, transduction_protocol_CoherentInfo_ECD_MM_EA


def load_selection_metadata(eta0):
    summary_path = DATA_ROOT / str(VQT_RUN_ID) / "selection_summary.tsv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("eta_folder") == eta_folder(eta0):
                return row
    return {}


def resolve_parameter_source(eta0):
    eta_dir = DATA_ROOT / str(VQT_RUN_ID) / eta_folder(eta0)
    param_path = eta_dir / "parameters_best_feasible.npy"
    ci_path = eta_dir / "best_feasible_ci.txt"
    if not param_path.exists():
        raise FileNotFoundError(
            f"Missing eta0={eta_label(eta0)} VQT parameter source: {param_path}"
        )
    if not ci_path.exists():
        raise FileNotFoundError(f"Missing eta0={eta_label(eta0)} stored CI file: {ci_path}")

    metadata = load_selection_metadata(eta0)
    seed = metadata.get("best_seed", "")
    return {
        "parameter_path": param_path,
        "stored_ci_path": ci_path,
        "stored_ci": read_float(ci_path),
        "selection_metadata": metadata,
        "seed": seed,
        "source_parameter_file_or_seed": f"{relpath(param_path)}; {seed}" if seed else relpath(param_path),
    }


def evaluate_vqt(torch, protocol, parameters, eta_eval):
    parameter_tensor = torch.as_tensor(parameters, dtype=torch.float64)
    with torch.no_grad():
        ci, ns_in, np_in, _state_rs, _state_pa = protocol(
            float(eta_eval), parameter_tensor, VQT_DEPTH, VQT_NT
        )
    return tensor_scalar(ci), tensor_scalar(ns_in), tensor_scalar(np_in)


def make_row(source, delta, minus_result, plus_result):
    eta_minus = round(ETA0 - delta, 2)
    eta_plus = round(ETA0 + delta, 2)
    assert 0.0 <= eta_minus <= 1.0, f"eta_minus out of range: {eta_minus}"
    assert 0.0 <= eta_plus <= 1.0, f"eta_plus out of range: {eta_plus}"

    ci_minus, ns_minus, np_minus = minus_result
    ci_plus, ns_plus, np_plus = plus_result
    for name, value in [
        ("CI_minus", ci_minus),
        ("CI_plus", ci_plus),
        ("ns_minus", ns_minus),
        ("np_minus", np_minus),
        ("ns_plus", ns_plus),
        ("np_plus", np_plus),
    ]:
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{name} is not finite for delta={delta:.2f}")

    energy_notes = []
    if ns_minus > NS_CONSTRAINT + 1e-5 or np_minus > NP_CONSTRAINT + 1e-5:
        energy_notes.append(f"minus energy ns={ns_minus:.8g} np={np_minus:.8g}")
    if ns_plus > NS_CONSTRAINT + 1e-5 or np_plus > NP_CONSTRAINT + 1e-5:
        energy_notes.append(f"plus energy ns={ns_plus:.8g} np={np_plus:.8g}")
    if energy_notes:
        print("Energy check:", "; ".join(energy_notes))

    return {
        "eta0": eta_label(ETA0),
        "delta": eta_label(delta),
        "eta_eval_minus": eta_label(eta_minus),
        "eta_eval_plus": eta_label(eta_plus),
        "CI_minus": format_float(ci_minus),
        "CI_plus": format_float(ci_plus),
        "source_parameter_eta": eta_label(ETA0),
        "source_parameter_file_or_seed": source["source_parameter_file_or_seed"],
        "source_parameter_file": relpath(source["parameter_path"]),
        "source_seed": source["seed"],
        "nominal_stored_CI": format_float(source["stored_ci"]),
        "ns_minus": format_float(ns_minus),
        "np_minus": format_float(np_minus),
        "ns_plus": format_float(ns_plus),
        "np_plus": format_float(np_plus),
    }


def write_outputs(rows, output_dir, source, deltas, output_stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_stem}.csv"
    json_path = output_dir / f"{output_stem}.json"
    npz_path = output_dir / f"{output_stem}.npz"
    suffix = "" if output_stem == DEFAULT_OUTPUT_STEM else f"_{output_stem}"
    config_path = output_dir / f"config{suffix}.json"
    readme_path = output_dir / f"README{suffix}.md"

    for path in [csv_path, json_path, npz_path, config_path, readme_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing Job 98 output: {path}")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    np.savez(
        npz_path,
        eta0=np.array([ETA0], dtype=float),
        delta=np.array([float(row["delta"]) for row in rows], dtype=float),
        eta_eval_minus=np.array([float(row["eta_eval_minus"]) for row in rows], dtype=float),
        eta_eval_plus=np.array([float(row["eta_eval_plus"]) for row in rows], dtype=float),
        CI_minus=np.array([float(row["CI_minus"]) for row in rows], dtype=float),
        CI_plus=np.array([float(row["CI_plus"]) for row in rows], dtype=float),
    )

    config = {
        "job": 98,
        "eta0": ETA0,
        "deltas": deltas,
        "output_stem": output_stem,
        "scheme": "VQT",
        "source_parameter_eta": ETA0,
        "source_parameter_file": relpath(source["parameter_path"]),
        "source_seed": source["seed"],
        "source_selection_metadata": source["selection_metadata"],
        "nominal_stored_CI": source["stored_ci"],
        "vqt_run_id": VQT_RUN_ID,
        "vqt_depth": VQT_DEPTH,
        "vqt_nt": VQT_NT,
        "n_s": NS_CONSTRAINT,
        "n_p": NP_CONSTRAINT,
        "notes": "Fixed-parameter VQT evaluation; no retraining or re-optimization at shifted eta.",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    readme_path.write_text(
        "# Job 98 Data\n\n"
        "Fixed-parameter VQT eta-uncertainty evaluation at `eta0 = 0.30`.\n\n"
        "- The parameter set is selected from `Data_HPC/84/eta=0.30/parameters_best_feasible.npy`.\n"
        "- The selected run-84 seed is recorded in `config.json` and in the CSV source fields.\n"
        "- The parameter set is not reoptimized for shifted eta values.\n"
        f"- Deltas are `{', '.join(f'{delta:.2f}' for delta in deltas)}`.\n"
        "- `minus` means `CI(eta0 - delta)` evaluated with the fixed eta0 parameter set.\n"
        "- `plus` means `CI(eta0 + delta)` evaluated with the fixed eta0 parameter set.\n",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {config_path}")
    print(f"Wrote {readme_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-parameter VQT coherent information around eta0=0.30."
    )
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "98")
    parser.add_argument("--deltas", nargs="+", type=float, default=DEFAULT_DELTAS)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def main():
    args = parse_args()
    deltas = [round(float(delta), 2) for delta in args.deltas]
    if not deltas:
        raise ValueError("At least one delta must be supplied")
    if deltas != sorted(set(deltas)):
        raise ValueError(f"Deltas must be unique and sorted; got {deltas}")

    source = resolve_parameter_source(ETA0)
    print(f"Fixed VQT parameter source: {relpath(source['parameter_path'])}")
    if source["seed"]:
        print(f"Fixed VQT source seed: {source['seed']}")
    print(f"Stored nominal CI at eta0={eta_label(ETA0)}: {source['stored_ci']:.12g}")

    torch, protocol = import_torch_protocol()
    parameters = np.load(source["parameter_path"])

    rows = []
    for delta in deltas:
        eta_minus = round(ETA0 - delta, 2)
        eta_plus = round(ETA0 + delta, 2)
        minus_result = evaluate_vqt(torch, protocol, parameters, eta_minus)
        plus_result = evaluate_vqt(torch, protocol, parameters, eta_plus)
        row = make_row(source, delta, minus_result, plus_result)
        rows.append(row)
        print(
            "[Job98] "
            f"delta={delta:.2f} eta_minus={eta_minus:.2f} eta_plus={eta_plus:.2f} "
            f"CI_minus={float(row['CI_minus']):.12g} CI_plus={float(row['CI_plus']):.12g}"
        )

    assert len(rows) == len(deltas), f"expected {len(deltas)} output rows, got {len(rows)}"
    write_outputs(rows, args.output, source, deltas, args.output_stem)


if __name__ == "__main__":
    main()
