import argparse
import csv
import json
from pathlib import Path

import numpy as np


job_dir = Path(__file__).resolve().parent
RUN_ID = 93
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)
SETUP_OUTPUTS = [
    ("noisy", "noisy_nPth=0p1_kS=0p99_kP=0p99"),
]


def eta_folder(eta):
    return f"eta={float(eta):.2f}"


def read_eta_result(setup_dir, eta_dir, setup_name, eta):
    eta_dir = setup_dir / eta_folder(eta)
    ci_path = eta_dir / "best_feasible_ci.txt"
    config_path = eta_dir / "noise_config.json"
    if not ci_path.exists():
        return {
            "setup_folder": setup_name,
            "setup_name": "",
            "eta": float(eta),
            "status": "missing",
            "ci_noise": "",
            "output_file": str(ci_path),
        }

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    return {
        "setup_folder": setup_name,
        "setup_name": config.get("setup_name", ""),
        "eta": float(eta),
        "status": "ok",
        "ci_noise": float(ci_path.read_text().strip()),
        "ns_input": config.get("ns_input", ""),
        "np_input": config.get("np_input", ""),
        "initial_p_thermal_nbar": config.get("initial_p_thermal_nbar", ""),
        "kappa_o": config.get("kappa_o", ""),
        "kappa_m": config.get("kappa_m", ""),
        "n_o": config.get("n_o", ""),
        "n_m": config.get("n_m", ""),
        "d1": config.get("d1", ""),
        "d2": config.get("d2", ""),
        "j2": config.get("j2", ""),
        "Nt": config.get("Nt", ""),
        "NR": config.get("NR", ""),
        "source_score": config.get("source_score", ""),
        "elapsed_seconds": config.get("elapsed_seconds", ""),
        "source_parameter_file": config.get("source_parameter_file", ""),
        "output_file": config.get("output_file", str(ci_path)),
    }


def write_combined_summary(output_root, rows):
    path = output_root / "noisy_gkp_93_summary.tsv"
    fieldnames = [
        "setup_name",
        "eta",
        "ci_noise",
        "ns_input",
        "np_input",
        "initial_p_thermal_nbar",
        "kappa_o",
        "kappa_m",
        "n_o",
        "n_m",
        "d1",
        "d2",
        "j2",
        "Nt",
        "NR",
        "source_score",
        "source_parameter_file",
        "elapsed_seconds",
        "output_file",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            if row.get("status") != "ok":
                continue
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_json_summary(output_root, rows):
    path = output_root / "noisy_gkp_93_summary.json"
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate noisy GKP CI outputs for job 93."
    )
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.data_root
    if not output_root.is_absolute():
        output_root = (job_dir / output_root).resolve()

    rows = []
    warnings = []
    for setup_name, setup_folder in SETUP_OUTPUTS:
        setup_dir = output_root / setup_folder
        for eta in DEFAULT_ETAS:
            eta_dir = setup_dir / eta_folder(eta)
            row = read_eta_result(setup_dir, eta_dir, setup_name, eta)
            if row.get("status") != "ok":
                warnings.append(f"Missing result: {setup_name}/{eta_folder(eta)}")
            rows.append(row)

    if args.strict and warnings:
        raise SystemExit("\n".join(warnings))
    for warning in warnings:
        print(f"Warning: {warning}")

    output_root.mkdir(parents=True, exist_ok=True)
    tsv_path = write_combined_summary(output_root, rows)
    json_path = write_json_summary(output_root, rows)
    missing = sum(1 for row in rows if row.get("status") != "ok")
    print(f"Wrote {tsv_path}")
    print(f"Wrote {json_path}")
    print(f"Rows: {len(rows)}")
    print(f"Missing results: {missing}")


if __name__ == "__main__":
    main()
