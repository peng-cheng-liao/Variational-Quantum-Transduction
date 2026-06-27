import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


job_dir = Path(__file__).resolve().parent
repo_dir = job_dir.parents[1]

RUN_ID = 93
SOURCE_RUN_ID = 94
MATCHED_VQT_RUN_ID = 99
DEFAULT_DATA_ROOT = Path("Data") / "gkp_eta_scans_nPth_grid_tauSP_0p99"
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)
GKP_NPTH_SETTINGS = [0.0, 0.001, 0.01, 0.03, 0.05, 0.07, 0.1]


def value_tag(value):
    text = f"{float(value):.6g}"
    return text.replace(".", "p").replace("-", "m")


def eta_folder(eta):
    return f"eta={float(eta):.2f}"


def setting_id_for_npth(npth):
    return f"nPth_{value_tag(npth)}_kS_0p99_kP_0p99"


def make_vqt_eta_case(case_id, npth, vqt_kappa_a, description):
    return {
        "case_id": case_id,
        "vqt_case_id": case_id,
        "shared_setting_id": setting_id_for_npth(npth),
        "scan_type": "eta",
        "vqt_initial_p_thermal_nbar_metadata_only": float(npth),
        "vqt_initial_a_thermal_nbar_metadata_only": float(npth),
        "vqt_kappa_a_metadata_only": float(vqt_kappa_a),
        "vqt_n_a_metadata_only": 0.0,
        "case_description": description,
    }


VQT_ETA_CASES = [
    make_vqt_eta_case(
        "case1_eta_scan_nthP_0_nthA_0_tauA_0p90",
        0.0,
        0.90,
        "eta scan, no initial thermal photons, VQT tau_A=0.90 metadata only",
    ),
    make_vqt_eta_case(
        "nthP_0_nthA_0_tauAll_0p99",
        0.0,
        0.99,
        "eta scan, no initial thermal photons, VQT tau_S=tau_P=tau_A=0.99 metadata only",
    ),
    make_vqt_eta_case(
        "case2_eta_scan_nthP_0p1_nthA_0p1_tauA_0p90",
        0.1,
        0.90,
        "eta scan, n_P^th=n_A^th=0.1, VQT tau_A=0.90 metadata only",
    ),
    make_vqt_eta_case(
        "nthP_0p001_nthA_0p001_tauAll_0p99",
        0.001,
        0.99,
        "eta scan, n_P^th=n_A^th=0.001, VQT tau_S=tau_P=tau_A=0.99 metadata only",
    ),
    make_vqt_eta_case(
        "nthP_0p01_nthA_0p01_tauAll_0p99",
        0.01,
        0.99,
        "eta scan, n_P^th=n_A^th=0.01, VQT tau_S=tau_P=tau_A=0.99 metadata only",
    ),
    make_vqt_eta_case(
        "nthP_0p03_nthA_0p03_tauAll_0p99",
        0.03,
        0.99,
        "eta scan, n_P^th=n_A^th=0.03, VQT tau_S=tau_P=tau_A=0.99 metadata only",
    ),
    make_vqt_eta_case(
        "nthP_0p05_nthA_0p05_tauAll_0p99",
        0.05,
        0.99,
        "eta scan, n_P^th=n_A^th=0.05, VQT tau_S=tau_P=tau_A=0.99 metadata only",
    ),
    make_vqt_eta_case(
        "nthP_0p07_nthA_0p07_tauAll_0p99",
        0.07,
        0.99,
        "eta scan, n_P^th=n_A^th=0.07, VQT tau_S=tau_P=tau_A=0.99 metadata only",
    ),
    make_vqt_eta_case(
        "nthP_0p1_nthA_0p1_tauAll_0p99",
        0.1,
        0.99,
        "eta scan, n_P^th=n_A^th=0.1, VQT tau_S=tau_P=tau_A=0.99 metadata only",
    ),
]


SUMMARY_FIELDS = [
    "case_id",
    "vqt_case_id",
    "shared_setting_id",
    "scan_type",
    "eta",
    "scan_value",
    "ci_noise",
    "initial_p_thermal_nbar",
    "kappa_o",
    "kappa_m",
    "n_o",
    "n_m",
    "gkp_has_no_auxiliary_mode_A",
    "gkp_independent_of_tau_A",
    "vqt_initial_a_thermal_nbar_metadata_only",
    "vqt_kappa_a_metadata_only",
    "source_parameter_file",
    "shared_result_folder",
    "output_folder",
    "status",
    "materialized",
]


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(repo_dir))
    except ValueError:
        return str(path)


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_materialized_config(shared_config, case, eta, shared_eta_dir, case_eta_dir):
    config = dict(shared_config)
    config.update(
        {
            "run_id": RUN_ID,
            "source_run_id": SOURCE_RUN_ID,
            "matched_vqt_run_id": MATCHED_VQT_RUN_ID,
            "case_id": case["case_id"],
            "vqt_case_id": case["vqt_case_id"],
            "case_description": case["case_description"],
            "shared_setting_id": case["shared_setting_id"],
            "scan_type": "eta",
            "eta": float(eta),
            "scan_value": float(eta),
            "gkp_has_no_auxiliary_mode_A": True,
            "gkp_independent_of_tau_A": True,
            "vqt_initial_p_thermal_nbar_metadata_only": case[
                "vqt_initial_p_thermal_nbar_metadata_only"
            ],
            "vqt_initial_a_thermal_nbar_metadata_only": case[
                "vqt_initial_a_thermal_nbar_metadata_only"
            ],
            "vqt_kappa_a_metadata_only": case["vqt_kappa_a_metadata_only"],
            "vqt_n_a_metadata_only": case["vqt_n_a_metadata_only"],
            "shared_result_folder": relative_to_repo(shared_eta_dir),
            "materialized_case_output": True,
            "output_file": relative_to_repo(case_eta_dir / "best_feasible_ci.txt"),
        }
    )
    return config


def summarize_case_row(case, eta, data_root):
    shared_eta_dir = data_root / "shared" / case["shared_setting_id"] / eta_folder(eta)
    case_eta_dir = data_root / "cases" / case["case_id"] / eta_folder(eta)
    shared_ci = shared_eta_dir / "best_feasible_ci.txt"
    shared_config = read_json(shared_eta_dir / "noise_config.json")
    row = {
        "case_id": case["case_id"],
        "vqt_case_id": case["vqt_case_id"],
        "shared_setting_id": case["shared_setting_id"],
        "scan_type": "eta",
        "eta": float(eta),
        "scan_value": float(eta),
        "ci_noise": "",
        "initial_p_thermal_nbar": shared_config.get("initial_p_thermal_nbar", ""),
        "kappa_o": shared_config.get("kappa_o", ""),
        "kappa_m": shared_config.get("kappa_m", ""),
        "n_o": shared_config.get("n_o", ""),
        "n_m": shared_config.get("n_m", ""),
        "gkp_has_no_auxiliary_mode_A": True,
        "gkp_independent_of_tau_A": True,
        "vqt_initial_a_thermal_nbar_metadata_only": case[
            "vqt_initial_a_thermal_nbar_metadata_only"
        ],
        "vqt_kappa_a_metadata_only": case["vqt_kappa_a_metadata_only"],
        "source_parameter_file": shared_config.get("source_parameter_file", ""),
        "shared_result_folder": relative_to_repo(shared_eta_dir),
        "output_folder": relative_to_repo(case_eta_dir),
        "status": "missing",
        "materialized": "",
    }
    if shared_ci.exists():
        row["ci_noise"] = float(shared_ci.read_text().strip())
        row["status"] = "ok"
    return row, shared_eta_dir, case_eta_dir, shared_config


def materialize_case(row, shared_eta_dir, case_eta_dir, shared_config, case, eta, overwrite):
    if row["status"] != "ok":
        return "missing_shared"

    shared_ci = shared_eta_dir / "best_feasible_ci.txt"
    shared_source = shared_eta_dir / "source_parameter_file.txt"
    targets = [case_eta_dir / "best_feasible_ci.txt", case_eta_dir / "noise_config.json"]
    if shared_source.exists():
        targets.append(case_eta_dir / "source_parameter_file.txt")
    if any(target.exists() for target in targets) and not overwrite:
        return "skipped_existing"

    case_eta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_ci, case_eta_dir / "best_feasible_ci.txt")
    if shared_source.exists():
        shutil.copy2(shared_source, case_eta_dir / "source_parameter_file.txt")
    config = build_materialized_config(shared_config, case, eta, shared_eta_dir, case_eta_dir)
    (case_eta_dir / "noise_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return "written"


def write_summaries(data_root, rows):
    data_root.mkdir(parents=True, exist_ok=True)
    tsv_path = data_root / "noise_ci_summary_93.tsv"
    json_path = data_root / "noise_ci_summary_93.json"
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return tsv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and materialize Job 93 noisy GKP eta-scan results."
    )
    parser.add_argument("--data-root", "--output-root", dest="data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--materialize-cases", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = (job_dir / data_root).resolve()

    rows = []
    warnings = []
    for case in VQT_ETA_CASES:
        for eta in DEFAULT_ETAS:
            row, shared_eta_dir, case_eta_dir, shared_config = summarize_case_row(
                case, eta, data_root
            )
            if row["status"] != "ok":
                warnings.append(
                    f"Missing shared result: {relative_to_repo(shared_eta_dir / 'best_feasible_ci.txt')}"
                )
            if args.materialize_cases:
                row["materialized"] = materialize_case(
                    row,
                    shared_eta_dir,
                    case_eta_dir,
                    shared_config,
                    case,
                    eta,
                    args.overwrite,
                )
            rows.append(row)

    if args.strict and warnings:
        raise SystemExit("\n".join(warnings))
    for warning in warnings:
        print(f"Warning: {warning}")

    tsv_path, json_path = write_summaries(data_root, rows)
    missing = sum(1 for row in rows if row.get("status") != "ok")
    written = sum(1 for row in rows if row.get("materialized") == "written")
    skipped = sum(1 for row in rows if row.get("materialized") == "skipped_existing")
    print(f"Wrote {relative_to_repo(tsv_path)}")
    print(f"Wrote {relative_to_repo(json_path)}")
    print(f"Rows: {len(rows)}")
    print(f"Missing shared results: {missing}")
    if args.materialize_cases:
        print(f"Materialized written: {written}")
        print(f"Materialized skipped existing: {skipped}")


if __name__ == "__main__":
    main()
