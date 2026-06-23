import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


job_dir = Path(__file__).resolve().parent
repo_dir = job_dir.parents[1]
DEFAULT_INPUT_ROOT = repo_dir / "Data_HPC" / "95"
DEFAULT_SUMMARY_PATH = DEFAULT_INPUT_ROOT / "noise_ci_summary.tsv"
DEFAULT_BEST_PATH = DEFAULT_INPUT_ROOT / "best_by_eta_kappaA_95.json"


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(repo_dir))
    except ValueError:
        return str(path)


def result_dirs(input_root):
    return sorted(path.parent for path in input_root.glob("*/eta=*/best_feasible_ci.txt"))


def row_from_result_dir(eta_out_dir):
    ci_path = eta_out_dir / "best_feasible_ci.txt"
    config_path = eta_out_dir / "noise_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing noise config: {config_path}")

    config = json.loads(config_path.read_text())
    ci_noise = float(ci_path.read_text().strip())
    return {
        "eta": config.get("eta", ""),
        "setup_name": config.get("setup_name", ""),
        "kappa_a": config.get("kappa_a", ""),
        "ci_noise": ci_noise,
        "ns_input": config.get("ns_input", ""),
        "np_input": config.get("np_input", ""),
        "initial_p_thermal_nbar": config.get("initial_p_thermal_nbar", ""),
        "kappa_o": config.get("kappa_o", ""),
        "kappa_m": config.get("kappa_m", ""),
        "n_o": config.get("n_o", ""),
        "n_m": config.get("n_m", ""),
        "n_a": config.get("n_a", ""),
        "source_parameter_file": config.get("source_parameter_file", ""),
        "output_folder": relative_to_repo(eta_out_dir),
    }


def group_key(row):
    return (row["eta"], row["kappa_a"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Job 95 noisy VQT auxiliary-loss outputs."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--best-json", type=Path, default=DEFAULT_BEST_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_root.is_absolute():
        args.input_root = (repo_dir / args.input_root).resolve()
    if not args.summary.is_absolute():
        args.summary = (repo_dir / args.summary).resolve()
    if not args.best_json.is_absolute():
        args.best_json = (repo_dir / args.best_json).resolve()

    rows = []
    errors = []
    for eta_out_dir in result_dirs(args.input_root):
        try:
            rows.append(row_from_result_dir(eta_out_dir))
        except Exception as exc:
            errors.append({"path": relative_to_repo(eta_out_dir), "error": str(exc)})

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
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (float(item["eta"]), float(item["kappa_a"]))):
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    best_rows = []
    for _, group_rows in grouped.items():
        best_rows.append(max(group_rows, key=lambda item: float(item["ci_noise"])))

    best_payload = {
        "summary_tsv": relative_to_repo(args.summary),
        "input_root": relative_to_repo(args.input_root),
        "result_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "best_by_eta_kappa_a": [
            {
                "eta": row["eta"],
                "setup_name": row["setup_name"],
                "kappa_a": row["kappa_a"],
                "ci_noise": row["ci_noise"],
                "output_folder": row["output_folder"],
            }
            for row in sorted(best_rows, key=lambda item: (float(item["eta"]), float(item["kappa_a"])))
        ],
    }
    args.best_json.write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} rows to {relative_to_repo(args.summary)}", flush=True)
    print(f"Wrote best summary to {relative_to_repo(args.best_json)}", flush=True)
    if errors:
        print(f"Skipped {len(errors)} corrupt outputs; see best summary JSON.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
