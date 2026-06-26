import argparse
import csv
import json
import sys
from pathlib import Path


job_dir = Path(__file__).resolve().parent
repo_dir = job_dir.parents[1]
DEFAULT_INPUT_ROOT = job_dir / "Data"
DEFAULT_SUMMARY_PATH = DEFAULT_INPUT_ROOT / "noise_ci_summary.tsv"
DEFAULT_JSON_PATH = DEFAULT_INPUT_ROOT / "noise_ci_summary_99.json"


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(repo_dir))
    except ValueError:
        return str(path)


def result_dirs(input_root):
    return sorted(path.parent for path in input_root.glob("*/eta=*/best_feasible_ci.txt"))


def row_from_result_dir(point_out_dir):
    ci_path = point_out_dir / "best_feasible_ci.txt"
    config_path = point_out_dir / "noise_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing noise config: {config_path}")

    config = json.loads(config_path.read_text())
    ci_noise = float(ci_path.read_text().strip())
    return {
        "case_id": config.get("case_id", ""),
        "scan_type": config.get("scan_type", ""),
        "scan_value": config.get("scan_value", ""),
        "eta": config.get("eta", ""),
        "kappa_a": config.get("kappa_a", ""),
        "ci_noise": ci_noise,
        "ns_input": config.get("ns_input", ""),
        "np_input": config.get("np_input", ""),
        "initial_p_thermal_nbar": config.get("initial_p_thermal_nbar", ""),
        "initial_a_thermal_nbar": config.get("initial_a_thermal_nbar", ""),
        "kappa_o": config.get("kappa_o", ""),
        "kappa_m": config.get("kappa_m", ""),
        "n_o": config.get("n_o", ""),
        "n_m": config.get("n_m", ""),
        "n_a": config.get("n_a", ""),
        "source_parameter_file": config.get("source_parameter_file", ""),
        "output_folder": relative_to_repo(point_out_dir),
    }


def row_sort_key(row):
    return (row["case_id"], float(row["eta"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Job 99 noisy VQT initial-P/A thermal outputs."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_root.is_absolute():
        args.input_root = (job_dir / args.input_root).resolve()
    if not args.summary.is_absolute():
        args.summary = (job_dir / args.summary).resolve()
    if not args.json.is_absolute():
        args.json = (job_dir / args.json).resolve()

    rows = []
    errors = []
    for point_out_dir in result_dirs(args.input_root):
        try:
            rows.append(row_from_result_dir(point_out_dir))
        except Exception as exc:
            errors.append({"path": relative_to_repo(point_out_dir), "error": str(exc)})

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
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=row_sort_key)
    with args.summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    payload = {
        "summary_tsv": relative_to_repo(args.summary),
        "input_root": relative_to_repo(args.input_root),
        "result_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "rows": sorted_rows,
    }
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} rows to {relative_to_repo(args.summary)}", flush=True)
    print(f"Wrote JSON summary to {relative_to_repo(args.json)}", flush=True)
    if errors:
        print(f"Skipped {len(errors)} corrupt outputs; see summary JSON.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
