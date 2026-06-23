import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


job_dir = Path(__file__).resolve().parent
repo_dir = job_dir.parents[1]
DEFAULT_INPUT_ROOT = repo_dir / "Data_HPC" / "95"
DEFAULT_SUMMARY_PATH = DEFAULT_INPUT_ROOT / "summary_95.csv"
DEFAULT_BEST_PATH = DEFAULT_INPUT_ROOT / "best_by_eta_kappaA_95.json"


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(repo_dir))
    except ValueError:
        return str(path)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return value


def result_files(input_root):
    return sorted(input_root.glob("*/eta=*/result_kappaA_*.pt"))


def row_from_payload(path, payload):
    return {
        "eta": payload.get("eta", ""),
        "eta_index": payload.get("eta_index", ""),
        "kappa_a": payload.get("kappa_a", ""),
        "ci": scalar(payload.get("CI", "")),
        "coherent_information": scalar(payload.get("coherent_information", payload.get("CI", ""))),
        "ns_input": scalar(payload.get("ns_input", "")),
        "np_input": scalar(payload.get("np_input", "")),
        "initial_p_thermal_nbar": payload.get("initial_p_thermal_nbar", ""),
        "kappa_o": payload.get("kappa_o", ""),
        "kappa_m": payload.get("kappa_m", ""),
        "n_a": payload.get("n_a", ""),
        "n_o": payload.get("n_o", ""),
        "n_m": payload.get("n_m", ""),
        "depth": payload.get("depth", ""),
        "Nt": payload.get("Nt", ""),
        "parameter_source_path": payload.get(
            "parameter_source_path",
            payload.get("source_parameter_file", ""),
        ),
        "seed": payload.get("seed", ""),
        "parameter_index": payload.get("parameter_index", ""),
        "runtime_seconds": payload.get("runtime_seconds", ""),
        "timestamp_utc": payload.get("timestamp_utc", ""),
        "function": payload.get("function", ""),
        "output_file": relative_to_repo(path),
    }


def group_key(row):
    return (
        row["eta"],
        row["kappa_a"],
        row["initial_p_thermal_nbar"],
        row["kappa_o"],
        row["kappa_m"],
        row["n_a"],
        row["n_o"],
        row["n_m"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Job 95 auxiliary-loss sweep outputs in Data_HPC/95."
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
    for path in result_files(args.input_root):
        try:
            rows.append(row_from_payload(path, safe_torch_load(path)))
        except Exception as exc:
            errors.append({"path": relative_to_repo(path), "error": str(exc)})

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
        "runtime_seconds",
        "timestamp_utc",
        "function",
        "output_file",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (float(item["eta"]), float(item["kappa_a"]))):
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    best_rows = []
    for _, group_rows in grouped.items():
        best_rows.append(max(group_rows, key=lambda item: float(item["ci"])))

    best_payload = {
        "summary_csv": relative_to_repo(args.summary),
        "input_root": relative_to_repo(args.input_root),
        "result_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "best_by_eta_kappa_a": [
            {
                "eta": row["eta"],
                "eta_index": row["eta_index"],
                "kappa_a": row["kappa_a"],
                "initial_p_thermal_nbar": row["initial_p_thermal_nbar"],
                "kappa_o": row["kappa_o"],
                "kappa_m": row["kappa_m"],
                "n_a": row["n_a"],
                "n_o": row["n_o"],
                "n_m": row["n_m"],
                "ci": row["ci"],
                "coherent_information": row["coherent_information"],
                "output_file": row["output_file"],
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
