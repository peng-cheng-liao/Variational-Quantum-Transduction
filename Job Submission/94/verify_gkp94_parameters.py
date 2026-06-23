#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_DATA_ROOTS = (
    Path("Data_HPC/94/Data-Download-partial-10000"),
    Path("Data-Download-partial-10000"),
    Path("Data-Download"),
)
OUTPUT_FIELDS = [
    "eta",
    "d1",
    "d2",
    "j2",
    "randomization",
    "best_index",
    "summary_score",
    "saved_best_feasible_ci",
    "history_ci_at_best_index",
    "recomputed_ci",
    "ci_error_vs_saved",
    "ci_error_vs_summary",
    "ci_error_vs_history",
    "saved_ns_at_best_index",
    "recomputed_ns",
    "ns_error",
    "saved_np_at_best_index",
    "recomputed_np",
    "np_error",
    "parameters_dtype",
    "parameters_shape",
    "status",
    "message",
]
ENERGY_TOL = 1e-5


sys.path.insert(0, str(BASE_DIR))
from QTorch.Transduction import transduction_protocol_CoherentInfo_GKP2  # noqa: E402


def eta_label(value):
    return f"{float(value):.2f}"


def read_text(path):
    return path.read_text().strip()


def read_float(path):
    return float(read_text(path))


def read_int(path):
    return int(read_text(path))


def blank_row():
    return {field: "" for field in OUTPUT_FIELDS}


def finite_abs_error(a, b):
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b)


def scalar_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def resolve_default_data_root():
    search_roots = (Path.cwd(), BASE_DIR, REPO_ROOT)
    checked = []
    for relative_path in DEFAULT_DATA_ROOTS:
        for root in search_roots:
            candidate = (root / relative_path).resolve()
            checked.append(str(candidate))
            if candidate.exists():
                return candidate
    paths = "\n  ".join(checked)
    raise FileNotFoundError(f"No default data root found. Checked:\n  {paths}")


def resolve_data_root(data_root):
    if data_root is not None:
        return data_root.expanduser().resolve()
    return resolve_default_data_root()


def resolve_output_path(output, data_root, filename):
    if output is None:
        return data_root / filename
    return output.expanduser().resolve()


def load_selection_rows(summary_path, requested_etas):
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing selection summary: {summary_path}")

    rows = []
    warnings = []
    requested = {eta_label(eta) for eta in requested_etas} if requested_etas else None

    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eta = row.get("eta", "")
            status = row.get("status", "")
            if status == "no_valid_candidate":
                warnings.append(f"selection_summary.tsv contains no_valid_candidate row for eta={eta}")
            if requested is not None and eta_label(eta) not in requested:
                continue
            if not status.startswith("selected"):
                continue
            rows.append(row)

    return rows, warnings


def metadata_value(summary_row, source_info, name, cast):
    value = summary_row.get(name, "")
    if value == "" and source_info is not None:
        value = source_info.get(name, "")
    if value == "":
        raise ValueError(f"missing metadata field {name}")
    return cast(value)


def load_source_info(eta_dir, messages):
    path = eta_dir / "source_info.json"
    if not path.exists():
        messages.append("WARNING source_info.json is missing")
        return None
    with path.open() as f:
        return json.load(f)


def maybe_load_history(eta_dir, name, messages):
    path = eta_dir / f"{name}.npy"
    if not path.exists():
        messages.append(f"WARNING {name}.npy is missing")
        return None
    return np.load(path)


def history_value(history, best_index, name, messages):
    if history is None:
        return None
    if best_index < 0 or best_index >= len(history):
        messages.append(f"WARNING best_index={best_index} is out of range for {name}.npy")
        return None
    return float(np.asarray(history)[best_index])


def parameters_tensor(parameters, dtype_option):
    if dtype_option == "auto":
        return torch.as_tensor(parameters)
    if dtype_option == "float32":
        return torch.as_tensor(parameters, dtype=torch.float32)
    if dtype_option == "float64":
        return torch.as_tensor(parameters, dtype=torch.float64)
    raise ValueError(f"unknown dtype option: {dtype_option}")


def check_row(summary_row, data_root, nt, dtype_option, tol):
    eta = eta_label(summary_row["eta"])
    eta_dir = data_root / f"eta={eta}"
    messages = []
    row = blank_row()
    row["eta"] = eta

    try:
        source_info = load_source_info(eta_dir, messages)
        eta_value = metadata_value(summary_row, source_info, "eta", float)
        d1 = metadata_value(summary_row, source_info, "d1", int)
        d2 = metadata_value(summary_row, source_info, "d2", int)
        j2 = metadata_value(summary_row, source_info, "j2", int)
        randomization = metadata_value(summary_row, source_info, "randomization", int)
        summary_score = metadata_value(summary_row, source_info, "score", float)
        best_index = metadata_value(summary_row, source_info, "best_index", int)

        row.update(
            {
                "d1": d1,
                "d2": d2,
                "j2": j2,
                "randomization": randomization,
                "best_index": best_index,
                "summary_score": f"{summary_score:.17g}",
            }
        )

        parameters_path = eta_dir / "parameters.npy"
        if not parameters_path.exists():
            messages.append("WARNING parameters.npy is missing")
            raise FileNotFoundError(parameters_path)
        parameters = np.load(parameters_path)
        row["parameters_dtype"] = str(parameters.dtype)
        row["parameters_shape"] = "x".join(str(dim) for dim in parameters.shape)

        best_ci_path = eta_dir / "best_feasible_ci.txt"
        if not best_ci_path.exists():
            messages.append("WARNING best_feasible_ci.txt is missing")
            raise FileNotFoundError(best_ci_path)
        saved_best_ci = read_float(best_ci_path)
        row["saved_best_feasible_ci"] = f"{saved_best_ci:.17g}"

        best_index_path = eta_dir / "best_index.txt"
        if not best_index_path.exists():
            messages.append("WARNING best_index.txt is missing")
            raise FileNotFoundError(best_index_path)
        saved_best_index = read_int(best_index_path)
        if saved_best_index != best_index:
            messages.append(
                f"WARNING best_index.txt={saved_best_index} differs from summary/source best_index={best_index}"
            )

        ci_list = maybe_load_history(eta_dir, "ci_list", messages)
        ns_list = maybe_load_history(eta_dir, "ns_list", messages)
        np_list = maybe_load_history(eta_dir, "np_list", messages)
        history_ci = history_value(ci_list, best_index, "ci_list", messages)
        saved_ns = history_value(ns_list, best_index, "ns_list", messages)
        saved_np = history_value(np_list, best_index, "np_list", messages)
        if history_ci is not None:
            row["history_ci_at_best_index"] = f"{history_ci:.17g}"
        if saved_ns is not None:
            row["saved_ns_at_best_index"] = f"{saved_ns:.17g}"
        if saved_np is not None:
            row["saved_np_at_best_index"] = f"{saved_np:.17g}"

        parameter_tensor = parameters_tensor(parameters, dtype_option)
        with torch.no_grad():
            CI, ns_input, np_input, state_RS, state_P = transduction_protocol_CoherentInfo_GKP2(
                eta_value, d1, d2, j2, parameter_tensor, nt, NR=d1
            )

        recomputed_ci = scalar_float(CI)
        recomputed_ns = scalar_float(ns_input)
        recomputed_np = scalar_float(np_input)
        row["recomputed_ci"] = f"{recomputed_ci:.17g}"
        row["recomputed_ns"] = f"{recomputed_ns:.17g}"
        row["recomputed_np"] = f"{recomputed_np:.17g}"

        ci_error_saved = finite_abs_error(recomputed_ci, saved_best_ci)
        ci_error_summary = finite_abs_error(recomputed_ci, summary_score)
        ci_error_history = finite_abs_error(recomputed_ci, history_ci)
        ns_error = finite_abs_error(recomputed_ns, saved_ns)
        np_error = finite_abs_error(recomputed_np, saved_np)

        row["ci_error_vs_saved"] = f"{ci_error_saved:.17g}"
        row["ci_error_vs_summary"] = f"{ci_error_summary:.17g}"
        if ci_error_history is not None:
            row["ci_error_vs_history"] = f"{ci_error_history:.17g}"
        if ns_error is not None:
            row["ns_error"] = f"{ns_error:.17g}"
        if np_error is not None:
            row["np_error"] = f"{np_error:.17g}"

        failures = []
        if ci_error_saved > tol:
            failures.append(f"CI vs best_feasible_ci error {ci_error_saved:.3g} > tol {tol:.3g}")
        if ci_error_summary > tol:
            failures.append(f"CI vs summary score error {ci_error_summary:.3g} > tol {tol:.3g}")
        if ci_error_history is not None and ci_error_history > tol:
            failures.append(f"CI vs history error {ci_error_history:.3g} > tol {tol:.3g}")
        if ns_error is not None and ns_error > ENERGY_TOL:
            failures.append(f"ns error {ns_error:.3g} > {ENERGY_TOL:.3g}")
        if np_error is not None and np_error > ENERGY_TOL:
            failures.append(f"np error {np_error:.3g} > {ENERGY_TOL:.3g}")

        if failures:
            row["status"] = "failed"
            messages.extend(failures)
        else:
            row["status"] = "passed"

    except Exception as exc:
        row["status"] = "failed"
        messages.append(str(exc))

    row["message"] = "; ".join(messages)
    return row


def error_float(row, field):
    value = row.get(field, "")
    if value == "":
        return None
    return float(value)


def max_error(rows, fields):
    values = []
    for row in rows:
        for field in fields:
            value = error_float(row, field)
            if value is not None and math.isfinite(value):
                values.append(value)
    return max(values) if values else None


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def write_json(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute Job 94 selected GKP coherent information from saved parameters."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Processed selected data root. By default tries "
            "Data_HPC/94/Data-Download-partial-10000, "
            "Data-Download-partial-10000, then Data-Download."
        ),
    )
    parser.add_argument("--nt", type=int, default=30)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float64"),
        default="auto",
        help="auto preserves the dtype loaded from parameters.npy.",
    )
    parser.add_argument("--etas", nargs="*", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any selected eta fails validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    data_root = resolve_data_root(args.data_root)
    output_path = resolve_output_path(
        args.output, data_root, "parameter_recompute_validation.tsv"
    )
    json_output_path = resolve_output_path(
        args.json_output, data_root, "parameter_recompute_validation.json"
    )

    selection_rows, warnings = load_selection_rows(
        data_root / "selection_summary.tsv", args.etas
    )
    rows = [
        check_row(summary_row, data_root, args.nt, args.dtype, args.tol)
        for summary_row in selection_rows
    ]

    passed = sum(1 for row in rows if row["status"] == "passed")
    failed_rows = [row for row in rows if row["status"] != "passed"]
    summary = {
        "total_selected_rows_checked": len(rows),
        "passed": passed,
        "failed": len(failed_rows),
        "maximum_ci_error": max_error(
            rows,
            ("ci_error_vs_saved", "ci_error_vs_summary", "ci_error_vs_history"),
        ),
        "maximum_ns_error": max_error(rows, ("ns_error",)),
        "maximum_np_error": max_error(rows, ("np_error",)),
        "failed_etas": [row["eta"] for row in failed_rows],
        "warnings": warnings,
        "arguments": {
            "data_root": str(data_root),
            "nt": args.nt,
            "num_threads": args.num_threads,
            "tol": args.tol,
            "dtype": args.dtype,
            "etas": args.etas,
            "output": str(output_path),
            "json_output": str(json_output_path),
            "strict": args.strict,
        },
    }

    write_tsv(output_path, rows)
    write_json(json_output_path, summary)

    for warning in warnings:
        print(f"WARNING {warning}")
    for row in rows:
        print(
            f"eta={row['eta']} {row['status']} "
            f"ci={row['recomputed_ci']} "
            f"err_saved={row['ci_error_vs_saved']} "
            f"err_summary={row['ci_error_vs_summary']} "
            f"err_history={row['ci_error_vs_history']} "
            f"ns_err={row['ns_error']} np_err={row['np_error']}"
        )
        if row["message"]:
            print(f"  {row['message']}")

    print(
        f"Checked {summary['total_selected_rows_checked']} selected rows: "
        f"{summary['passed']} passed, {summary['failed']} failed."
    )
    print(f"Wrote TSV report: {output_path}")
    print(f"Wrote JSON summary: {json_output_path}")

    if args.strict and failed_rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
