#!/usr/bin/env python3
"""Process Job 96 scratch outputs into compact Data_Download results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ETA = "0.30"
NT = 30
DEPTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18]
SEEDS = range(200)
JOB96_RUN_ID = "96"
REFERENCE_RUN_ID = "84"
REFERENCE_DEPTH = 20


def read_float(path: Path) -> float | None:
    try:
        return float(path.read_text().strip())
    except Exception:
        return None


def path_text(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def default_output_root() -> Path:
    job_dir = Path(__file__).resolve().parent
    return job_dir / "Data_Download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and export best feasible Job 96 results by depth."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root containing scratch eta=..._depth=..._Nt=.../seed_* run outputs.",
    )
    parser.add_argument(
        "--parameters-root",
        type=Path,
        default=Path("training_parameters"),
        help="Root containing matching scratch checkpoint/parameter outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Destination folder for compact downloadable selected outputs.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent / "Data_HPC" / REFERENCE_RUN_ID,
        help=(
            "Read-only Job 84 reference root. Supports processed Data_HPC/84 "
            "layout or raw HPC /home1/.../84 layout."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without copying or writing outputs.",
    )
    return parser.parse_args()


def result_root_name(depth: int) -> str:
    return f"eta={ETA}_depth={depth:02d}_Nt={NT}"


def output_depth_folder(depth: int) -> str:
    return f"eta={ETA}_depth={depth:02d}_Nt={NT}"


def select_depth(depth: int, runs_root: Path, parameters_root: Path) -> dict[str, str]:
    source_folder = result_root_name(depth)
    run_root = runs_root / source_folder
    rows: list[tuple[float, int, Path, Path]] = []

    for seed in SEEDS:
        score_path = run_root / f"seed_{seed}" / "best_feasible_ci.txt"
        score = read_float(score_path)
        if score is None:
            continue
        parameter_path = (
            parameters_root
            / source_folder
            / f"seed_{seed}"
            / "parameters_best_feasible.npy"
        )
        if not parameter_path.exists():
            continue
        rows.append((score, seed, score_path, parameter_path))

    rows.sort(key=lambda item: item[0], reverse=True)

    if not rows:
        return {
            "source_run_id": JOB96_RUN_ID,
            "row_type": "job96_result",
            "depth": str(depth),
            "eta": ETA,
            "Nt": str(NT),
            "output_folder": output_depth_folder(depth),
            "source_depth_folder": source_folder,
            "best_seed": "",
            "best_feasible_ci": "",
            "valid_seed_count": "0",
            "tie_count": "0",
            "score_source": "",
            "parameter_source": "",
            "state_rs_source": "",
            "state_pa_source": "",
            "status": "no_valid_candidate",
        }

    best_score, best_seed, score_path, parameter_path = rows[0]
    state_rs_path = parameter_path.parent / "state_RS_best_feasible.npy"
    state_pa_path = parameter_path.parent / "state_PA_best_feasible.npy"
    tie_count = sum(1 for score, *_ in rows if score == best_score)
    return {
        "source_run_id": JOB96_RUN_ID,
        "row_type": "job96_result",
        "depth": str(depth),
        "eta": ETA,
        "Nt": str(NT),
        "output_folder": output_depth_folder(depth),
        "source_depth_folder": source_folder,
        "best_seed": f"seed_{best_seed}",
        "best_feasible_ci": f"{best_score:.17g}",
        "valid_seed_count": str(len(rows)),
        "tie_count": str(tie_count),
        "score_source": path_text(score_path),
        "parameter_source": path_text(parameter_path),
        "state_rs_source": path_text(state_rs_path) if state_rs_path.exists() else "",
        "state_pa_source": path_text(state_pa_path) if state_pa_path.exists() else "",
        "status": "ok",
    }


def copy_selected(row: dict[str, str], output_root: Path) -> None:
    if row["status"] != "ok":
        return
    dest = output_root / row["output_folder"]
    dest.mkdir(parents=True, exist_ok=True)

    score_source = Path(row["score_source"])
    parameter_source = Path(row["parameter_source"])
    shutil.copy2(score_source, dest / "best_feasible_ci.txt")
    shutil.copy2(parameter_source, dest / "parameters_best_feasible.npy")
    copied_optional = []
    for source_key, filename in (
        ("state_rs_source", "state_RS_best_feasible.npy"),
        ("state_pa_source", "state_PA_best_feasible.npy"),
    ):
        source = row.get(source_key, "")
        if not source:
            continue
        source_path = Path(source)
        if source_path.exists():
            shutil.copy2(source_path, dest / filename)
            copied_optional.append(filename)

    metadata = {
        "selection_summary": row,
        "job": JOB96_RUN_ID,
        "objective": "coherent_information_nonadaptive_VQT_EA_ns_np_2",
        "eta": ETA,
        "Nt": NT,
        "depth": int(row["depth"]),
        "feasibility_rule": "ns <= 2 + 0.01 and np <= 2 + 0.01",
        "scratch_score_source": row["score_source"],
        "scratch_parameter_source": row["parameter_source"],
        "copied_optional_files": copied_optional,
    }
    (dest / "source_info.json").write_text(json.dumps(metadata, indent=2) + "\n")


def select_raw_reference(reference_root: Path) -> dict[str, str] | None:
    source_folder = f"eta={ETA}_depth={REFERENCE_DEPTH}_Nt={NT}"
    run_root = reference_root / "runs" / source_folder
    params_root = reference_root / "training_parameters" / source_folder
    if not run_root.exists() or not params_root.exists():
        return None

    rows: list[tuple[float, int, Path, Path]] = []
    for seed in SEEDS:
        score_path = run_root / f"seed_{seed}" / "best_feasible_ci.txt"
        score = read_float(score_path)
        parameter_path = params_root / f"seed_{seed}" / "parameters_best_feasible.npy"
        if score is None or not parameter_path.exists():
            continue
        rows.append((score, seed, score_path, parameter_path))

    if not rows:
        return None

    rows.sort(key=lambda item: item[0], reverse=True)
    best_score, best_seed, score_path, parameter_path = rows[0]
    tie_count = sum(1 for score, *_ in rows if score == best_score)
    return {
        "source_run_id": REFERENCE_RUN_ID,
        "row_type": "reference",
        "depth": str(REFERENCE_DEPTH),
        "eta": ETA,
        "Nt": str(NT),
        "output_folder": f"eta={ETA}",
        "source_depth_folder": f"run{REFERENCE_RUN_ID}:{source_folder}",
        "best_seed": f"seed_{best_seed}",
        "best_feasible_ci": f"{best_score:.17g}",
        "valid_seed_count": str(len(rows)),
        "tie_count": str(tie_count),
        "score_source": path_text(score_path),
        "parameter_source": path_text(parameter_path),
        "state_rs_source": "",
        "state_pa_source": "",
        "status": "ok",
    }


def add_reference_row(rows: list[dict[str, str]], reference_root: Path) -> list[dict[str, str]]:
    raw_reference = select_raw_reference(reference_root)
    if raw_reference is not None:
        return [*rows, raw_reference]

    ref_folder = f"eta={ETA}"
    score_path = reference_root / ref_folder / "best_feasible_ci.txt"
    parameter_path = reference_root / ref_folder / "parameters_best_feasible.npy"
    score = read_float(score_path)
    status = "ok" if score is not None and parameter_path.exists() else "missing_reference"

    reference = {
        "source_run_id": REFERENCE_RUN_ID,
        "row_type": "reference",
        "depth": str(REFERENCE_DEPTH),
        "eta": ETA,
        "Nt": str(NT),
        "output_folder": ref_folder,
        "source_depth_folder": f"run{REFERENCE_RUN_ID}:{ref_folder}",
        "best_seed": "",
        "best_feasible_ci": "" if score is None else f"{score:.17g}",
        "valid_seed_count": "",
        "tie_count": "",
        "score_source": path_text(score_path),
        "parameter_source": path_text(parameter_path),
        "state_rs_source": "",
        "state_pa_source": "",
        "status": status,
    }
    return [*rows, reference]


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_run_id",
        "row_type",
        "depth",
        "eta",
        "Nt",
        "output_folder",
        "source_depth_folder",
        "best_seed",
        "best_feasible_ci",
        "valid_seed_count",
        "tie_count",
        "score_source",
        "parameter_source",
        "state_rs_source",
        "state_pa_source",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = [
        select_depth(depth, args.runs_root, args.parameters_root)
        for depth in DEPTHS
    ]
    with_reference = add_reference_row(rows, args.reference_root)

    if args.dry_run:
        for row in with_reference:
            print(row)
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        copy_selected(row, args.output_root)
    write_tsv(args.output_root / "selection_summary.tsv", rows)
    write_tsv(args.output_root / "depth_scan_with_job84_reference.tsv", with_reference)
    print(f"Wrote {args.output_root / 'selection_summary.tsv'}")
    print(f"Wrote {args.output_root / 'depth_scan_with_job84_reference.tsv'}")


if __name__ == "__main__":
    main()
