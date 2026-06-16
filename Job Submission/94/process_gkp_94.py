#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = BASE_DIR / "Data"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "Data-Download"
DEFAULT_ETAS = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)]
VERIFY_TOL = 1e-8

SUMMARY_FIELDS = [
    "eta",
    "eta_source",
    "score",
    "best_index",
    "randomization",
    "d1",
    "d2",
    "j2",
    "status",
    "copied_files",
    "missing_optional_files",
    "dest",
    "folder",
    "prefix",
    "verification_status",
    "verification_abs_error",
]

FOLDER_RE = re.compile(
    r"^eta=(?P<eta>[0-9]+(?:\.[0-9]+)?)_d1=(?P<d1>\d+)_d2=(?P<d2>\d+)$"
)
FILE_RE = re.compile(
    r"^j2=(?P<j2>\d+)_randomization=(?P<randomization>\d+)_"
    r"(?P<suffix>ci_list|ns_list|np_list|best_parameters|best_state_RS|best_state_P|"
    r"best_index|best_feasible_ci|best_ns|best_np|final_parameters)"
    r"(?P<ext>\.npy|\.txt)$"
)

REQUIRED_ARRAYS = ("ci_list", "ns_list", "np_list")
REQUIRED_BEST = ("best_parameters",)
OPTIONAL_BEST = ("best_state_RS", "best_state_P")


@dataclass(frozen=True)
class CandidateKey:
    eta: float
    eta_source: str
    d1: int
    d2: int
    j2: int
    randomization: int
    folder: Path
    prefix: str


@dataclass
class CandidateResult:
    key: CandidateKey
    score: float
    best_index: int
    verification_status: str
    verification_abs_error: float
    missing_optional_files: list[str]


def eta_label(eta):
    return f"{float(eta):.2f}"


def text_float(path):
    return float(path.read_text().strip())


def text_int(path):
    return int(path.read_text().strip())


def discover_candidates(source_root):
    candidates = {}
    available_etas = set()
    if not source_root.exists():
        return candidates, []

    for folder in sorted(source_root.iterdir()):
        if not folder.is_dir():
            continue
        folder_match = FOLDER_RE.match(folder.name)
        if not folder_match:
            continue

        eta_source = folder_match.group("eta")
        eta = round(float(eta_source), 2)
        d1 = int(folder_match.group("d1"))
        d2 = int(folder_match.group("d2"))
        available_etas.add(eta)

        for path in sorted(folder.iterdir()):
            file_match = FILE_RE.match(path.name)
            if not file_match:
                continue

            j2 = int(file_match.group("j2"))
            randomization = int(file_match.group("randomization"))
            prefix = f"j2={j2}_randomization={randomization}_"
            key = CandidateKey(
                eta=eta,
                eta_source=eta_source,
                d1=d1,
                d2=d2,
                j2=j2,
                randomization=randomization,
                folder=folder,
                prefix=prefix,
            )
            candidates.setdefault(key, {})[file_match.group("suffix")] = path

    return candidates, sorted(available_etas)


def masked_best(ci_data, ns_data, np_data, energy_limit):
    if ci_data.shape != ns_data.shape or ci_data.shape != np_data.shape:
        raise ValueError(
            "shape mismatch: "
            f"ci_list{ci_data.shape}, ns_list{ns_data.shape}, np_list{np_data.shape}"
        )
    if ci_data.size == 0:
        raise ValueError("empty ci/ns/np arrays")

    feasible = (ns_data < energy_limit) & (np_data < energy_limit) & np.isfinite(ci_data)
    if not np.any(feasible):
        raise ValueError("no feasible finite CI values")

    feasible_indices = np.flatnonzero(feasible)
    feasible_scores = ci_data[feasible]
    local_best = int(np.nanargmax(feasible_scores))
    best_index = int(feasible_indices[local_best])
    score = float(feasible_scores[local_best])
    return score, best_index


def evaluate_candidate(key, files, energy_limit, strict):
    missing = [name for name in REQUIRED_ARRAYS + REQUIRED_BEST if name not in files]
    if missing:
        raise ValueError("missing required files: " + ",".join(missing))

    ci_data = np.load(files["ci_list"])
    ns_data = np.load(files["ns_list"])
    np_data = np.load(files["np_list"])
    history_score, history_index = masked_best(ci_data, ns_data, np_data, energy_limit)

    has_metadata = "best_feasible_ci" in files and "best_index" in files
    if has_metadata:
        saved_score = text_float(files["best_feasible_ci"])
        saved_index = text_int(files["best_index"])
        abs_error = abs(saved_score - history_score)
        index_matches = saved_index == history_index
        if abs_error > VERIFY_TOL or not index_matches:
            message = (
                f"best metadata mismatch for {key.folder / key.prefix}: "
                f"saved_score={saved_score:.17g}, history_score={history_score:.17g}, "
                f"saved_index={saved_index}, history_index={history_index}, "
                f"abs_error={abs_error:.3g}"
            )
            if strict:
                raise ValueError(message)
            raise ValueError(message)
        score = saved_score
        best_index = saved_index
        verification_status = "verified"
        verification_abs_error = abs_error
    else:
        if strict:
            raise ValueError("missing best_feasible_ci.txt or best_index.txt")
        score = history_score
        best_index = history_index
        verification_status = "computed_from_histories_missing_metadata"
        verification_abs_error = math.nan

    missing_optional = [
        f"{name}.npy" for name in OPTIONAL_BEST if name not in files
    ]
    return CandidateResult(
        key=key,
        score=score,
        best_index=best_index,
        verification_status=verification_status,
        verification_abs_error=verification_abs_error,
        missing_optional_files=missing_optional,
    )


def select_best(candidates, target_etas, energy_limit, strict):
    best_by_eta = {}
    stats = {
        "total_candidates_scanned": len(candidates),
        "valid_candidates": 0,
        "invalid_candidates": 0,
    }
    warnings = []

    for key, files in sorted(
        candidates.items(),
        key=lambda item: (
            item[0].eta,
            item[0].d1,
            item[0].d2,
            item[0].j2,
            item[0].randomization,
        ),
    ):
        try:
            result = evaluate_candidate(key, files, energy_limit, strict)
        except Exception as exc:
            stats["invalid_candidates"] += 1
            warnings.append(str(exc))
            continue

        stats["valid_candidates"] += 1
        current = best_by_eta.get(key.eta)
        if current is None or result.score > current.score:
            best_by_eta[key.eta] = result

    no_valid_etas = [eta for eta in target_etas if eta not in best_by_eta]
    return best_by_eta, no_valid_etas, stats, warnings


def ensure_output_path(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")


def copy_file(source, target, overwrite):
    ensure_output_path(target, overwrite)
    shutil.copy2(source, target)


def copied_files_for(result, files):
    copied = [
        "ci_list.npy",
        "ns_list.npy",
        "np_list.npy",
        "parameters.npy",
        "best_index.txt",
        "best_feasible_ci.txt",
        "source_info.json",
    ]
    if "best_state_RS" in files:
        copied.append("state_RS.npy")
    if "best_state_P" in files:
        copied.append("state_P.npy")
    return copied


def summary_row(result, files, output_root, status):
    key = result.key
    dest = output_root / f"eta={eta_label(key.eta)}"
    abs_error = result.verification_abs_error
    return {
        "eta": eta_label(key.eta),
        "eta_source": key.eta_source,
        "score": f"{result.score:.17g}",
        "best_index": result.best_index,
        "randomization": key.randomization,
        "d1": key.d1,
        "d2": key.d2,
        "j2": key.j2,
        "status": status,
        "copied_files": ",".join(copied_files_for(result, files)),
        "missing_optional_files": ",".join(result.missing_optional_files),
        "dest": str(dest),
        "folder": str(key.folder),
        "prefix": str(key.folder / key.prefix),
        "verification_status": result.verification_status,
        "verification_abs_error": "" if math.isnan(abs_error) else f"{abs_error:.17g}",
    }


def no_valid_row(eta):
    return {
        "eta": eta_label(eta),
        "eta_source": "",
        "score": "",
        "best_index": "",
        "randomization": "",
        "d1": "",
        "d2": "",
        "j2": "",
        "status": "no_valid_candidate",
        "copied_files": "",
        "missing_optional_files": "",
        "dest": "",
        "folder": "",
        "prefix": "",
        "verification_status": "",
        "verification_abs_error": "",
    }


def copy_selected(best_by_eta, candidates, output_root, overwrite, dry_run):
    rows = []
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    for eta, result in sorted(best_by_eta.items()):
        key = result.key
        files = candidates[key]
        dest = output_root / f"eta={eta_label(eta)}"

        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            copy_file(files["ci_list"], dest / "ci_list.npy", overwrite)
            copy_file(files["ns_list"], dest / "ns_list.npy", overwrite)
            copy_file(files["np_list"], dest / "np_list.npy", overwrite)
            copy_file(files["best_parameters"], dest / "parameters.npy", overwrite)

            if "best_state_RS" in files:
                copy_file(files["best_state_RS"], dest / "state_RS.npy", overwrite)
            if "best_state_P" in files:
                copy_file(files["best_state_P"], dest / "state_P.npy", overwrite)

            best_index_path = dest / "best_index.txt"
            best_ci_path = dest / "best_feasible_ci.txt"
            ensure_output_path(best_index_path, overwrite)
            ensure_output_path(best_ci_path, overwrite)
            best_index_path.write_text(f"{result.best_index}\n")
            best_ci_path.write_text(f"{result.score:.17g}\n")

            info = {
                "eta": eta_label(key.eta),
                "eta_source": key.eta_source,
                "score": f"{result.score:.17g}",
                "best_index": result.best_index,
                "d1": key.d1,
                "d2": key.d2,
                "j2": key.j2,
                "randomization": key.randomization,
                "source_folder": str(key.folder),
                "source_prefix": str(key.folder / key.prefix),
                "copied_files": copied_files_for(result, files),
                "missing_optional_files": result.missing_optional_files,
                "verification_status": result.verification_status,
                "verification_abs_error": result.verification_abs_error,
                "selection_note": "Candidates without best_parameters.npy were skipped.",
            }
            info_path = dest / "source_info.json"
            ensure_output_path(info_path, overwrite)
            info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")

        rows.append(summary_row(result, files, output_root, "selected_dry_run" if dry_run else "selected"))

    return rows


def write_summary(output_root, rows, overwrite, dry_run):
    path = output_root / "selection_summary.tsv"
    if dry_run:
        return path
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_output_path(path, overwrite)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    return path


def print_selection(best_by_eta, no_valid_etas, stats, warnings, strict):
    for eta, result in sorted(best_by_eta.items()):
        key = result.key
        print(
            "Selected "
            f"eta={eta_label(eta)} "
            f"score={result.score:.17g} "
            f"best_index={result.best_index} "
            f"d1={key.d1} d2={key.d2} j2={key.j2} "
            f"randomization={key.randomization} "
            f"verification={result.verification_status} "
            f"prefix={key.folder / key.prefix}"
        )
    if no_valid_etas:
        print("Etas with no valid candidate: " + ", ".join(eta_label(eta) for eta in no_valid_etas))
    else:
        print("Etas with no valid candidate: none")

    print(
        "Candidate stats: "
        f"scanned={stats['total_candidates_scanned']} "
        f"valid={stats['valid_candidates']} "
        f"invalid={stats['invalid_candidates']}"
    )
    if warnings:
        label = "Invalid candidates" if strict else "Warnings"
        print(f"{label}:")
        for warning in warnings[:50]:
            print(f"  {warning}")
        if len(warnings) > 50:
            print(f"  ... {len(warnings) - 50} more")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select optimal feasible GKP job 94 results into Data-Download."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--energy-limit", type=float, default=2.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail candidates missing best metadata instead of computing from histories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    candidates, available_etas = discover_candidates(args.source_root)
    target_etas = sorted(set(DEFAULT_ETAS).union(available_etas))
    best_by_eta, no_valid_etas, stats, warnings = select_best(
        candidates, target_etas, args.energy_limit, args.strict
    )
    rows = copy_selected(
        best_by_eta, candidates, args.output_root, args.overwrite, args.dry_run
    )
    rows.extend(no_valid_row(eta) for eta in no_valid_etas)
    summary_path = write_summary(args.output_root, rows, args.overwrite, args.dry_run)
    print_selection(best_by_eta, no_valid_etas, stats, warnings, args.strict)
    if args.dry_run:
        print(f"Dry run: would write summary to {summary_path}")
    else:
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
