#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SOURCE_ROOT = Path("/home1/liaopeng/QuantumTransduction/64_v2/Data")
DEFAULT_OUTPUT_ROOT = Path("/home1/liaopeng/QuantumTransduction/64_v2/Data-download-2")
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2)
REQUIRED_SUFFIXES = ("ci_list", "ns_list", "np_list", "parameters")
OPTIONAL_SUFFIXES = ("state_P", "state_RS")
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
]

FOLDER_RE = re.compile(
    r"^eta=(?P<eta>[0-9]+(?:\.[0-9]+)?)_d1=(?P<d1>\d+)_d2=(?P<d2>\d+)$"
)
FILE_RE = re.compile(
    r"^j2=(?P<j2>\d+)_randomization=(?P<randomization>\d+)_"
    r"(?P<suffix>ci_list|ns_list|np_list|parameters|state_P|state_RS)\.npy$"
)


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


def eta_label(eta):
    return f"{float(eta):.2f}"


def discover_candidates(source_root):
    candidates = {}
    available_etas = set()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

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


def load_score(key, files, energy_limit):
    ci_data = np.load(files["ci_list"])
    ns_data = np.load(files["ns_list"])
    np_data = np.load(files["np_list"])

    if ci_data.shape != ns_data.shape or ci_data.shape != np_data.shape:
        raise ValueError(
            "shape mismatch: "
            f"ci_list{ci_data.shape}, ns_list{ns_data.shape}, np_list{np_data.shape}"
        )
    if ci_data.size == 0:
        raise ValueError("empty ci/ns/np arrays")

    ci_masked = np.array(ci_data, copy=True)
    mask = np.logical_or(np.array(ns_data) >= energy_limit, np.array(np_data) >= energy_limit)
    ci_masked[mask] = 0
    if np.all(np.isnan(ci_masked)):
        raise ValueError("all feasible CI values are NaN")

    return CandidateResult(
        key=key,
        score=float(np.nanmax(ci_masked)),
        best_index=int(np.nanargmax(ci_masked)),
    )


def select_best(candidates, etas, energy_limit):
    best_by_eta = {}
    stats = {
        "total_candidates_scanned": len(candidates),
        "valid_candidates": 0,
        "skipped_missing_parameters": 0,
        "skipped_missing_required_arrays": 0,
        "skipped_invalid_arrays": 0,
    }
    invalid_messages = []

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
        missing_required = [suffix for suffix in REQUIRED_SUFFIXES if suffix not in files]
        if "parameters" in missing_required:
            stats["skipped_missing_parameters"] += 1
            continue
        if missing_required:
            stats["skipped_missing_required_arrays"] += 1
            continue

        try:
            result = load_score(key, files, energy_limit)
        except Exception as exc:
            stats["skipped_invalid_arrays"] += 1
            invalid_messages.append(f"{key.folder / key.prefix}: {exc}")
            continue

        stats["valid_candidates"] += 1
        current = best_by_eta.get(key.eta)
        if current is None or result.score > current.score:
            best_by_eta[key.eta] = result

    no_valid_etas = [eta for eta in etas if eta not in best_by_eta]
    return best_by_eta, no_valid_etas, stats, invalid_messages


def copied_and_missing_optional(result, files):
    copied = [f"{suffix}.npy" for suffix in REQUIRED_SUFFIXES]
    missing_optional = []
    for suffix in OPTIONAL_SUFFIXES:
        if suffix in files:
            copied.append(f"{suffix}.npy")
        else:
            missing_optional.append(f"{suffix}.npy")
    return copied, missing_optional


def summary_row_for_result(result, files, output_root, status):
    key = result.key
    dest = output_root / f"eta={eta_label(key.eta)}"
    copied, missing_optional = copied_and_missing_optional(result, files)
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
        "copied_files": ",".join(copied),
        "missing_optional_files": ",".join(missing_optional),
        "dest": str(dest),
        "folder": str(key.folder),
        "prefix": str(key.folder / key.prefix),
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
    }


def ensure_output_path(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")


def copy_selected(best_by_eta, candidates, output_root, overwrite):
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for eta, result in sorted(best_by_eta.items()):
        key = result.key
        files = candidates[key]
        dest = output_root / f"eta={eta_label(eta)}"
        dest.mkdir(parents=True, exist_ok=True)

        for suffix in REQUIRED_SUFFIXES + OPTIONAL_SUFFIXES:
            source = files.get(suffix)
            if source is None:
                continue
            target = dest / f"{suffix}.npy"
            ensure_output_path(target, overwrite)
            shutil.copy2(source, target)

        copied, missing_optional = copied_and_missing_optional(result, files)
        info = {
            "eta": eta_label(key.eta),
            "score": f"{result.score:.17g}",
            "best_index": str(result.best_index),
            "d1": str(key.d1),
            "d2": str(key.d2),
            "j2": str(key.j2),
            "randomization": str(key.randomization),
            "source_folder": str(key.folder),
            "source_prefix": str(key.folder / key.prefix),
            "copied_files": copied,
            "missing_optional_files": missing_optional,
            "selection_note": "Candidates without parameters.npy were skipped.",
        }
        info_path = dest / "source_info.json"
        ensure_output_path(info_path, overwrite)
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        rows.append(summary_row_for_result(result, files, output_root, "selected"))
    return rows


def write_summary(output_root, rows, overwrite):
    path = output_root / "selection_summary.tsv"
    ensure_output_path(path, overwrite)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    return path


def print_selection(best_by_eta, no_valid_etas):
    for eta, result in sorted(best_by_eta.items()):
        key = result.key
        print(
            "Selected "
            f"eta={eta_label(eta)} "
            f"score={result.score:.17g} "
            f"best_index={result.best_index} "
            f"d1={key.d1} d2={key.d2} j2={key.j2} "
            f"randomization={key.randomization} "
            f"prefix={key.folder / key.prefix}"
        )
    if no_valid_etas:
        print("Etas with no valid candidate: " + ", ".join(eta_label(eta) for eta in no_valid_etas))
    else:
        print("Etas with no valid candidate: none")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and copy optimal feasible GKP results for 64_v2."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--energy-limit", type=float, default=2.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates, etas = discover_candidates(args.source_root)
    if not etas:
        etas = [float(eta) for eta in DEFAULT_ETAS]

    best_by_eta, no_valid_etas, stats, invalid_messages = select_best(
        candidates, etas, args.energy_limit
    )

    print(f"Source root: {args.source_root}")
    print(f"Output root: {args.output_root}")
    print(f"Energy limit: {args.energy_limit}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    if invalid_messages:
        print("Invalid candidates:")
        for message in invalid_messages[:20]:
            print(f"  {message}")
        if len(invalid_messages) > 20:
            print(f"  ... {len(invalid_messages) - 20} more")
    print_selection(best_by_eta, no_valid_etas)

    no_valid_rows = [no_valid_row(eta) for eta in no_valid_etas]

    if args.dry_run:
        print("Dry run: no files copied and no summary written.")
        return

    copied_rows = copy_selected(best_by_eta, candidates, args.output_root, args.overwrite)
    rows = copied_rows + no_valid_rows
    summary_path = write_summary(args.output_root, rows, args.overwrite)
    print(f"Wrote {summary_path}")
    print(f"Selected eta folders written: {len(copied_rows)}")


if __name__ == "__main__":
    main()
