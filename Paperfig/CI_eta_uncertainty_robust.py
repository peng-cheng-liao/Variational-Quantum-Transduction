#!/usr/bin/env python3
"""Plot eta-uncertainty robust coherent information and Job 98 delta scan."""

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


JOB97_DATA_PATH = REPO_ROOT / "Data_HPC" / "97" / "robust_ci_summary.csv"
JOB98_DIR = REPO_ROOT / "Data_HPC" / "98"
JOB98_DATA_PATHS = [
    JOB98_DIR / "vqt_eta_uncertainty_fixed_eta0_0p30.csv",
    JOB98_DIR / "vqt_eta_uncertainty_fixed_eta0_0p30_delta_0p11_0p20.csv",
]
VQT_OPT_DATA_DIR = REPO_ROOT / "Data_HPC" / "84"
FIG_DIR = REPO_ROOT / "Figs"

SCHEMES = ["VQT", "GKP", "TMS-EA", "QT"]
RETAINED_DELTA = 0.05
JOB98_ETA0 = 0.30
JOB98_EXPECTED_DELTAS = [round(x, 2) for x in np.arange(0.01, 0.201, 0.01)]
OPT_REFERENCE_DELTAS = [0.05, 0.10, 0.15, 0.20]
LABELS = {
    "VQT": "VQT",
    "GKP": "GKP",
    "TMS-EA": "TMS-EA",
    "QT": "QT",
}
COLORS = {
    "VQT": "#1f77b4",
    "GKP": "#ff7f0e",
    "TMS-EA": "#2ca02c",
    "QT": "#d62728",
}
MARKERS = {
    "VQT": "o",
    "GKP": "P",
    "TMS-EA": "^",
    "QT": "v",
}
LINESTYLES = {
    "VQT": "-",
    "GKP": "-.",
    "TMS-EA": "-",
    "QT": "--",
}


def parse_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def delta_key(value):
    return f"{parse_float(value):.2f}"


def read_float(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing optimized VQT reference file: {path}")
    return float(path.read_text().strip())


def eta_folder(eta):
    return f"eta={float(eta):.2f}"


def load_optimized_vqt_ci(eta):
    return read_float(VQT_OPT_DATA_DIR / eta_folder(eta) / "best_feasible_ci.txt")


def load_rows(path, missing_hint):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {missing_hint}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_columns(rows, required, path):
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {', '.join(missing)}")


def warn_missing_job97(rows):
    missing = []
    for row in rows:
        ci_avg = parse_float(row.get("ci_avg"))
        if row.get("status") != "ok" or not math.isfinite(ci_avg):
            missing.append(
                f"{row.get('scheme')} delta={row.get('delta')} eta0={row.get('eta0')} status={row.get('status')}"
            )
    if missing:
        print("Warning: missing or invalid Job 97 robust-CI points:")
        for item in missing:
            print(f"  {item}")


def grouped_job97_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scheme"], delta_key(row["delta"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: parse_float(row["eta0"]))
    return grouped


def load_job97():
    rows = load_rows(
        JOB97_DATA_PATH,
        "Run Job Submission/97/calculate_eta_uncertainty_robust_ci.py first.",
    )
    validate_columns(
        rows,
        {"scheme", "eta0", "delta", "ci_avg", "status"},
        JOB97_DATA_PATH,
    )
    warn_missing_job97(rows)
    return grouped_job97_rows(rows)


def load_job98():
    if not JOB98_DIR.exists():
        raise FileNotFoundError(
            f"Missing {JOB98_DIR}. Run Job Submission/98/calculate_fixed_eta0_vqt_uncertainty.py first."
        )
    rows = []
    for path in JOB98_DATA_PATHS:
        path_rows = load_rows(
            path,
            "Run Job Submission/98/calculate_fixed_eta0_vqt_uncertainty.py for the missing delta range first.",
        )
        validate_columns(
            path_rows,
            {
                "eta0",
                "delta",
                "eta_eval_minus",
                "eta_eval_plus",
                "CI_minus",
                "CI_plus",
            },
            path,
        )
        rows.extend(path_rows)
    rows.sort(key=lambda row: parse_float(row["delta"]))
    deltas = [round(parse_float(row["delta"]), 2) for row in rows]
    if len(deltas) != len(set(deltas)):
        raise RuntimeError(f"Duplicate Job 98 delta values found across {JOB98_DATA_PATHS}: {deltas}")
    if deltas != JOB98_EXPECTED_DELTAS:
        raise RuntimeError(
            f"Job 98 delta grid mismatch across {JOB98_DATA_PATHS}: expected {JOB98_EXPECTED_DELTAS}, got {deltas}"
        )
    return rows


def load_job98_reference_points():
    reference = {
        "nominal": load_optimized_vqt_ci(JOB98_ETA0),
        "minus": [],
        "plus": [],
    }
    for delta in OPT_REFERENCE_DELTAS:
        reference["minus"].append((delta, load_optimized_vqt_ci(JOB98_ETA0 - delta)))
        reference["plus"].append((delta, load_optimized_vqt_ci(JOB98_ETA0 + delta)))
    return reference


def plot_job97_delta(ax, grouped):
    delta = delta_key(RETAINED_DELTA)
    for scheme in SCHEMES:
        scheme_rows = grouped.get((scheme, delta), [])
        eta_values = np.array([parse_float(row["eta0"]) for row in scheme_rows], dtype=float)
        ci_values = np.array([parse_float(row["ci_avg"]) for row in scheme_rows], dtype=float)
        finite = np.isfinite(eta_values) & np.isfinite(ci_values) & (eta_values < 0.95)
        if not np.any(finite):
            print(f"Warning: no finite Job 97 points for {scheme} at delta={delta}")
            continue
        ax.plot(
            eta_values[finite],
            ci_values[finite],
            label=LABELS[scheme],
            color=COLORS[scheme],
            marker=MARKERS[scheme],
            linestyle=LINESTYLES[scheme],
        )

    ax.set_title(r"$\delta=" + delta + r"$")
    ax.set_xlabel(r"Nominal transmissivity $\eta_0$")
    ax.set_ylabel("Average robust CI")
    ax.tick_params(axis="both", which="major")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper left", frameon=False)


def plot_job98_delta_scan(ax, rows, reference):
    deltas = np.array([0.0] + [parse_float(row["delta"]) for row in rows], dtype=float)
    ci_minus = np.array(
        [reference["nominal"]] + [parse_float(row["CI_minus"]) for row in rows],
        dtype=float,
    )
    ci_plus = np.array(
        [reference["nominal"]] + [parse_float(row["CI_plus"]) for row in rows],
        dtype=float,
    )
    finite_minus = np.isfinite(deltas) & np.isfinite(ci_minus)
    finite_plus = np.isfinite(deltas) & np.isfinite(ci_plus)
    if not np.all(finite_minus) or not np.all(finite_plus):
        raise RuntimeError(f"Job 98 data contain non-finite CI values in {JOB98_DATA_PATHS}")

    ax.plot(
        deltas,
        ci_minus,
        label="negative deviation",
        color=COLORS["VQT"],
        marker="o",
        linestyle="-",
    )
    ax.plot(
        deltas,
        ci_plus,
        label="positive deviation",
        color=COLORS["GKP"],
        marker="s",
        linestyle="--",
    )
    ref_minus = np.array(reference["minus"], dtype=float)
    ref_plus = np.array(reference["plus"], dtype=float)
    ax.scatter(
        ref_minus[:, 0],
        ref_minus[:, 1],
        label="negative optimized",
        color=COLORS["VQT"],
        marker="D",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )
    ax.scatter(
        ref_plus[:, 0],
        ref_plus[:, 1],
        label="positive optimized",
        color=COLORS["GKP"],
        marker="D",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )
    ax.set_title(r"$\eta_0 = 0.30$")
    ax.set_xlabel(r"Uncertainty $\delta$")
    ax.set_ylabel("Coherent information")
    ax.set_xlim(left=0.0)
    ax.tick_params(axis="both", which="major")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="best", frameon=False)


def main():
    job97_grouped = load_job97()
    job98_rows = load_job98()
    job98_reference = load_job98_reference_points()

    rc("text", usetex=True)
    rc("font", family="serif")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    plot_job97_delta(axes[0], job97_grouped)
    plot_job98_delta_scan(axes[1], job98_rows, job98_reference)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    jpg_path = FIG_DIR / "CI_eta_uncertainty_robust.jpg"
    pdf_path = FIG_DIR / "CI_eta_uncertainty_robust.pdf"
    fig.savefig(jpg_path, dpi=500)
    fig.savefig(pdf_path)
    print(f"Wrote {jpg_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
