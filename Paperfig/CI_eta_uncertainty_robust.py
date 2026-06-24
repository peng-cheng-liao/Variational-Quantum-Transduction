#!/usr/bin/env python3
"""Plot Job 97 eta-uncertainty robust coherent information."""

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


DATA_PATH = REPO_ROOT / "Data_HPC" / "97" / "robust_ci_summary.csv"
FIG_DIR = REPO_ROOT / "Figs"

SCHEMES = ["VQT", "GKP", "TMS-EA", "QT"]
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


def load_rows(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Job Submission/97/calculate_eta_uncertainty_robust_ci.py first."
        )
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def warn_missing(rows):
    missing = []
    for row in rows:
        ci_avg = parse_float(row.get("ci_avg"))
        if row.get("status") != "ok" or not math.isfinite(ci_avg):
            missing.append(
                f"{row.get('scheme')} delta={row.get('delta')} eta0={row.get('eta0')} status={row.get('status')}"
            )
    if missing:
        print("Warning: missing or invalid robust-CI points:")
        for item in missing:
            print(f"  {item}")


def grouped_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scheme"], row["delta"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: parse_float(row["eta0"]))
    return grouped


def main():
    rows = load_rows(DATA_PATH)
    warn_missing(rows)
    grouped = grouped_rows(rows)
    deltas = sorted({row["delta"] for row in rows}, key=parse_float)
    if not deltas:
        raise RuntimeError("No delta values found in robust CI summary")

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

    fig, axes = plt.subplots(1, len(deltas), figsize=(5.4 * len(deltas), 4.2), sharey=True)
    if len(deltas) == 1:
        axes = [axes]

    for ax, delta in zip(axes, deltas):
        for scheme in SCHEMES:
            scheme_rows = grouped.get((scheme, delta), [])
            eta_values = np.array([parse_float(row["eta0"]) for row in scheme_rows], dtype=float)
            ci_values = np.array([parse_float(row["ci_avg"]) for row in scheme_rows], dtype=float)
            finite = np.isfinite(eta_values) & np.isfinite(ci_values)
            if not np.any(finite):
                print(f"Warning: no finite points for {scheme} at delta={delta}")
                continue
            ax.plot(
                eta_values[finite],
                ci_values[finite],
                label=LABELS[scheme],
                color=COLORS[scheme],
                marker=MARKERS[scheme],
                linestyle=LINESTYLES[scheme],
            )

        ax.set_title(r"$\delta=" + f"{parse_float(delta):.2f}" + r"$")
        ax.set_xlabel(r"Nominal transmissivity $\eta_0$")
        ax.tick_params(axis="both", which="major")
        ax.grid(True, alpha=0.25, linewidth=0.7)

    axes[0].set_ylabel("Average robust CI")
    axes[0].legend(loc="upper left", frameon=False)

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
