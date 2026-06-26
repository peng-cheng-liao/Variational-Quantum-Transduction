#!/usr/bin/env python3
"""Plot Job 96 best-feasible coherent information vs variational depth.

Depths 2--18 are Job 96 non-adaptive VQT-EA results at eta=0.30, Nt=30,
ns=np=2. The depth-20 point is the matching Job 84 reference at eta=0.30.
"""

import csv
import math
import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


JOB96_SUMMARY_PATH = REPO_ROOT / "Data_HPC" / "96" / "depth_scan_with_job84_reference.tsv"
JOB84_REFERENCE_PATH = REPO_ROOT / "Data_HPC" / "84" / "eta=0.30" / "best_feasible_ci.txt"
FIG_DIR = REPO_ROOT / "Figs"

JOB96_DEPTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18]
REFERENCE_DEPTH = 20
ETA = 0.30
NT = 30

COLOR = "#1f77b4"


def parse_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def read_float(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing Job 84 reference file: {path}")
    return float(path.read_text().strip())


def load_summary_rows(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Job 96 depth summary: {path}. Run Job Submission/96/process_depth_scan_96.py first."
        )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    required = {
        "source_run_id",
        "row_type",
        "depth",
        "eta",
        "Nt",
        "best_feasible_ci",
        "status",
    }
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {', '.join(missing)}")
    return rows


def load_depth_scan_points():
    rows = load_summary_rows(JOB96_SUMMARY_PATH)
    points = {}
    reference_from_summary = None

    for row in rows:
        depth = int(parse_float(row["depth"]))
        eta = parse_float(row["eta"])
        nt = int(parse_float(row["Nt"]))
        ci = parse_float(row["best_feasible_ci"])
        if row["status"] != "ok" or not math.isfinite(ci):
            print(
                "Warning: skipping invalid row "
                f"source_run_id={row.get('source_run_id')} depth={row.get('depth')} status={row.get('status')}"
            )
            continue
        if not math.isclose(eta, ETA, rel_tol=0.0, abs_tol=1e-12) or nt != NT:
            print(
                "Warning: skipping row with unexpected eta/Nt "
                f"source_run_id={row.get('source_run_id')} depth={depth} eta={eta} Nt={nt}"
            )
            continue

        if row["source_run_id"] == "96" and depth in JOB96_DEPTHS:
            points[depth] = ci
        elif row["source_run_id"] == "84" and depth == REFERENCE_DEPTH:
            reference_from_summary = ci

    missing_depths = [depth for depth in JOB96_DEPTHS if depth not in points]
    if missing_depths:
        print(f"Warning: missing Job 96 depths in {JOB96_SUMMARY_PATH}: {missing_depths}")

    reference_ci = read_float(JOB84_REFERENCE_PATH)
    if reference_from_summary is not None and not math.isclose(
        reference_from_summary, reference_ci, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "Job 84 reference mismatch: "
            f"{JOB96_SUMMARY_PATH} has {reference_from_summary}, "
            f"but {JOB84_REFERENCE_PATH} has {reference_ci}"
        )
    points[REFERENCE_DEPTH] = reference_ci

    if not points:
        raise RuntimeError("No finite depth-scan points were loaded")
    return sorted(points.items())


def configure_style():
    rc("text", usetex=shutil.which("latex") is not None)
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


def plot_depth_scan(points):
    depths = np.array([depth for depth, _ in points], dtype=float)
    ci_values = np.array([ci for _, ci in points], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    ax.plot(
        depths,
        ci_values,
        label="VQT-EA",
        color=COLOR,
        marker="o",
        linestyle="-",
        zorder=3,
    )

    ref_mask = depths == REFERENCE_DEPTH
    if np.any(ref_mask):
        ax.plot(
            depths[ref_mask],
            ci_values[ref_mask],
            label="Job 84 reference",
            color=COLOR,
            marker="D",
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="None",
            zorder=4,
        )

    ax.set_xlabel("Depth")
    ax.set_ylabel("Best feasible coherent information")
    ax.set_xticks(depths.astype(int))
    ax.set_xlim(left=depths.min() - 0.75, right=depths.max() + 0.75)
    ax.tick_params(axis="both", which="major")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def main():
    points = load_depth_scan_points()
    print("Loaded CI depth-scan points:")
    for depth, ci in points:
        source = "Job 84 reference" if depth == REFERENCE_DEPTH else "Job 96"
        print(f"  depth={depth:2d}  CI={ci:.12g}  source={source}")

    configure_style()
    fig = plot_depth_scan(points)

    FIG_DIR.mkdir(exist_ok=True)
    pdf_path = FIG_DIR / "CI_depth_scan_96.pdf"
    png_path = FIG_DIR / "CI_depth_scan_96.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=500)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
