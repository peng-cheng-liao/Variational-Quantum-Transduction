#!/usr/bin/env python3
"""Create a composite appendix figure with additional numerical diagnostics."""

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


DATA_DIR = REPO_ROOT / "Data_HPC"
FIG_DIR = REPO_ROOT / "Figs"

eta_grid_n3 = np.around(np.arange(0.05, 1.0, 0.05), 2)
n_s_n3 = 3
n_p_n3 = 3
VQT_RUN_ID_N3 = 82
VQT_WITHOUT_EA_RUN_ID_N3 = 69
GKP_RUN_ID = 63
GKP_PREFIX = "Transduction_CoherentInfo_GKP_Nt=30_ns=3_np=3_N=2000_processed"

training_etas = [0.20, 0.50, 0.80]
training_depth = 20
training_nt = 30
num_seeds = 200
num_steps = None
TRAINING_RUN_ID = 84
DATA_DOWNLOAD_DIR = "Data-Download-2"

JOB96_SUMMARY_PATH = DATA_DIR / "96" / "depth_scan_with_job84_reference.tsv"
JOB84_REFERENCE_PATH = DATA_DIR / "84" / "eta=0.30" / "best_feasible_ci.txt"
JOB96_DEPTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18]
REFERENCE_DEPTH = 20
ETA = 0.30
NT = 30

default_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def configure_style():
    rc("text", usetex=shutil.which("latex") is not None)
    rc("font", family="serif")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.6,
            "axes.linewidth": 0.8,
        }
    )


def g(x):
    scalar_input = np.isscalar(x)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = (x[mask] + 1.0) * np.log2(x[mask] + 1.0) - x[mask] * np.log2(x[mask])
    if scalar_input:
        return float(out)
    return out


def load_best_feasible_ci(run_id, etas=eta_grid_n3):
    path = DATA_DIR / str(run_id) / f"best_feasible_ci_list_{run_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing best-feasible CI list: {path}")
    ci_list = np.load(path)
    if len(ci_list) != len(etas):
        raise ValueError(f"{path} has {len(ci_list)} entries, expected {len(etas)}")
    return ci_list


def load_gkp_processed_ci():
    run_dir = DATA_DIR / str(GKP_RUN_ID)
    eta_path = run_dir / f"{GKP_PREFIX}_eta_list.npy"
    ci_path = run_dir / f"{GKP_PREFIX}_ci_list.npy"
    if not eta_path.exists():
        raise FileNotFoundError(f"Missing GKP eta list: {eta_path}")
    if not ci_path.exists():
        raise FileNotFoundError(f"Missing GKP CI list: {ci_path}")

    gkp_etas = np.load(eta_path)
    ci_list = np.load(ci_path)
    if len(ci_list) != len(gkp_etas):
        raise ValueError("GKP processed CI and eta arrays have different lengths")
    return gkp_etas, ci_list


def plot_n3_ci(ax):
    vqt_ci = load_best_feasible_ci(VQT_RUN_ID_N3)
    vqt_without_ea_ci = load_best_feasible_ci(VQT_WITHOUT_EA_RUN_ID_N3)

    r = np.arcsinh(np.sqrt(n_p_n3))
    gain = np.cosh(r) ** 2
    tms_ea_ci = []
    qt_ci = []
    for eta in eta_grid_n3:
        eta_ea = 1.0 / (1.0 + (1.0 - eta) / (eta * gain))
        tms_ea_ci.append(max(0.0, g(eta_ea * n_s_n3) - g((1.0 - eta_ea) * n_s_n3)))
        qt_ci.append(max(0.0, g(eta * n_s_n3) - g((1.0 - eta) * n_s_n3)))

    ax.plot(eta_grid_n3, vqt_ci, label="VQT", marker="o", color=default_colors[0])
    ax.plot(
        eta_grid_n3,
        vqt_without_ea_ci,
        label="VQT without EA",
        marker="s",
        linestyle="--",
        color=default_colors[0],
    )
    ax.plot(eta_grid_n3, tms_ea_ci, label="TMS-EA", marker="^", color=default_colors[2])
    ax.plot(eta_grid_n3, qt_ci, label="QT", linestyle="--", marker="v", color=default_colors[3])

    ax.set_xlabel(r"Transmissivity $\eta$")
    ax.set_ylabel("Coherent information")
    ax.set_xlim(0.03, 0.97)
    ax.tick_params(axis="both", which="major")
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.8)

    print("Panel (a) n_S=n_P=3 diagnostics:")
    print(f"  VQT max CI={np.nanmax(vqt_ci):.12g}")
    print(f"  VQT without EA max CI={np.nanmax(vqt_without_ea_ci):.12g}")


def resolve_training_root():
    requested_path = DATA_DIR / str(TRAINING_RUN_ID) / DATA_DOWNLOAD_DIR
    if requested_path.exists():
        return requested_path

    lowercase_path = DATA_DIR / str(TRAINING_RUN_ID) / "Data-download-2"
    if lowercase_path.exists():
        return lowercase_path

    raise FileNotFoundError(
        "No training data directory found. Tried "
        f"{requested_path} and {lowercase_path}."
    )


def best_so_far(ci_list):
    ci_history = np.maximum.accumulate(np.where(np.isnan(ci_list), -np.inf, ci_list))
    ci_history[np.isneginf(ci_history)] = np.nan
    return ci_history


def load_training_histories(eta):
    training_root = resolve_training_root()
    histories = []
    seed_list = []
    eta_str = f"{eta:.2f}"
    eta_dir = training_root / f"eta={eta_str}_depth={training_depth}_Nt={training_nt}"
    if not eta_dir.exists():
        raise FileNotFoundError(f"Missing training directory for eta={eta:.2f}: {eta_dir}")

    def seed_index(seed_dir):
        try:
            return int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return math.inf

    seed_dirs = sorted(eta_dir.glob("seed_*"), key=seed_index)
    for seed_dir in seed_dirs:
        seed = seed_index(seed_dir)
        path = seed_dir / "data_ci_list.npy"
        try:
            ci_list = np.load(path).astype(float)
        except OSError:
            continue

        if num_steps is not None:
            ci_list = ci_list[:num_steps]
        histories.append(best_so_far(ci_list))
        seed_list.append(seed)

    if not histories:
        raise FileNotFoundError(f"No training histories found for eta={eta:.2f} under {eta_dir}")

    final_values = np.array([final_finite_value(history) for history in histories], dtype=float)
    max_ci = np.nanmax([np.nanmax(history) for history in histories])
    print(
        f"eta={eta:.2f}, loaded seeds={len(seed_list)}, "
        f"max final/best-so-far CI={max_ci:.12g}, "
        f"best final CI={np.nanmax(final_values):.12g}, "
        f"final median CI={np.nanmedian(final_values):.12g}"
    )
    return seed_list, histories


def pad_histories(histories):
    max_len = max(len(history) for history in histories)
    padded = np.full((len(histories), max_len), np.nan, dtype=float)
    for i, history in enumerate(histories):
        padded[i, : len(history)] = history
    return padded


def final_finite_value(ci_list):
    finite_ci = ci_list[np.isfinite(ci_list)]
    if len(finite_ci) == 0:
        return np.nan
    return finite_ci[-1]


def plot_training_summary(ax):
    for i, eta in enumerate(training_etas):
        seed_list, histories = load_training_histories(eta)
        final_values = np.array([final_finite_value(history) for history in histories], dtype=float)
        plot_order = np.argsort(final_values)
        color = default_colors[i]
        best_index = int(np.nanargmax(final_values))

        for history_index in plot_order:
            if history_index == best_index:
                continue
            history = histories[history_index]
            steps = np.arange(len(history))
            ax.plot(
                steps,
                history,
                color=color,
                alpha=0.28,
                linewidth=0.55,
            )

        best_history = histories[best_index]
        best_steps = np.arange(len(best_history))
        ax.plot(
            best_steps,
            best_history,
            label=rf"$\eta={eta:.2f}$",
            color=color,
            alpha=1.0,
            linewidth=2.2,
            zorder=5,
        )

        print(
            f"  eta={eta:.2f}, plotted {len(histories)} seed histories, "
            f"best seed={seed_list[best_index]}, best final CI={final_values[best_index]:.12g}"
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Coherent information")
    ax.tick_params(axis="both", which="major")
    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.83, 0.02),
        frameon=False,
        handlelength=2.2,
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.0)
        handle.set_alpha(1.0)


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
        depth_value = parse_float(row["depth"])
        eta = parse_float(row["eta"])
        nt_value = parse_float(row["Nt"])
        ci = parse_float(row["best_feasible_ci"])

        if not math.isfinite(depth_value) or not math.isfinite(nt_value):
            print(
                "Warning: skipping row with invalid depth/Nt "
                f"source_run_id={row.get('source_run_id')} depth={row.get('depth')} Nt={row.get('Nt')}"
            )
            continue
        depth = int(depth_value)
        nt = int(nt_value)

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


def plot_depth_scan(ax):
    points = load_depth_scan_points()
    depths = np.array([depth for depth, _ in points], dtype=float)
    ci_values = np.array([ci for _, ci in points], dtype=float)

    ax.plot(depths, ci_values, color=default_colors[0], marker="o", linestyle="-", zorder=3)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Coherent information")
    ax.set_xticks(depths.astype(int))
    ax.set_xlim(left=depths.min() - 0.75, right=depths.max() + 0.75)
    ax.tick_params(axis="both", which="major")
    ax.grid(True, alpha=0.25, linewidth=0.7)

    print("Panel (c) depth-scan points:")
    for depth, ci in points:
        source = "Job 84 reference" if depth == REFERENCE_DEPTH else "Job 96"
        print(f"  depth={depth:2d}  CI={ci:.12g}  source={source}")


def add_panel_label(ax, label):
    ax.text(
        0.96,
        0.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def main():
    configure_style()
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.2, 1.0],
        width_ratios=[1.0, 1.0],
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    plot_n3_ci(ax_a)
    plot_training_summary(ax_b)
    plot_depth_scan(ax_c)

    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")
    add_panel_label(ax_c, "(c)")

    FIG_DIR.mkdir(exist_ok=True)
    pdf_path = FIG_DIR / "CI_additional_numerical_diagnostics.pdf"
    png_path = FIG_DIR / "CI_additional_numerical_diagnostics.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=500)
    plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
