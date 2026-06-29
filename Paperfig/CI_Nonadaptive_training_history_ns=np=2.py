import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
os.environ.setdefault("MPLCONFIGDIR", str(script_dir / ".mplconfig"))
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors as mcolors
from matplotlib import rc
from Quantum_Plotting import *

rc('text', usetex=True)

parent_dir = str(repo_dir)
data_dir = repo_dir / "Data_HPC"
fig_dir = repo_dir / "Figs"

etalist = [0.20, 0.50, 0.80]
depth = 20
Nt = 30
num_seeds = 200
num_steps = None
plot_best_so_far = True
output_filename = "CI_training_history_all_seeds_Data_HPC_84_ns=np=2.jpg"
VQT_RUN_ID = 84
DATA_DOWNLOAD_DIR = "Data-Download-2"


def resolve_training_root():
    requested_path = data_dir / str(VQT_RUN_ID) / DATA_DOWNLOAD_DIR
    if requested_path.exists():
        return requested_path

    lowercase_path = data_dir / str(VQT_RUN_ID) / "Data-download-2"
    if lowercase_path.exists():
        return lowercase_path

    raise FileNotFoundError(f"No training data directory found at {requested_path}")


def load_training_history(eta, seed):
    eta_str = f"{eta:.2f}"
    seed_dir = training_root / f"eta={eta_str}_depth={depth}_Nt={Nt}" / f"seed_{seed}"

    CI_list = np.load(seed_dir / "data_ci_list.npy").astype(float)

    if num_steps is not None:
        CI_list = CI_list[:num_steps]

    return CI_list


def load_eta_training_histories(eta):
    CI_histories = []
    seed_list = []

    for seed in range(num_seeds):
        try:
            CI_list = load_training_history(eta, seed)
        except OSError:
            continue

        if plot_best_so_far:
            CI_list = best_so_far(CI_list)

        CI_histories.append(CI_list)
        seed_list.append(seed)

    if len(CI_histories) == 0:
        raise FileNotFoundError(f"No training history found for eta={eta:.2f}")

    max_ci = np.nanmax([np.nanmax(CI_list) for CI_list in CI_histories])
    print(f"eta={eta:.2f}, number of seeds={len(seed_list)}, max CI={max_ci}")
    return seed_list, CI_histories


def best_so_far(CI_list):
    CI_history = np.maximum.accumulate(np.where(np.isnan(CI_list), -np.inf, CI_list))
    CI_history[np.isneginf(CI_history)] = np.nan
    return CI_history


def final_training_value(CI_list):
    finite_CI_list = CI_list[np.isfinite(CI_list)]
    if len(finite_CI_list) == 0:
        return np.nan
    return finite_CI_list[-1]


def shade_color(base_color, value, vmin, vmax):
    base_rgb = np.array(mcolors.to_rgb(base_color))
    if not np.isfinite(value):
        shade = 0.15
    elif vmax <= vmin:
        shade = 0.75
    else:
        shade = 0.18 + 0.82 * (value - vmin) / (vmax - vmin)

    return tuple((1 - shade) + shade * base_rgb)


training_root = resolve_training_root()


def main():
    fs = 20
    plt.rcParams.update({
        'font.size': fs,
        'axes.labelsize': fs,
        'legend.fontsize': fs - 6,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'lines.linewidth': 1.5,
        'lines.markersize': 5
    })

    fig, axes = plt.subplots(1, len(etalist), figsize=(18, 5), sharey=True)

    if len(etalist) == 1:
        axes = [axes]

    for i, eta in enumerate(etalist):
        seed_list, CI_histories = load_eta_training_histories(eta)
        ax = axes[i]
        final_CI_list = np.array([final_training_value(CI_list) for CI_list in CI_histories])
        finite_final_CI_list = final_CI_list[np.isfinite(final_CI_list)]
        final_CI_min = np.min(finite_final_CI_list)
        final_CI_max = np.max(finite_final_CI_list)
        plot_order = np.argsort(final_CI_list)

        for j in plot_order:
            CI_list = CI_histories[j]
            final_CI = final_CI_list[j]
            steps = np.arange(len(CI_list))
            ax.plot(
                steps,
                CI_list,
                color=shade_color(default_colors[i], final_CI, final_CI_min, final_CI_max),
                alpha=0.75,
                linewidth=0.8,
            )

        ax.set_title(rf"$\eta={eta:.2f}$")
        ax.tick_params(axis='both', which='major')
        ax.set_xlabel("Training step")

    axes[0].set_ylabel("Coherent Information(CI)")

    plt.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / output_filename, dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
