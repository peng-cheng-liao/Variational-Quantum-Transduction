import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import qutip as qt

from Quantum_Plotting import wignermatrix, moments_onemode, qt2torch  # keep your originals

rc("text", usetex=True)

# -------------------- config --------------------
Nt = 30
n_s = 2
n_p = 2

fs = 35
qplimit = 6
bins = 250
vmin, vmax = -0.05, 0.31
levels = np.arange(vmin, vmax + 1e-12, 0.01)
cmap = "RdBu_r"

plt.rcParams.update({
    "font.size": fs,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

eta_list = np.around(np.arange(0.1, 1.0, 0.1), 1)

# precompute qp-grid once
q = np.linspace(-qplimit, qplimit, bins)
p = np.linspace(-qplimit, qplimit, bins)
X, Y = np.meshgrid(q, p)

# shared color scaling (so both row colorbars match the panels)
norm = mpl.colors.BoundaryNorm(levels, ncolors=mpl.colormaps[cmap].N, clip=True)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# -------------------- helpers --------------------
def g(x):
    return (x + 1) * np.log2(x + 1) - x * np.log2(x)

def best_randomization_indices_45(eta_list, randomizationlist=range(20), depth=20, N=10000):
    CI = np.full((len(eta_list), len(list(randomizationlist))), -np.inf)
    for i, eta in enumerate(eta_list):
        for j, r in enumerate(randomizationlist):
            f = (parent_dir + f"/Data_HPC/45/"
                 f"Transduction_CoherentInfo_EATele_Nt={Nt}_depth={depth}_eta={eta}_ns={n_s}_np={n_p}_N={N}_randomization={r}_ci_list.npy")
            try:
                ci_history = np.load(f)
                CI[i, j] = np.max(ci_history)
            except OSError:
                pass
    return np.argmax(CI, axis=1)

def load_rhos_45(eta, r, depth=20, N=10000):
    base = (parent_dir + f"/Data_HPC/45/"
            f"Transduction_CoherentInfo_EATele_Nt={Nt}_depth={depth}_eta={eta}_ns={n_s}_np={n_p}_N={N}_randomization={r}")

    state_RS = qt.Qobj(np.load(base + "_state_RS.npy"), dims=[[Nt, Nt], [1, 1]])
    rho_S = qt.ptrace(state_RS, 1)

    state_PA = qt.Qobj(np.load(base + "_state_PA.npy"), dims=[[Nt, Nt], [1, 1]])
    rho_P = qt.ptrace(state_PA, 0)
    rho_A = qt.ptrace(state_PA, 1)

    return {"S": rho_S, "P": rho_P, "A": rho_A}

state_label = {"S": r"$\rho_S$", "P": r"$\rho_P$", "A": r"$\rho_A$"}

def draw_wigner(ax, W, plabel=True, qlabel=True):
    ax.contourf(X, Y, W, levels=levels, cmap=cmap, norm=norm)

    if plabel:
        ax.set_ylabel("p", fontsize=fs+8)
    else:
        ax.set_yticks([])

    if qlabel:
        ax.set_xlabel("q", fontsize=fs+8)


    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=fs)

def plot_panel(ax, eta, mode, plabel=True, qlabel=True, show_info=True):
    r = idx_45[np.where(eta_list == eta)[0][0]]
    rho = load_rhos_45(eta, r)[mode]

    W, negativity, energy = wignermatrix(rho)

    if show_info:
        moments = moments_onemode(qt2torch(rho), Nt)
        moment_2nd = np.around(moments[1].detach().numpy(), 3)
        S = qt.entropy_vn(rho)
        print("eta", eta, "mode", mode, "S", S, "g(np)", g(n_p), "2nd", moment_2nd)

    ax.text(-5, 4, state_label[mode], fontsize=fs)
    ax.text(-1, 4, r"$\eta$" + f"={eta}", fontsize=fs)
    draw_wigner(ax, W, plabel=plabel, qlabel=qlabel)

# -------------------- compute best randomizations --------------------
idx_45 = best_randomization_indices_45(eta_list)

# -------------------- layout: 2x(3 panels + 1 cbar col) --------------------
fig = plt.figure(figsize=(9, 6), layout="constrained")
gs = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 0.12], wspace=0.05, hspace=0.02)

axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
cax = fig.add_subplot(gs[:, 3])
ticks = [-0.1, -0.05, 0.0, 0.05, 0.10, 0.15,0.2,0.25,0.3]
# -------------------- plotting --------------------
# Row 0: S (no x-labels)
plot_panel(axes[0, 0], 0.2, "S", plabel=True,  qlabel=False)
plot_panel(axes[0, 1], 0.5, "S", plabel=False, qlabel=False)
plot_panel(axes[0, 2], 0.8, "S", plabel=False, qlabel=False)

cb0 = fig.colorbar(sm, cax=cax, boundaries=levels, ticks=np.arange(vmin, vmax+0.01, 0.06))
cb0.ax.tick_params(labelsize=fs)

# Row 1: P (with x-labels)
plot_panel(axes[1, 0], 0.2, "P", plabel=True,  qlabel=False)
plot_panel(axes[1, 1], 0.5, "P", plabel=False, qlabel=True)
plot_panel(axes[1, 2], 0.8, "P", plabel=False, qlabel=False)

cb = fig.colorbar(sm, cax=cax, ticks=ticks)
cb.ax.tick_params(labelsize=fs)

for r in range(2):
    axes[r, 0].set_ylabel("")

fig.supylabel("p", x=0, y=0.6, fontsize=fs+8)

# (optional) ticks/labels on left side of the colorbar axis
# for cb in (cb0, cb1):
#     cb.ax.yaxis.set_ticks_position("left")
#     cb.ax.yaxis.set_label_position("left")

fig.savefig(parent_dir + "/Figs/wigner_function_adaptive_ECD_MM.pdf", dpi=500)
# plt.show()
