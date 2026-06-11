import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

import qutip as qt
from Quantum_Plotting import wignermatrix, moments_onemode, qt2torch  # adjust if needed

rc("text", usetex=True)

# -------------------- config --------------------
Nt = 30
n_s = 2
n_p = 2

fs = 25
qplimit = 6
bins = 250
vmin, vmax = -0.10, 0.30
levels = np.arange(vmin, vmax, 0.01)
cmap = "RdBu_r"

plt.rcParams.update({
    "font.size": fs,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
})

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

# precompute grid once
q = np.linspace(-qplimit, qplimit, bins)
p = np.linspace(-qplimit, qplimit, bins)
X, Y = np.meshgrid(q, p)

# shared norm for consistent color scale
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])


def load_rhos(eta):
    state_RS = np.load(parent_dir + f"/Data_HPC/84/best_feasible_state/state_RS_best_feasible_eta={eta}.npy")
    state_RS = qt.Qobj(state_RS, dims=[[Nt, Nt], [1, 1]])
    rho_S = qt.ptrace(state_RS, 1)

    state_PA = np.load(parent_dir + f"/Data_HPC/84/best_feasible_state/state_PA_best_feasible_eta={eta}.npy")
    state_PA = qt.Qobj(state_PA, dims=[[Nt, Nt], [1, 1]])
    rho_P = qt.ptrace(state_PA, 0)
    rho_A = qt.ptrace(state_PA, 1)

    return {"S": rho_S, "P": rho_P, "A": rho_A}

def draw_wigner(ax, wigner_matrix, plabel=True, qlabel=True):
    m = ax.contourf(X, Y, wigner_matrix, levels=levels, cmap=cmap, norm=norm)

    # We'll ONLY use tick labels on left column; "p" label will be added globally later.
    if plabel:
        # keep y tick labels
        pass
    else:
        ax.set_yticks([])

    if qlabel:
        ax.set_xlabel("q",fontsize = fs+6)

    ax.set_aspect("equal")
    return m



state_label = {"S": r"$\rho_S$", "P": r"$\rho_P$", "A": r"$\rho_A$"}

def plot_panel(ax, eta, mode, plabel=True, qlabel=True, show_info=False):

    rho = load_rhos(eta)[mode]
    wigner_matrix, negativity, energy = wignermatrix(rho,qplimit=qplimit,bins=bins)
    print(eta, np.max(wigner_matrix), np.min(wigner_matrix))

    if show_info:
        moments = moments_onemode(qt2torch(rho), Nt)
        moment_2nd = np.around(moments[1].detach().numpy(), 3)
        S = qt.entropy_vn(rho)
        print( eta, mode, "S=", S, "g(np)=", g(n_p), "2nd=", moment_2nd)

    ax.text(-4, 4, state_label[mode], fontsize=fs)
    ax.text(0, 4, r"$\eta$" + f"={float(eta)}", fontsize=fs)
    return draw_wigner(ax, wigner_matrix, plabel=plabel, qlabel=qlabel)

# -------------------- layout + plotting --------------------
fig = plt.figure(figsize=(9, 6), layout="constrained")
gs = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 0.12], wspace=0.05, hspace=0.02)

axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])

# one colorbar axis spanning both rows
cax = fig.add_subplot(gs[:, 3])

ticks = [-0.1, -0.05, 0.0, 0.05, 0.10, 0.15]

# Row 0: mode S
plot_panel(axes[0, 0], "0.20", "S", plabel=True,  qlabel=False)
plot_panel(axes[0, 1], "0.40", "S", plabel=False, qlabel=False)
plot_panel(axes[0, 2], "0.60", "S", plabel=False, qlabel=False)

# Row 1: mode P
plot_panel(axes[1, 0], "0.20", "P", plabel=True,  qlabel=False)
plot_panel(axes[1, 1], "0.40", "P", plabel=False, qlabel=True)
plot_panel(axes[1, 2], "0.60", "P", plabel=False, qlabel=False)

# single shared colorbar for both rows
cb = fig.colorbar(sm, cax=cax, ) #ticks=ticks
cb.ax.tick_params(labelsize=fs)

# ---------- ONE shared "p" label between the two rows ----------
# Remove per-axes y-labels (if any were set elsewhere)
for r in range(2):
    axes[r, 0].set_ylabel("")

# Add a single figure-level y label, vertically centered between rows
fig.supylabel("p", x=0, y=0.55, fontsize=fs+6)

#plt.savefig(parent_dir + "/Figs/wigner_function_nonadaptive_ECD_MM_n=2.pdf", dpi=500)
plt.show()
