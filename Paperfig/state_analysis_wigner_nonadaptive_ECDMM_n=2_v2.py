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


def hex_lattice_generators(r_hex, phi1_hex, phi2_hex, d, rot_sign=-1, squeeze_q_exp=-1):
    """
    rot_sign = +1 means R(phi)=[[cos,-sin],[sin,cos]].
    rot_sign = -1 flips angle: R(-phi).

    squeeze_q_exp = -1 means q->e^{-r} q, p->e^{+r} p (S=diag(e^{-r},e^{+r}))
    squeeze_q_exp = +1 means q->e^{+r} q, p->e^{-r} p (S=diag(e^{+r},e^{-r}))
    """
    alpha = np.sqrt(np.pi * d / 2)
    r = float(r_hex)
    phi1 = rot_sign * float(np.asarray(phi1_hex).item())
    phi2 = rot_sign * float(np.asarray(phi2_hex).item())

    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    eq = np.exp(squeeze_q_exp * r)
    ep = np.exp(-squeeze_q_exp * r)  # opposite exponent for p

    # M = R(phi2) S(r) R(phi1)
    # columns are images of e1,e2
    g1 = alpha * np.array([
        c2 * eq * c1 - s2 * ep * s1,
        s2 * eq * c1 + c2 * ep * s1
    ], dtype=float)

    g2 = alpha * np.array([
        -c2 * eq * s1 - s2 * ep * c1,
        -s2 * eq * s1 + c2 * ep * c1
    ], dtype=float)

    return g1, g2




def load_rhos(eta):
    state_RS = np.load(parent_dir + f"/Data_HPC/84/best_feasible_state/state_RS_best_feasible_eta={eta}.npy")
    state_RS = qt.Qobj(state_RS, dims=[[Nt, Nt], [1, 1]])
    rho_S = qt.ptrace(state_RS, 1)

    state_PA = np.load(parent_dir + f"/Data_HPC/84/best_feasible_state/state_PA_best_feasible_eta={eta}.npy")
    state_PA = qt.Qobj(state_PA, dims=[[Nt, Nt], [1, 1]])
    rho_P = qt.ptrace(state_PA, 0)
    rho_A = qt.ptrace(state_PA, 1)

    return {"S": rho_S, "P": rho_P, "A": rho_A}


def draw_wigner(ax, wigner_matrix, plabel=True, qlabel=True,
                v1=None, v2=None, O=(0.0, 0.0),
                quiver_kwargs=None, text_kwargs=None):
    """
    Draw Wigner contour and optionally plot vectors v1, v2 from origin O in (q,p).

    Notes:
      - Assumes X, Y, levels, cmap, norm, fs exist in the caller scope (as in your original).
      - v1, v2 should be (dq, dp) in the same units as X,Y axes.
    """
    m = ax.contourf(X, Y, wigner_matrix, levels=levels, cmap=cmap, norm=norm)

    if not plabel:
        ax.set_yticks([])

    if qlabel:
        ax.set_xlabel("q", fontsize=fs + 6)

    ax.set_aspect("equal")

    # ---- draw vectors ----
    q0, p0 = float(O[0]), float(O[1])

    quiver_kwargs = {} if quiver_kwargs is None else dict(quiver_kwargs)
    quiver_kwargs.setdefault("angles", "xy")
    quiver_kwargs.setdefault("scale_units", "xy")
    quiver_kwargs.setdefault("scale", 1.0)
    quiver_kwargs.setdefault("width", 0.004)
    quiver_kwargs.setdefault("color", "k")

    text_kwargs = {} if text_kwargs is None else dict(text_kwargs)
    text_kwargs.setdefault("ha", "left")
    text_kwargs.setdefault("va", "bottom")
    text_kwargs.setdefault("fontsize", fs)

    def _draw(vec, label):
        if vec is None:
            return
        vec = np.asarray(vec).reshape(-1)
        if vec.size != 2:
            raise ValueError(f"{label} must be length-2 (dq,dp); got shape {np.asarray(vec).shape}")
        dq, dp = float(vec[0]), float(vec[1])
        ax.quiver([q0], [p0], [dq], [dp], **quiver_kwargs)
        #ax.text(q0 + dq, p0 + dp, label, **text_kwargs)

    _draw(v1, "v1")
    _draw(v2, "v2")

    # mark origin
    ax.plot([q0], [p0], marker="o", markersize=3, color=quiver_kwargs.get("color", "k"))

    return m


state_label = {"S": r"$\rho_S$", "P": r"$\rho_P$", "A": r"$\rho_A$"}


def plot_panel(ax, eta, mode, plabel=True, qlabel=True, show_info=False, plot_lattice=False, O=(0, 0)):
    rho = load_rhos(eta)[mode]
    wigner_matrix, negativity, energy = wignermatrix(rho, qplimit=qplimit, bins=bins)
    print(eta, np.max(wigner_matrix), np.min(wigner_matrix))

    if show_info:
        moments = moments_onemode(qt2torch(rho), Nt)
        moment_2nd = np.around(moments[1].detach().numpy(), 3)
        S = qt.entropy_vn(rho)
        print(eta, mode, "S=", S, "g(np)=", g(n_p), "2nd=", moment_2nd)
    if plot_lattice:
        lattice_parametes = np.load(
            parent_dir + f"/Data_HPC/91_pp/GKP_state_match_fidelity_mixed_eta={eta}_mode={mode}_optimal_lattice_parameters.npy")
        d, r_hex, phi1_hex, phi2_hex = lattice_parametes
        g1, g2 = hex_lattice_generators(r_hex, phi1_hex, phi2_hex, d)
    else:
        g1 = None
        g2 = None

    ax.text(-4, 4, state_label[mode], fontsize=fs)
    ax.text(0, 4, r"$\eta$" + f"={float(eta)}", fontsize=fs)
    return draw_wigner(ax, wigner_matrix, plabel=plabel, qlabel=qlabel, v1=g1, v2=g2,O=O)


# -------------------- layout + plotting --------------------
fig = plt.figure(figsize=(9, 6), layout="constrained")
gs = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 0.12], wspace=0.05, hspace=0.02)

axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])

# one colorbar axis spanning both rows
cax = fig.add_subplot(gs[:, 3])

ticks = [-0.1, -0.05, 0.0, 0.05, 0.10, 0.15]

# Row 0: mode S
plot_panel(axes[0, 0], "0.20", "S", plabel=True, qlabel=False,plot_lattice=True)
plot_panel(axes[0, 1], "0.40", "S", plabel=False, qlabel=False,plot_lattice=True)
plot_panel(axes[0, 2], "0.60", "S", plabel=False, qlabel=False,plot_lattice=False)

# Row 1: mode P
plot_panel(axes[1, 0], "0.20", "P", plabel=True, qlabel=False,plot_lattice=True)
plot_panel(axes[1, 1], "0.40", "P", plabel=False, qlabel=True,plot_lattice=True)
plot_panel(axes[1, 2], "0.60", "P", plabel=False, qlabel=False,plot_lattice=False)

# single shared colorbar for both rows
cb = fig.colorbar(sm, cax=cax, )  #ticks=ticks
cb.ax.tick_params(labelsize=fs)

# ---------- ONE shared "p" label between the two rows ----------
# Remove per-axes y-labels (if any were set elsewhere)
for r in range(2):
    axes[r, 0].set_ylabel("")

# Add a single figure-level y label, vertically centered between rows
fig.supylabel("p", x=0, y=0.55, fontsize=fs + 6)

#plt.savefig(parent_dir + "/Figs/wigner_function_nonadaptive_ECD_MM_n=2.pdf", dpi=500)
plt.show()
