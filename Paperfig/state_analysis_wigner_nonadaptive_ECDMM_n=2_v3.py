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
qlist = np.linspace(-qplimit, qplimit, bins)
plist = np.linspace(-qplimit, qplimit, bins)
X, Y = np.meshgrid(qlist, plist)

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


def brightest_peak_and_nearest_vectors(
        W: np.ndarray,
        qlist: np.ndarray,
        plist: np.ndarray,
        *,
        neighborhood: int = 5,
        min_separation: float | None = None,
        rel_threshold: float = 0.5,
):
    """
    Given a Wigner matrix W on a (q,p) grid, return:
      O  : (q0, p0) coordinate of the brightest peak in |W|
      v1 : vector (dq, dp) from O to the nearest other peak in |W|
      v2 : vector (dq, dp) from O to the second-nearest other peak in |W|

    Parameters
    ----------
    W : (len(plist), len(qlist)) or (len(qlist), len(plist)) ndarray
        Wigner values on the grid. This function assumes W[p_index, q_index].
        If your W is W[q_index, p_index], set W = W.T before calling.
    qlist, plist : 1D ndarrays
        Grid coordinates (monotone increasing).
    neighborhood : int
        Local-maximum check radius (in grid points). Larger is stricter.
    min_separation : float or None
        Minimum Euclidean separation (in coordinate units) from O to consider
        a distinct peak. If None, uses ~1.5 * min(grid spacings).
    rel_threshold : float
        Only consider candidate peaks with |W| >= rel_threshold * max(|W|).

    Returns
    -------
    O : np.ndarray shape (2,)
        [q0, p0]
    v1, v2 : np.ndarray shape (2,)
        [dq, dp] vectors to the nearest two distinct peaks.

    Notes
    -----
    - Peaks are detected as strict local maxima of |W| within a square window
      of size (2*neighborhood+1)^2.
    - If fewer than 3 peaks are found (including O), raises ValueError.
    """
    W = np.asarray(W)
    qlist = np.asarray(qlist)
    plist = np.asarray(plist)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")
    if qlist.ndim != 1 or plist.ndim != 1:
        raise ValueError("qlist and plist must be 1D arrays.")

    # Expect W[p, q]. Check shape.

    if W.shape != (len(plist), len(qlist)):
        raise ValueError(
            f"Shape mismatch: W.shape={W.shape}, expected ({len(plist)}, {len(qlist)})"
        )

    A = np.abs(W)
    maxA = float(A.max())
    if maxA == 0.0:
        raise ValueError("All entries of W are zero; no peaks to find.")

    # Brightest peak (absolute value)
    p0_idx, q0_idx = np.unravel_index(np.argmax(A), A.shape)
    q0 = float(qlist[q0_idx])
    p0 = float(plist[p0_idx])
    O = np.array([q0, p0], dtype=float)

    # Default min separation in coordinate units
    dq_min = np.min(np.diff(qlist)) if len(qlist) > 1 else 0.0
    dp_min = np.min(np.diff(plist)) if len(plist) > 1 else 0.0
    if min_separation is None:
        base = min(x for x in [dq_min, dp_min] if x > 0) if (dq_min > 0 or dp_min > 0) else 0.0
        min_separation = 1.5 * base if base > 0 else 0.0

    # Candidate peaks: local maxima of A above threshold
    thr = rel_threshold * maxA
    peaks = []
    nh = int(neighborhood)

    # Avoid edges where window would go out-of-bounds
    p_lo, p_hi = nh, A.shape[0] - nh
    q_lo, q_hi = nh, A.shape[1] - nh

    for pi in range(p_lo, p_hi):
        for qi in range(q_lo, q_hi):
            val = A[pi, qi]
            if val < thr:
                continue
            window = A[pi - nh: pi + nh + 1, qi - nh: qi + nh + 1]
            # strict maximum (unique) inside window
            if val == window.max() and np.count_nonzero(window == val) == 1:
                peaks.append((pi, qi, float(val)))

    # Ensure O is included as a peak even if it sits near boundary
    if not any((pi == p0_idx and qi == q0_idx) for (pi, qi, _) in peaks):
        peaks.append((p0_idx, q0_idx, float(A[p0_idx, q0_idx])))

    # Convert to coordinates and compute distances from O
    cand = []
    for pi, qi, val in peaks:
        q = float(qlist[qi])
        p = float(plist[pi])
        d = np.hypot(q - q0, p - p0)
        if d == 0:
            continue
        if d < float(min_separation):
            continue
        cand.append((d, np.array([q - q0, p - p0], dtype=float), (pi, qi), val))

    if len(cand) < 2:
        raise ValueError(
            f"Found {len(cand)} distinct peak(s) besides O. "
            "Try decreasing neighborhood, decreasing min_separation, or lowering rel_threshold."
        )

    cand.sort(key=lambda x: x[0])
    v1 = cand[0][1]
    v2 = cand[1][1]
    return O, v1, v2


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
        print(wigner_matrix.shape, len(qlist), len(plist))
        if mode == "S":
            O, g1, g2 = brightest_peak_and_nearest_vectors(-wigner_matrix, qlist, plist)
        else:
            O, g1, g2 = brightest_peak_and_nearest_vectors(wigner_matrix, qlist, plist)


    else:
        g1 = None
        g2 = None

    ax.text(-4, 4, state_label[mode], fontsize=fs)
    ax.text(0, 4, r"$\eta$" + f"={float(eta)}", fontsize=fs)
    return draw_wigner(ax, wigner_matrix, plabel=plabel, qlabel=qlabel, v1=g1, v2=g2, O=O)


# -------------------- layout + plotting --------------------
fig = plt.figure(figsize=(9, 6), layout="constrained")
gs = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 0.12], wspace=0.05, hspace=0.02)

axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])

# one colorbar axis spanning both rows
cax = fig.add_subplot(gs[:, 3])

ticks = [-0.1, -0.05, 0.0, 0.05, 0.10, 0.15]

# Row 0: mode S
plot_panel(axes[0, 0], "0.20", "S", plabel=True, qlabel=False, plot_lattice=True)
plot_panel(axes[0, 1], "0.40", "S", plabel=False, qlabel=False, plot_lattice=True)
plot_panel(axes[0, 2], "0.60", "S", plabel=False, qlabel=False, plot_lattice=False)

# Row 1: mode P
plot_panel(axes[1, 0], "0.20", "P", plabel=True, qlabel=False, plot_lattice=True)
plot_panel(axes[1, 1], "0.40", "P", plabel=False, qlabel=True, plot_lattice=True)
plot_panel(axes[1, 2], "0.60", "P", plabel=False, qlabel=False, plot_lattice=False)

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
