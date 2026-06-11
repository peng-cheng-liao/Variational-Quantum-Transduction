import os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


def hex_lattice_generators(r_hex, phi1_hex, phi2_hex, d, rot_sign=-1, squeeze_q_exp=-1):
    alpha = 1.0
    r = float(r_hex)
    phi1 = rot_sign * float(np.asarray(phi1_hex).item())
    phi2 = rot_sign * float(np.asarray(phi2_hex).item())

    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    eq = np.exp(squeeze_q_exp * r)
    ep = np.exp(-squeeze_q_exp * r)

    g1 = alpha * np.array([
        c2 * eq * c1 - s2 * ep * s1,
        s2 * eq * c1 + c2 * ep * s1
    ], dtype=float)

    g2 = alpha * np.array([
        -c2 * eq * s1 - s2 * ep * c1,
        -s2 * eq * s1 + c2 * ep * c1
    ], dtype=float)

    S = np.vstack((g1, g2))
    return g1, g2, S


def angle_between_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("Zero-length vector")
    cos_theta = np.dot(a, b) / (na * nb)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def load_best_for_eta(eta_str, mode="S", d1_max=10, r_max=50):
    f_matrix = np.zeros((d1_max, r_max), dtype=float)

    for d1 in range(1, d1_max + 1):
        for r in range(r_max):
            path = (parent_dir
                    + f"/Data_HPC/91/GKP_state_match_fidelity_mixed_eta={eta_str}_mode={mode}_d1={d1}_r={r}.npy")
            try:
                data = np.load(path)
                f_matrix[d1 - 1, r] = float(data[0])
            except Exception:
                f_matrix[d1 - 1, r] = 0.0

    f_matrix[np.isnan(f_matrix)] = 0.0
    d_idx, r_idx = np.unravel_index(np.argmax(f_matrix), f_matrix.shape)  # 0-based
    best_fid = float(np.max(f_matrix))

    best_path = (parent_dir
                 + f"/Data_HPC/91/GKP_state_match_fidelity_mixed_eta={eta_str}_mode={mode}_d1={d_idx + 1}_r={r_idx}.npy")
    x = np.load(best_path).tolist()

    g1, g2, _ = hex_lattice_generators(x[2], x[3], x[4], d_idx, rot_sign=-1, squeeze_q_exp=-1)
    theta12 = angle_between_deg(g1, g2)

    return {
        "eta": eta_str,
        "fid": best_fid,
        "d_max": d_idx + 1,  # 1-based d1
        "g1": g1,
        "g2": g2,
        "theta12": theta12,
    }


def plot_all_etas(mode="S", fs=25):
    eta_list = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    eta_strs = [f"{eta:.2f}" for eta in eta_list]
    results = [load_best_for_eta(es, mode=mode) for es in eta_strs]

    # shared limits across all subplots
    max_len_global = max(
        max(np.linalg.norm(res["g1"]), np.linalg.norm(res["g2"]))
        for res in results
    )
    L = 1.75* max_len_global if max_len_global > 0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes = axes.ravel()


    for ax, res in zip(axes, results):
        ax.tick_params(labelsize=fs)
        g1, g2 = res["g1"], res["g2"]
        eta_str = res["eta"]
        fid = res["fid"]
        d_max = res["d_max"]
        theta12 = res["theta12"]

        ax.quiver(0, 0, g1[0], g1[1], angles="xy", scale_units="xy", scale=1,width=0.01,)
        ax.quiver(0, 0, g2[0], g2[1], angles="xy", scale_units="xy", scale=1,width=0.01,)

        l = 1.15
        #ax.text(g1[0] * l, g1[1] * l, r"$g_{1}$", fontsize=11)
        #ax.text(g2[0] * l, g2[1] * l, r"$g_{2}$", fontsize=11)

        ax.set_title(fr"$\eta={eta_str}$",fontsize=fs)

        info_top = (
            f"fid = {fid:.2f}\n"
            f"d = {d_max}\n"
            f"∠ = {theta12:.1f}°"
        )
        ax.text(
            0.03, 0.97, info_top,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=fs
        )

        info_bottom = (
            f"$g_{{1}}$ = ({g1[0]:.1f}, {g1[1]:.1f})\n"
            f"$g_{{2}}$ = ({g2[0]:.1f}, {g2[1]:.1f})"
        )
        ax.text(
            0.03, 0.03, info_bottom,
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=fs
        )

        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, linewidth=1.5)
        ax.axvline(0, linewidth=1.5)
        ax.set_xlabel("q",fontsize=fs)
        ax.set_ylabel("p",fontsize=fs)

    fig.suptitle(f"Optimal lattice vectors, included angle, and coordinates (mode={mode})", y=1.02)
    plt.savefig(parent_dir + f"/Figs/GKP_state_match_mode={mode}.pdf", dpi=500)
    #plt.show()


if __name__ == "__main__":
    plot_all_etas(mode="P")