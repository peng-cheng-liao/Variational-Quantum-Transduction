import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from QTorch.Transduction import *

default_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'  # Cyan
]


def wignermatrix(rho, qplimit=8, bins=250):
    qlist = np.linspace(-qplimit, qplimit, bins)
    plist = np.linspace(-qplimit, qplimit, bins)
    interval = 2 * qplimit / bins
    square = interval ** 2
    Nt = rho.shape[0]

    rho = qt.Qobj(rho, dims=[[Nt], [Nt]])
    wigner_matrix = qt.wigner(rho, qlist, plist)
    wigner_matrix = wigner_matrix.reshape((bins, bins))
    norm = np.sum(wigner_matrix) * square
    norm_abs = np.sum(np.abs(wigner_matrix)) * square
    negativity = norm_abs - norm
    energy = np.sum(np.arange(Nt) * np.diag(rho.full()))
    print(
        "norm", norm,
        "negativity", negativity,
        "min", np.min(wigner_matrix),
        "energy", energy.real)

    return wigner_matrix, negativity, energy


def plot_qpcontour(wigner_matrix, fig, ax, plabel=True, fs=10, qplimit=8, bins=250,vmin=-0.1, vmax = 0.5,colorbar=True):
    qlist = np.linspace(-qplimit, qplimit, bins)
    plist = np.linspace(-qplimit, qplimit, bins)
    X, Y = np.meshgrid(qlist, plist)  # Create the grid of X and Y coordinates
    px = ax.contourf(X, Y, wigner_matrix, levels=np.arange(vmin, vmax, 0.01),cmap='RdBu_r') #
    if colorbar:
        fig.colorbar(px, shrink=0.72, )

    ax.set_xlabel('q', fontsize=fs + 4)
    if plabel:
        ax.set_ylabel('p', fontsize=fs + 4)

    ax.set_aspect("equal")
    ax.tick_params(axis='both', labelsize=fs)
    #plt.tight_layout()


"""
etalist = np.around(np.arange(0.1, 1.0, 0.1), 1)

alphalist1 = 1/np.sqrt(etalist)
alphalist12 = [-2.23382354254349, -1.7204215498599167, -1.5076123912113628, -1.2193293889568346, -1.076527721229661, -0.8143261641762596, -0.6523488852003506, -0.49825610542931753, -0.33222931172041703]

plt.plot(etalist,alphalist1)
plt.plot(etalist,np.abs(alphalist12))
plt.show()

"""
