import torch
import qutip as qt
import numpy as np
import math
import string
from datetime import datetime


#######################################################################################
# Basic Quantum operators and functions
#######################################################################################


def qt2torch(qt_object, requires_grad=False):
    return torch.tensor(qt_object.full(), dtype=torch.complex128, requires_grad=requires_grad)


def torch2qt(torch_object):
    return qt.Qobj(torch_object.detach().numpy())


# Pauli operators in pytorch
paulio = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
paulix = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
pauliy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
pauliz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


# Conjugate Transpose
def dagger(A):
    return A.conj().t()


def von_neumann_entropy(rho: torch.Tensor,
                        base: float = 2.0,
                        eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the von Neumann entropy of a density matrix (or a batch of them).

    Parameters
    ----------
    rho  : torch.Tensor
        Shape (..., N, N).  Must be Hermitian and trace-1.
        Real or complex dtypes are fine.
    base : float, optional
        Logarithm base.  The default (2) returns entropy in bits.
    eps  : float, optional
        Adds a tiny offset so log(0) never occurs.

    Returns
    -------
    torch.Tensor
        Entropy with shape equal to `rho.shape[:-2]`.
    """
    # Eigenvalues of a Hermitian matrix are real; use eigvalsh for numerical stability
    evals = torch.linalg.eigvalsh(rho).real  # (..., N)

    # Clamp tiny negative numerical noise to zero
    probs = torch.clamp(evals, min=0.0) + eps  # avoid log(0)

    # Change log base if requested
    logp = torch.log(probs)
    if base != torch.e:
        logp = logp / torch.log(torch.tensor(base,
                                             dtype=logp.dtype,
                                             device=logp.device))

    return -(probs * logp).sum(dim=-1)


def state_fidelity(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the fidelity between two quantum states represented by torch tensors.

    Conventions:
      - Pure state: an N_t x 1 column vector.
      - Mixed state: an N_t x N_t density matrix.

    For two pure states |ψ⟩ and |φ⟩, the fidelity is given by:
          F = |⟨ψ|φ⟩|².

    For the general case (including mixed states), the Uhlmann fidelity is computed:
          F(ρ, σ) = (Tr[sqrt(sqrt(ρ) σ sqrt(ρ))])².

    All operations are performed using PyTorch.
    """

    def is_pure(state: torch.Tensor) -> bool:
        # A pure state is an N_t x 1 column vector.
        return state.ndim == 2 and state.shape[1] == 1

    def is_mixed(state: torch.Tensor) -> bool:
        # A mixed state is an N_t x N_t square matrix.
        return state.ndim == 2 and state.shape[0] == state.shape[1] and state.shape[1] != 1

    def to_density_matrix(state: torch.Tensor) -> torch.Tensor:
        # Convert a pure state (column vector) to a density matrix.
        if is_pure(state):
            return state @ state.conj().T
        elif is_mixed(state):
            return state
        else:
            raise ValueError("State must be either an N_t x 1 pure state vector or an N_t x N_t density matrix.")

    def matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
        # Compute the square root of a positive semidefinite Hermitian matrix using eigen-decomposition.
        eigenvalues, eigenvectors = torch.linalg.eigh(mat)
        # Clamp eigenvalues to ensure they are non-negative, then compute the square root.
        sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
        # Ensure the sqrt_eigenvalues have the same dtype as the eigenvectors.
        sqrt_eigenvalues = sqrt_eigenvalues.to(eigenvectors.dtype)
        ms = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.conj().T
        ms = ms.to(torch.complex128)
        return ms

    def matrix_sqrt_svd(A):
        # Perform Singular Value Decomposition
        U, S, Vh = torch.linalg.svd(A)

        # Take the square root of the singular values
        S_sqrt = torch.sqrt(S).to(A.dtype)

        # Reconstruct the matrix square root: A^(1/2) = U S^(1/2) Vh
        return U @ torch.diag(S_sqrt) @ Vh

    # For two pure states, use the simple inner product formula.
    if is_pure(state1) and is_pure(state2):
        inner_product = state1.conj().T @ state2
        return torch.abs(inner_product[0, 0]) ** 2

    # Convert states to density matrices.
    rho = to_density_matrix(state1)
    sigma = to_density_matrix(state2)

    # Compute Uhlmann fidelity.
    sqrt_rho = matrix_sqrt(rho)
    intermediate = sqrt_rho @ sigma @ sqrt_rho
    sqrt_intermediate = matrix_sqrt(intermediate)
    fid = torch.real(torch.trace(sqrt_intermediate)) ** 2
    return fid


def trace_distance(rho, sigma):
    """
    Calculate the trace distance between two quantum states.

    Args:
        rho (torch.Tensor): First density matrix (shape: [d, d]).
        sigma (torch.Tensor): Second density matrix (shape: [d, d]).

    Returns:
        torch.Tensor: The trace distance between rho and sigma.
    """
    if rho.ndim == 2 and rho.shape[1] == 1:
        rho = rho @ rho.conj().T  # (Nt, 1) @ (1, Nt)
    if sigma.ndim == 2 and sigma.shape[1] == 1:
        sigma = sigma @ sigma.conj().T

    delta = rho - sigma
    # Hermitian matrix: delta = delta†
    delta = 0.5 * (delta + delta.conj().T)

    # Compute eigenvalues of |delta|
    eigenvalues = torch.linalg.eigvalsh(delta @ delta).clamp(min=0).sqrt()

    return 0.5 * eigenvalues.sum().real


# Basic bosonic operators
def destroy(N):
    a = qt.destroy(N)
    return qt2torch(a)


def create(N):
    a_dag = qt.create(N)
    return qt2torch(a_dag)


def position(N):
    a, a_dag = qt.destroy(N), qt.create(N)
    q = (a + a_dag) / np.sqrt(2)
    return qt2torch(q)


def momentum(N):
    a, a_dag = qt.destroy(N), qt.create(N)
    p = 1j * (a_dag - a) / np.sqrt(2)
    return qt2torch(p)


def Q_displacement(q, Nt):
    """

    :param r:
    :return: exp(-1j* r* operator(p))
    """
    H = -1j * q * momentum(Nt)
    Dq = torch.matrix_exp(H)

    return Dq


def complex_displacement_operator(alpha_real, alpha_img, Nt):
    """
    :param alpha_real: the real part of alpha
    :param alpha_img: the imaginary part of alpha
    :param Nt: the truncation dimension
    :return: the displacement operatorD(alpha) representation in the number basis
    """

    x = position(Nt)
    p = momentum(Nt)

    eigx, dx = torch.linalg.eigh(x)  # diagonalizing x
    Dx = torch.mm(torch.mm(dx, torch.diag_embed(torch.exp(1j * np.sqrt(2) * alpha_img * eigx))),
                  dagger(dx))  # displacement in x

    eigp, dp = torch.linalg.eigh(p)  # diagonalizing p
    Dp = torch.mm(torch.mm(dp, torch.diag_embed(torch.exp(-1j * np.sqrt(2) * alpha_real * eigp))),
                  dagger(dp))  # displacement in p

    # D = torch.mm(Dx, Dp) * torch.exp(-1j * alpha_img * alpha_real)
    cmu = torch.diag(torch.mm(x, p) - torch.mm(p, x))
    # print("cmu:", cmu)
    D = torch.mm(Dx, Dp) @ torch.diag_embed(torch.exp(-alpha_real * alpha_img * cmu))
    return D


def squeezed_operator(r, theta, Nt):
    """

    :param r:
    :param theta:
    :return: the matrix representation of S(xi) = exp( (z^* a^2- z a^dagger^2 )/2 ), z = r exp(i theta)
    """
    xi = r * torch.exp(1j * theta)
    EM = (torch.conj(xi) * destroy(Nt) @ destroy(Nt) - xi * create(Nt) @ create(Nt)) / 2
    # eig, T = torch.linalg.eig(EM)  # diagonalizing Exponent matrix
    # print(np.around(dagger(T) @ EM @ T, 2))
    #eig = eig.to(torch.complex128)
    #S = T @ torch.diag_embed(torch.exp(eig)) @ dagger(T)
    S = torch.matrix_exp(EM)
    return S


def unitary_beam_splitter(theta, Nt):
    """

    :param theta:
    :param Nt: cutoff dimension
    :return:
    """
    a = destroy(Nt)
    adg = create(Nt)
    b = destroy(Nt)
    bdg = create(Nt)

    H = theta * (-torch.kron(a, bdg) + torch.kron(adg, b))  # the Exponent matrix
    """
    eig, T = torch.linalg.eigh(H)  # diagonalizing Exponent matrix
    # print(np.around(dagger(T) @ H @ T, 2))
    eig = eig.to(torch.complex128)
    U = T @ torch.diag_embed(torch.exp(-1j * eig)) @ dagger(T)
    """

    U2 = torch.matrix_exp(1 * H)
    return U2


def phase_rotation(theta, Nt):
    n = torch.arange(0, Nt)
    R = torch.diag_embed(torch.exp(-1j * theta * n))
    R = R.to(torch.complex128)
    return R


def kerr(theta, Nt):
    n = torch.arange(0, Nt)
    H = n * n
    K = torch.diag_embed(torch.exp(-1j * theta * H))
    K = K.to(torch.complex128)
    return K


def One_mode_Gaussian_Unitary(alpha, phi1, r, phi2, Nt):
    R2 = phase_rotation(phi2, Nt)
    R1 = phase_rotation(phi1, Nt)
    S = squeezed_operator(r, torch.tensor(0), Nt)
    D = complex_displacement_operator(torch.real(alpha), torch.imag(alpha), Nt)
    U = D @ R1 @ S @ R2
    return U


def SUM_gate(Nt, alpha=1):
    H = torch.kron(position(Nt), momentum(Nt))
    U = torch.matrix_exp(-1j * alpha * H)
    return U


def SUM_gate_var(Nt, alpha=1):
    H = torch.kron(momentum(Nt), position(Nt))
    U = torch.matrix_exp(1j * alpha * H)
    return U


def bosonic_controlled_displacement(Nt, alpha):
    Q = position(Nt)
    eigvals, eigvecs = torch.linalg.eigh(Q)
    CD = torch.zeros((Nt * Nt, Nt * Nt), dtype=torch.complex128)
    for i, q in enumerate(eigvals):
        state1 = eigvecs[:, i].reshape(-1, 1)
        O1 = state1 @ dagger(state1)
        O2 = Q_displacement(q * alpha, Nt)
        CD = CD + torch.kron(O1, O2)
    return CD


def two_mode_squeeze(r, Nt, phi=torch.tensor(0)):
    """
    Generates the two-mode squeezing operator S2(r) in the Fock basis.

    Parameters:
        r (float): Squeezing parameter.
        dim (int): Hilbert space truncation (Fock space dimension per mode).

    Returns:
        Qobj: The two-mode squeezing unitary operator.
    """
    # Create annihilation operators for each mode
    a1 = destroy(Nt)
    a2 = destroy(Nt)

    # Define the two-mode squeezing generator
    #H = r * (torch.kron(a1, a2) - torch.kron(dagger(a1), dagger(a2)))
    H = r * (torch.exp(-1j * phi) * torch.kron(a1, a2) - torch.exp(1j * phi) * torch.kron(dagger(a1), dagger(a2)))

    # Compute the unitary operator S2(r) = exp(S2_generator)
    S = torch.matrix_exp(H)

    return S


def bosonic_pure_loss_channel_sf(rho, eta):
    """
    Applies the bosonic loss channel to a density matrix rho using transmissivity eta.

    Args:
        rho (torch.Tensor): Input density matrix of shape (N, N), assumed to be in Fock basis.
        eta (float): Transmissivity (0 <= eta <= 1).

    Returns:
        torch.Tensor: Output density matrix after applying the bosonic loss channel.
    """
    N = rho.shape[0]
    output_rho = torch.zeros_like(rho, dtype=torch.cdouble)

    for n in range(N):
        for m in range(N):
            rho_nm = rho[n, m]
            if torch.abs(rho_nm) < 1e-12:
                continue  # Skip negligible entries

            min_l = min(n, m)
            for l in range(min_l + 1):
                # Compute the coefficient from the formula
                coeff = ((1 - eta) / eta) ** l
                coeff *= eta ** ((n + m) / 2)
                coeff /= math.factorial(l)
                coeff *= math.sqrt(
                    math.factorial(n) * math.factorial(m) /
                    (math.factorial(n - l) * math.factorial(m - l))
                )

                new_n = n - l
                new_m = m - l
                output_rho[new_n, new_m] += coeff * rho_nm

    return output_rho


def bosonic_pure_loss_channel_BSD(rho, eta):
    """
     beam splitter dilation
    :param rho:
    :param eta: eta = cos(theta) **2 # this is different from the transduction protocol
    :return:
    """
    Nt = rho.shape[0]
    theta = np.arccos(np.sqrt(eta))

    state_E = torch.zeros(Nt, 1)
    state_E[0, 0] = 1
    rho_E = state_E @ dagger(state_E)

    U_BS = unitary_beam_splitter(theta, Nt)
    rho_SE = torch.kron(rho, rho_E)
    rho_SE = U_BS @ rho_SE @ dagger(U_BS)
    rho_SE = rho_SE.reshape(Nt, Nt, Nt, Nt)
    rho_S = torch.einsum("abcb->ac", rho_SE)
    rho_E = torch.einsum("babc->ac", rho_SE)
    return rho_S


def bosonic_pure_loss_channel_kraus(rho, eta, cutoff=None):
    """
    Apply a bosonic pure-loss (attenuator) channel to a density matrix ρ.

    Parameters
    ----------
    rho    : torch.Tensor, shape (..., d, d), complex-dtype
             Input density matrix in the Fock basis.  Batch dims allowed.
    eta    : float in [0, 1] eta = cos(theta) **2
             Transmissivity of the channel.
    cutoff : int or None
             Dimension d of the truncated Fock space.  If None, infer from rho.

    Returns
    -------
    rho_out : torch.Tensor, same shape as `rho`
    """
    if cutoff is None:
        cutoff = rho.shape[-1]
    if rho.shape[-2:] != (cutoff, cutoff):
        raise ValueError("ρ must be square with dimension = cutoff")

    device, dtype = rho.device, rho.dtype
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

    # ── 1. Annihilation operator â  ────────────────────────────────────────────
    # â_{m,n} = √n · δ_{m,n-1}
    a = destroy(cutoff)

    # ── 2. η^{N/2} diagonal (N = â†â)  ─────────────────────────────────────────
    k = torch.arange(cutoff, device=device, dtype=real_dtype)
    D_eta = torch.diag((eta ** (k / 2)).to(dtype))  # η^{k/2} on the diag

    # ── 3. Kraus loop  K_n = √[(1-η)^n / n!] · η^{N/2} âⁿ  ─────────────────────
    eye = torch.eye(cutoff, dtype=dtype, device=device)
    a_power = eye.clone()  # will hold âⁿ
    rho_out = torch.zeros_like(rho)

    # keep factorial incrementally to avoid overflow
    fact = 1.0
    one_me = (1.0 - eta)

    for n in range(cutoff):
        if n > 0:
            fact *= n  # n!
            a_power = a @ a_power  # âⁿ = â · âⁿ⁻¹
        coef = math.sqrt(one_me ** n / fact)  # √[(1-η)^n / n!]
        K_n = coef * (D_eta @ a_power)  # matrix of shape (d,d)

        # broadcast matmul over any leading batch dims in ρ
        rho_out = rho_out + K_n @ rho @ K_n.conj().T

    return rho_out


# Calculate the average energy of a bosonic mode
def energy_n_M(state, Nt):
    if state.numel() == Nt:
        state = state.reshape(1, Nt)
        prob = torch.abs(state) ** 2
    elif state.numel() == Nt ** 2:
        rho = state.reshape(Nt, Nt)
        prob = torch.abs(torch.diag(rho))
    else:
        raise ValueError("size is not correct")
    n = torch.sum(torch.arange(0, Nt) * prob)
    return n


def moments_onemode(state, Nt):
    if state.numel() == Nt:
        state = state.reshape(Nt, 1)
        rho = state @ dagger(state)
    elif state.numel() == Nt ** 2:
        rho = state.reshape(Nt, Nt)
    else:
        raise ValueError("size is not correct")
    q = position(Nt)
    p = momentum(Nt)

    exp_q = torch.trace(rho @ q)
    exp_p = torch.trace(rho @ p)
    exp_q2 = torch.trace(rho @ q @ q)
    exp_p2 = torch.trace(rho @ p @ p)
    exp_qp = torch.trace(rho @ q @ p)
    exp_pq = torch.trace(rho @ p @ q)
    qp_sym = 0.5 * (exp_qp + exp_pq)  # symmetrized

    var_q = exp_q2 - exp_q * exp_q
    var_p = exp_p2 - exp_p * exp_p
    cov_qp = qp_sym - exp_q * exp_p

    V = torch.empty((2, 2), dtype=torch.float64)
    V[0, 0] = var_q.real
    V[1, 1] = var_p.real
    V[0, 1] = V[1, 0] = cov_qp.real

    return torch.tensor([exp_q, exp_p]), V


# Common one mode bosonic state
def number_state(n, Nt):
    state = torch.zeros((Nt, 1), dtype=torch.complex128)
    state[n, 0] = 1
    return state


def coherent_state(alpha, Nt):
    """

    :param alpha: not torch data
    :param Nt:
    :return:
    """
    state = qt.coherent(Nt, alpha)
    state = qt2torch(state)
    return state


def coherent_state_torch(alpha, N):
    """
    Generate a coherent state |alpha> as a PyTorch tensor.

    Parameters:
        N (int): Hilbert space dimension (Fock basis cutoff).
        alpha (complex or torch.complex): Coherent amplitude.

    Returns:
        torch.Tensor: Shape (N,), complex-valued normalized coherent state.
    """
    n = torch.arange(N, dtype=torch.float64)

    # Compute each coefficient
    coeffs = alpha ** n / torch.sqrt(torch.tensor([math.factorial(int(k)) for k in n], dtype=torch.float64))
    coeffs = coeffs.to(torch.complex128)

    # Apply normalization
    state = torch.exp(-0.5 * torch.abs(alpha) ** 2) * coeffs
    norm = torch.linalg.norm(state)
    state = state.reshape(N, 1)

    return state / norm  # ensure unit norm


def squeezing_vacuum_state(r, Nt):
    state = torch.zeros((Nt, 1))
    for n in range(int(Nt / 2)):
        state[2 * n, 0] = (-1) ** n * (1 / np.sqrt(np.cosh(r))) * (np.tanh(r) ** n) * math.sqrt(
            math.factorial(2 * n)) / (
                                  2 ** n * math.factorial(n))
    norm = torch.sqrt(torch.sum(torch.abs(state) ** 2))
    state = state / norm
    return state


def squeezed_vacuum_state_analytical(r, phi=0.0, Nt=30, dtype=torch.complex128):
    """
    Return a (Nt, 1) column vector |ψ_sq⟩ in the Fock basis.

    Parameters
    ----------
    r      : float                  # squeezing magnitude
    phi    : float, optional        # squeezing phase (radians). default 0
    Nt  : int,   optional        # Hilbert-space truncation (≥ 2)
    dtype  : torch.dtype, optional  # complex precision
    """
    # pre-compute factorials (real dtype suffices)

    # allocate state |ψ⟩
    psi = torch.zeros((Nt, 1), dtype=dtype)

    # common factors
    tanh_r = math.tanh(r)
    phase = torch.exp(1j * phi)
    norm = 1 / math.sqrt(math.cosh(r))

    for n in range(0, Nt // 2):
        k = 2 * n  # photon number
        coeff = ((-phase * tanh_r) ** n) * norm
        coeff *= np.sqrt(math.factorial(k)) / (2.0 ** n * math.factorial(n))
        psi[k, 0] = coeff

    # Optional: renormalise if Nt is small
    psi = psi / torch.linalg.vector_norm(psi)

    return psi


def squeezed_vacuum_state_numerical(r, phi, Nt):
    state = torch.zeros((Nt, 1), dtype=torch.complex128)
    state[0, 0] = 1
    state = squeezed_operator(r, phi, Nt) @ state
    return state


def tmsv_state_analytical(r: float,
                          Nt: int,
                          *,
                          dtype: torch.dtype = torch.complex128,
                          device=None) -> torch.Tensor:
    """
    Two-mode squeezed-vacuum |TMSV(r)> in the joint Fock basis, truncated to Nt.
    Returns a column vector of shape (Nt**2, 1).
    """
    if device is None:
        device = torch.device("cpu")

    # Squeezing parameter
    r_t = torch.tensor(r, dtype=torch.float64, device=device)
    lam = torch.tanh(r_t)  # λ = tanh r

    # --- 1) keep indices as integers ----------------------------------------
    n_int = torch.arange(Nt, dtype=torch.long, device=device)  # int64
    n_float = n_int.to(torch.float64)  # for λ^n

    # Amplitudes c_n = √(1−λ²) λⁿ
    c_n = torch.sqrt(1 - lam ** 2) * lam ** n_float

    # Allocate joint vector and fill the |n,n⟩ diagonal
    psi = torch.zeros(Nt * Nt, dtype=dtype, device=device)

    # --- 2) diag_idx is already integer-typed -------------------------------
    diag_idx = n_int * (Nt + 1)  # maps (n,n) → n*Nt + n
    psi[diag_idx] = c_n.to(dtype)

    return psi.unsqueeze(1)


def tmsv_state_numerical(r, phi, Nt):
    state = torch.zeros(Nt * Nt, dtype=torch.complex128)
    state[0, 0] = 1
    state = two_mode_squeeze(r, Nt, phi) @ state
    return state.reshape(-1, 1)


def cat_state(alpha, theta, Nt, torch_data=True):
    state0 = torch.zeros((Nt, 1), dtype=torch.complex128)
    state0[0, 0] = 1
    D1 = complex_displacement_operator(alpha, 0, Nt)
    D2 = complex_displacement_operator(-alpha, 0, Nt)

    state = (D1 @ state0 + torch.tensor(np.exp(1j * theta)) * D2 @ state0)
    norm = torch.sum(torch.abs(state) ** 2)
    state = state / torch.sqrt(norm)
    if torch_data:
        return state
    else:
        return state.detach().numpy()


def thermal_state(nbar, Nt):
    """Return the density matrix of a single bosonic mode
       in a thermal state, truncated to Fock dimension N_cut."""
    lam = nbar / (nbar + 1.0)
    diag = [(1 - lam) * lam ** n for n in range(Nt)]
    return torch.diag(torch.tensor(diag, dtype=torch.complex128))


def thermal_state_twomode_pure(nbar, Nt):
    """Return the density matrix of a single bosonic mode
       in a thermal state, truncated to Fock dimension N_cut."""
    lam = nbar / (nbar + 1.0)
    state = torch.zeros(Nt * Nt, 1, dtype=torch.complex128)
    for n in range(Nt):
        state = state + torch.sqrt((1 - lam) * lam ** n) * torch.kron(number_state(n, Nt), number_state(n, Nt))

    return state


##########################################
# Common Bosonic Error Correction Codes
##########################################
def generate_cat_code(alpha, Nt):
    alpha_p = qt.coherent(Nt, alpha)
    alpha_n = qt.coherent(Nt, -alpha)
    L0 = (alpha_p + alpha_n).unit()
    L1 = (alpha_p - alpha_n).unit()
    L0 = qt2torch(L0)
    L1 = qt2torch(L1)
    return L0, L1

from numpy.polynomial.hermite import hermval

def position_eigenstate_coeffs(x, N):
    """
    Compute coefficients <n|x> for the position eigenstate |x>
    expanded in the number basis up to N-1.

    Parameters
    ----------
    x : float
        Position value where the eigenstate is localized.
    N : int
        Truncation dimension (number of Fock states).

    Returns
    -------
    coeffs : np.ndarray (complex)
        Array of shape (N,), coefficients <n|x>.
    """
    coeffs = np.zeros(N, dtype=np.complex128)
    norm_prefactor = np.pi ** (-0.25) * np.exp(-x ** 2 / 2)

    for n in range(N):
        # Hermite polynomial H_n(x)
        Hn = hermval(x, [0] * n + [1])  # physicists' Hermite
        coeffs[n] = norm_prefactor * Hn / np.sqrt((2.0 ** n) * math.factorial(n))
    return coeffs


def position_eigenstate(x,Nt):
    state = torch.zeros((Nt,1),dtype=torch.complex128)
    coeffs = position_eigenstate_coeffs(x,Nt)
    for n in range(Nt):
        state = coeffs[n] * number_state(n, Nt) + state
    return state/torch.norm(state)

def GKP_state_approx(j, delta, kappa, d, Nt=50, slimit=5, type_torch=False):
    alpha = np.sqrt(2 * np.pi / d)
    state = torch.zeros((Nt, 1), dtype=torch.complex128)
    for s in torch.arange(-slimit, slimit, 1):
        state0 = torch.zeros((Nt, 1), dtype=torch.complex128)
        state0[0, 0] = 1
        state0 = (Q_displacement(alpha * (d * s + j), Nt) @
                  squeezed_operator(-torch.log((torch.abs(delta))), torch.tensor([0]), Nt) @ state0)

        state0 = torch.exp(-0.5 * (kappa * alpha * (d * s + j)) ** 2) * state0 * torch.exp(1j * np.pi * s)
        state = state + state0
    state = state / torch.norm(state)
    if type_torch:
        return state
    else:
        return state.detach().numpy()


def GKP_state_approx2(j, kappa, r, slimit, Nt):
    sv = squeezed_vacuum_state_numerical(r, torch.tensor(0), Nt)
    state = torch.zeros((Nt, 1), dtype=torch.complex128)
    for s in np.arange(-slimit, slimit):
        omega_s = np.exp(-2 * np.pi * (kappa ** 2) * (s ** 2))
        xs = (2 * s + j) * np.pi
        state = state + omega_s * complex_displacement_operator(xs / np.sqrt(2), 0, Nt) @ sv
    state = state / torch.norm(state)
    return state


def gkp_square_finite_energy_qudit(Nt, delta, kappa, d, j,
                                   grid_range: int = 5,
                                   sqrt_pi: float = np.sqrt(np.pi)):
    """
    Construct a single-mode finite-energy square-lattice GKP |j>_Z (qudit) in the Fock basis.

    Conventions:
      - Stabilizers are displacements by 2√π in q and p (same as qubit).
      - Logical Z-basis states |j>_Z are realized by shifting the lattice in q by (2√π/d)*j.
        (For d=2, this gives offsets 0 and √π as usual.)
      - 'delta' and 'kappa' control finite-energy smearing/envelope in q and p respectively.

    Args
    ----
    N_cut : Fock cutoff (Hilbert space dimension)
    delta : intrapeak width (q-smearing)
    kappa : envelope width (p-smearing)
    d     : qudit dimension (integer >= 1). d=1, j=0 gives the qunaught.
    j     : logical index in {0, 1, ..., d-1}
    grid_range : sum over lattice integers n,m in [-grid_range, ..., +grid_range]
    sqrt_pi : handle for √π (do not change unless you use a different convention)

    Returns
    -------
    |ψ⟩ as a normalized qt.Qobj ket (dim N_cut).
    """
    if not (isinstance(d, int) and d >= 1):
        raise ValueError("d must be an integer >= 1.")
    if not (isinstance(j, int) and 0 <= j <= d - 1):
        raise ValueError("j must be an integer in {0,1,...,d-1}.")

    # Z-basis logical offset along q:
    q_offset = (2.0 * sqrt_pi / d) * j
    # No offset along p for Z-basis |j>_Z:
    p_offset = 0.0

    psi = number_state(0, Nt) * 0.0

    # Lattice points: (q, p) = (2√π n + q_offset, 2√π m + p_offset)
    # Map (q,p) to α = (q + i p)/√2 for D(α)
    for n in range(-grid_range, grid_range + 1):
        q_n = 2.0 * sqrt_pi * n + q_offset
        for m in range(-grid_range, grid_range + 1):
            p_m = 2.0 * sqrt_pi * m + p_offset

            # Finite-energy weight (Gaussian envelope in both directions)
            w = torch.exp(-0.5 * ((q_n ** 2) / (2 * delta ** 2) + (p_m ** 2) / (2 * kappa ** 2)))
            #print(n, m, w)

            alpha = (q_n + 1j * p_m) / np.sqrt(2.0)
            #D = qt.displace(N_cut, alpha)
            D = complex_displacement_operator(q_n / np.sqrt(2.0), p_m / np.sqrt(2.0), Nt)

            # Optional: one could add a local intrapeak squeeze; we keep it simple:
            #psi += w * D * qt.basis(N_cut, 0)
            psi += w * D @ number_state(0, Nt)
    psi = psi / torch.norm(psi)

    return psi


def hex_gkp_qudit(Nt, delta, kappa, d, j,
                  r_hex=torch.tensor(0.274653),
                  phi_hex=torch.tensor(np.pi / 12),
                  grid_range: int = 5):
    """
    Build a finite-energy hexagonal-lattice GKP |j>_Z for arbitrary qudit dimension d.

    Examples:
      - Qubit |0>_Z: d=2, j=0
      - Qubit |1>_Z: d=2, j=1
      - Qunaught:    d=1, j=0
    """
    psi_sq = gkp_square_finite_energy_qudit(Nt, delta, kappa, d, j, grid_range)
    R = phase_rotation(phi_hex, Nt)
    S = squeezed_operator(r_hex, torch.tensor(0), Nt)
    psi_hex = R @ S @ psi_sq
    psi_hex = psi_hex / torch.norm(psi_hex)
    return psi_hex


def GKP_square_lattice_approximate(Nt, d, j,delta,grid_range):
    state = torch.zeros(Nt,1)
    alpha = np.sqrt(2*np.pi/d)
    for k in range(-grid_range,grid_range):
        state_qk =position_eigenstate((d*k+j)*alpha,Nt)
        state = state+ state_qk

    n_vec = torch.arange(Nt)
    weights = torch.exp(-n_vec * (delta ** 2))  # shape (Nt,)
    state = state * weights.unsqueeze(1)
    state = state/torch.norm(state)
    return state


def GKP_hex_lattice_approximate(Nt, d, j,delta,r_hex,phi1_hex,phi2_hex,grid_range):
    R1 = phase_rotation(phi1_hex, Nt)
    S = squeezed_operator(r_hex, torch.tensor(0), Nt)
    R2 = phase_rotation(phi2_hex, Nt)
    state = GKP_square_lattice_approximate(Nt, d, j,delta,grid_range)
    state_hex = R2 @ S @ R1 @ state
    return state_hex/torch.norm(state_hex)





def partial_measurement(psi: torch.Tensor,
                        observable: torch.Tensor,
                        target: str = "A",
                        atol: float = 1e-6):
    """
    Projectively measure `observable` on subsystem `target` ('A' or 'B')
    of a bipartite pure state |ψ⟩ ∈ 𝓗_A ⊗ 𝓗_B.

    Parameters
    ----------
    psi : torch.Tensor
        Pure-state wave-function.  Accepts shape (dA, dB) **or**
        flattened shape (dA * dB,).  Must be complex dtype.
    observable : torch.Tensor
        Hermitian operator acting on the measured subsystem
        (dA×dA if target='A', else dB×dB).  Complex or real.
    target : str
        Which subsystem to measure ('A' or 'B').
    atol : float
        Numerical threshold below which a probability is treated as 0.

    Returns
    -------
    outcomes : list of tuples
        Each tuple is (p_i, state_A_i, state_B_i) where
        * p_i           : float -- probability of outcome i
        * state_A_i     : length-dA vector (pure state of A after outcome i)
        * state_B_i     : length-dB vector (pure state of B after outcome i)

    Notes
    -----
    * Only **non-degenerate** projectors (rank-1 eigenstates) are handled.
    * The returned {|state_A_i⟩⊗|state_B_i⟩} are *normalized* and satisfy
      Σ_i p_i = 1 up to numerical error.
    """
    # --- reshape ψ to (dA, dB) --------------------------------------------
    if target.upper() == "A":
        dA = observable.shape[0]
        dB = psi.numel() // dA
    else:  # target == 'B'
        dB = observable.shape[0]
        dA = psi.numel() // dB
    psi = psi.reshape(dA, dB)

    # --- diagonalize the observable ---------------------------------------
    # eigh guarantees real eigenvalues for Hermitian matrices
    eigvals, eigvecs = torch.linalg.eigh(observable)

    outcomes = []
    for k in range(len(eigvals)):
        v = eigvecs[:, k]  # |v_k⟩  (already normalized)

        # projector P_k = |v_k⟩⟨v_k|
        if target.upper() == "A":
            # (P_k ⊗ I) |ψ⟩   ⇒ multiply on the first index
            psi_k = torch.einsum('a,ab->b', v.conj(), psi)
        else:  # target == 'B'
            # (I ⊗ P_k) |ψ⟩   ⇒ multiply on the second index
            psi_k = torch.einsum('b,ab->a', v.conj(), psi)

        # probability of this outcome
        psi_k = psi_k.reshape(-1, 1)
        p_k = torch.sum(psi_k.conj() * psi_k).real
        if p_k < atol:
            continue

        # normalise collapsed state
        psi_k /= torch.sqrt(p_k)

        if target.upper() == "A":
            state_A = v  # |v_k⟩
            # |state_B⟩ = ⟨v_k| ψ_k⟩
            state_B = psi_k
        else:
            state_B = v  # |v_k⟩
            # |state_A⟩ = ψ_k |v_k⟩*
            state_A = psi_k

        # final normalisation of the conditional states
        state_A /= torch.linalg.norm(state_A)
        state_B /= torch.linalg.norm(state_B)

        outcomes.append((p_k.item(), eigvals[k].item(), state_A.reshape(-1, 1), state_B.reshape(-1, 1)))

    return outcomes


def partial_trace_qt2torch(state, shape, sel):
    """

    :param state: the input state torch multi-partite pure state of shape [-1,1]
    :param shape:
    :param sel: the part that needs to be kept
    :return:
    """
    state = state.detach().numpy()
    dims = [shape, [1] * len(shape)]
    state = qt.Qobj(state, dims)
    rho_qt = qt.ptrace(state, sel)
    rho_torch = qt2torch(rho_qt, True)
    return rho_torch


def make_partial_trace_einsum_string(shape, keep):
    """
    Return an einsum string that contracts over all subsystems **not** in `keep`.

    Parameters
    ----------
    shape : list[int]
        Local dimensions of |ψ⟩; only its length matters here.
    keep  : Iterable[int]
        0-based indices of the subsystems you want to keep.

    Returns
    -------
    str
        An einsum signature like 'abc,abd->cd'
        (first term is ψ*, second is ψ, right-hand side is ρ_keep).
    """
    n = len(shape)
    keep = set(keep)

    # 52 single-character labels: a-z, A-Z
    letters = list(string.ascii_lowercase + string.ascii_uppercase)
    if 2 * n > len(letters):
        raise ValueError("Too many subsystems; need more than 52 labels.")

    # Labels for ψ* (conjugate)
    left = letters[:n]

    # Labels for ψ; share a label if traced out, new label if kept
    right = []
    out = []
    for i in range(n):
        if i in keep:
            new_label = letters[n + i]  # fresh label
            right.append(new_label)
            out.extend([left[i], new_label])  # ρ indices come in pairs
        else:
            right.append(left[i])  # same label → contracted
    out_l = []
    out_r = []
    for i in range(len(out)):
        if i % 2 == 0:
            out_l.append(out[i])
        else:
            out_r.append(out[i])
    out = out_l + out_r

    einsum_str = f"{''.join(left)},{''.join(right)}->{''.join(out)}"
    return einsum_str


def partial_trace_torch(state, shape, sel):
    """

    :param state: the input state torch multi-partite pure state of shape [-1,1]
    :param shape:list of shape
    :param sel: the part that needs to be kept
    :return:
    """
    state = state.reshape(*shape)
    string = make_partial_trace_einsum_string(shape, sel)
    rho = torch.einsum(string, state, torch.conj(state))
    return rho


