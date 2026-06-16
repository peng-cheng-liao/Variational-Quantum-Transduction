import torch

from .Basic import *


def ECD_unitary(alpha1, alpha2, psi, theta, Nt):
    D = complex_displacement_operator(alpha1, alpha2, Nt)
    ul = dagger(D) * torch.sin(theta / 2) * torch.exp(1j * (psi - torch.tensor(np.pi / 2)))
    ur = dagger(D) * torch.cos(theta / 2)
    ll = D * torch.cos(theta / 2)
    lr = - D * torch.sin(theta / 2) * torch.exp(-1j * (psi - torch.tensor(np.pi / 2)))
    ECD = torch.cat([torch.cat([ul, ur], 1), torch.cat([ll, lr], 1)], 0)
    return ECD


# ECD gate acting on one qubit and one mode with multiples layers
def ECD_unitary_multi_layers(alphas, betas, Nt, depth):
    Uecd = torch.eye(2 * Nt, dtype=torch.complex128)

    for i in range(depth):
        alpha1 = alphas[2 * i]
        alpha2 = alphas[1 + 2 * i]
        psi = betas[2 * i]
        theta = betas[1 + 2 * i]
        Uecd = torch.mm(ECD_unitary(alpha1, alpha2, psi, theta, Nt), Uecd)
    # Uecd = torch.mm(ECD_unitary(torch.tensor(0), torch.tensor(0), betas[-1], betas[-2], Nt), Uecd)
    # Uecd = torch.kron(torch.eye(2), displacement_operator(alphas[-1], alphas[-2], Nt)) @ Uecd
    return Uecd


# ECD gate acting on one qubit and two mode state
def ECD_unitary_QMM(alpha1, alpha2, psi, theta, Nt, target, state):
    # Compute the ECD gate and reshape it to (2, Nt, 2, Nt)
    ECD_QM = ECD_unitary(alpha1, alpha2, psi, theta, Nt)
    ECD_QM = ECD_QM.reshape(2, Nt, 2, Nt)

    # Detect if state is a pure state vector (assumed shape: (d, 1))
    if state.dim() == 2 and state.shape[1] == 1:
        # Reshape pure state into (qubit, boson0, boson1, 1)
        state = state.reshape(2, Nt, Nt, 1)
        if target == 0:
            # Apply gate on boson0 (2nd index)
            state_post = torch.einsum("iajb,jbcs->iacs", ECD_QM, state)
        elif target == 1:
            # Apply gate on boson1 (3rd index)
            state_post = torch.einsum("iajb,jcbs->icas", ECD_QM, state)
        # Reshape back to pure state vector
        state_post = state_post.reshape(-1, 1)

    else:
        # Assume state is a density matrix.
        # Reshape state to have indices:
        # (qubit_ket, boson0_ket, boson1_ket, qubit_bra, boson0_bra, boson1_bra)
        state = state.reshape(2, Nt, Nt, 2, Nt, Nt)
        ECD_QM_adjoint = ECD_QM.conj().permute(2, 3, 0, 1)
        if target == 0:
            # For target 0, the gate acts on boson0 indices.
            # Transformation: ρ'_{i,a,c,j,b,d} = sum_{k,e,l,f} U_{i,a,k,e} ρ_{k,e,c,l,f,d} U†_{l,f,j,b}
            state_post = torch.einsum("iake,kec lfd,lfjb->iacjbd", ECD_QM, state, ECD_QM_adjoint)
        elif target == 1:
            # For target 1, the gate acts on boson1 indices.
            # Transformation: ρ'_{i,a,c,j,b,d} = sum_{k,e,l,f} U_{i,c,k,e} ρ_{k,a,e,l,b,f,d} U†_{l,f,j,d}
            state_post = torch.einsum("icke,kaelbf,lfjd->iacjbd", ECD_QM, state, ECD_QM_adjoint)
        # Reshape the output back into a square matrix form
        #state_post = state_post.reshape(2 * Nt * Nt, 2 * Nt * Nt)

    return state_post


# ECD gate acting on one qubit and two mode state and extra system
def ECD_unitary_QMM_plus(alpha1, alpha2, psi, theta, Nt, target, state):
    # Compute the ECD gate and reshape it to (2, Nt, 2, Nt)
    ECD_QM = ECD_unitary(alpha1, alpha2, psi, theta, Nt)
    ECD_QM = ECD_QM.reshape(2, Nt, 2, Nt)

    state = state.reshape(2, Nt, Nt, -1)
    if target == 0:
        # Apply gate on boson0 (2nd index)
        state_post = torch.einsum("iajb,jbcs->iacs", ECD_QM, state)
    elif target == 1:
        # Apply gate on boson1 (3rd index)
        state_post = torch.einsum("iajb,jcbs->icas", ECD_QM, state)
    # Reshape back to pure state vector
    state_post = state_post.reshape(-1, 1)

    return state_post


# ECD gate acting on one qubit and three mode state
def ECD_unitary_QMMM(alpha1, alpha2, psi, theta, Nt, target, state):
    ECD_QM = ECD_unitary(alpha1, alpha2, psi, theta, Nt)
    ECD_QM = ECD_QM.reshape(2, Nt, 2, Nt)
    state = state.reshape(2, Nt, Nt, Nt)
    if target == 0:
        state_post = torch.einsum("iajb,jbcd->iacd", ECD_QM, state)
    elif target == 1:
        state_post = torch.einsum("iajc,jbcd->ibad", ECD_QM, state)
    else:
        state_post = torch.einsum("iajd,jbcd->ibca", ECD_QM, state)
    state_post = state_post.reshape(-1, 1)
    return state_post



def energy_n_QM(state, Nt):
    state = state.reshape(2, Nt)
    prob = torch.abs(state) ** 2
    plist = torch.sum(prob, dim=0)
    n = torch.sum(torch.arange(0, Nt) * plist)
    return n


def p1p2_MM(state, Nt):
    state = state.reshape(Nt, Nt)
    plist = torch.abs(state) ** 2
    p1list = torch.sum(plist, dim=1)
    p2list = torch.sum(plist, dim=0)
    return p1list, p2list


def energy_n1n2_MM(state, Nt):
    p1mode, p2mode = p1p2_MM(state, Nt)
    n1 = torch.sum(torch.arange(0, Nt) * p1mode)
    n2 = torch.sum(torch.arange(0, Nt) * p2mode)
    return n1, n2


def p1p2_QMM(state, Nt):
    state = state.reshape(2, Nt, Nt)
    prob = torch.abs(state) ** 2
    plist = torch.sum(prob, dim=0)
    p1list = torch.sum(plist, dim=1)
    p2list = torch.sum(plist, dim=0)
    return p1list, p2list


def energy_n1n2_QMM(state, Nt):
    p1mode, p2mode = p1p2_QMM(state, Nt)
    n1 = torch.sum(torch.arange(0, Nt) * p1mode)
    n2 = torch.sum(torch.arange(0, Nt) * p2mode)
    return n1, n2


def p1p2p3_QMMM(state, Nt):
    state = state.reshape(2, Nt, Nt, Nt)
    prob = torch.abs(state) ** 2
    plist = torch.sum(prob, dim=0)
    p1list = torch.sum(plist, dim=(1, 2))
    p2list = torch.sum(plist, dim=(0, 2))
    p3list = torch.sum(plist, dim=(0, 1))
    return p1list, p2list, p3list


def energy_n1n2n3_QMMM(state, Nt):
    p1in, p2in, p3in = p1p2p3_QMMM(state, Nt)
    n1 = torch.sum(torch.arange(0, Nt) * p1in)
    n2 = torch.sum(torch.arange(0, Nt) * p2in)
    n3 = torch.sum(torch.arange(0, Nt) * p3in)
    return n1, n2, n3


def ECD_state_generation_QM(depth, parameters, Nt,state_initial_M=None):
    """

    :param depth: circuit depth
    :param parameters: [alphas_t0,alphas_t1,betas_t0,betas_t1] all have length 2*depth
    :param Nt: cut off dimension
    :return:
    """

    # data structure
    alphas_t0 = parameters[0 * depth: 2 * depth]
    betas_t0 = parameters[2 * depth: 4 * depth]

    # prepare the probe state and ancilla
    if state_initial_M is None:
        state_QM = torch.zeros(2 * Nt, 1, dtype=torch.complex128)
        state_QM[0, 0] = 1
    else:
        state_M = state_initial_M.reshape(-1, 1)
        state_Q = torch.zeros(2, 1, dtype=torch.complex128)
        state_Q[0, 0] = 1
        state_QM = torch.kron(state_Q, state_M)

    state_QM = ECD_unitary_multi_layers(alphas_t0, betas_t0, Nt, depth) @ state_QM
    return state_QM.reshape(-1, 1)



def ECD_state_generation_M(depth, parameters, Nt, state_initial_M=None):
    """
    :param depth: circuit depth
    :param parameters: [alphas_t0,alphas_t1,betas_t0,betas_t1] all have length 2*depth
    :param Nt: cut off dimension
    :return:
    """

    state_QM = ECD_state_generation_QM(depth, parameters, Nt, state_initial_M)
    state_M0 = state_QM[0:Nt, 0]
    state_M = (state_M0 / torch.norm(state_M0)).reshape(-1, 1)
    return state_M


def ECD_state_generation_QMM(depth, parameters, Nt, state_initial_MM=None):
    """
    :param state_inital_MM:
    :param depth: circuit depth
    :param parameters: [alphas_t0,alphas_t1,betas_t0,betas_t1] all have length 2*depth
    :param Nt: cut off dimension
    :return:
    """
    # data structure
    alphas_t0 = parameters[0 * depth: 2 * depth]
    alphas_t1 = parameters[2 * depth: 4 * depth]
    betas_t0 = parameters[4 * depth: 6 * depth]
    betas_t1 = parameters[6 * depth: 8 * depth]

    # prepare the probe state and ancilla
    if state_initial_MM is None:
        state_QMM = torch.zeros(2 * Nt * Nt, 1, dtype=torch.complex128)
        state_QMM[0, 0] = 1
    else:
        state_MM = state_initial_MM.reshape(-1,1)
        state_Q = torch.zeros(2, 1, dtype=torch.complex128)
        state_Q[0, 0] = 1
        state_QMM = torch.kron(state_Q, state_MM)




    for i in range(depth):
        state_QMM = ECD_unitary_QMM(alpha1=alphas_t0[2 * i], alpha2=alphas_t0[2 * i + 1],
                                    psi=betas_t0[2 * i], theta=betas_t0[2 * i + 1],
                                    Nt=Nt, target=0, state=state_QMM)
        state_QMM = ECD_unitary_QMM(alpha1=alphas_t1[2 * i], alpha2=alphas_t1[2 * i + 1],
                                    psi=betas_t1[2 * i], theta=betas_t1[2 * i + 1],
                                    Nt=Nt, target=1, state=state_QMM)
    return state_QMM


def ECD_state_generation_MM(depth, parameters, Nt, state_initial_MM=None):
    state_QMM = ECD_state_generation_QMM(depth, parameters, Nt, state_initial_MM)
    state_MM0 = state_QMM[0:Nt * Nt, 0]
    state_MM = (state_MM0 / torch.norm(state_MM0)).reshape(-1, 1)
    return state_MM
