import numpy as np
import torch

from .ECD import *


#######################################################################################
# Transduction Protocols
#######################################################################################

def transduction_protocol_ECD_QMM(state_signal, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_t0_m = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_t1_m = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_t0_p = parameters[8 * depth:10 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in preparation
    betas_t0_m = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_t1_m = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement

    # initial state for P and S
    dimension = 2 * Nt * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state
    state_in = state.clone()
    n2_input = energy_n1n2_QMM(state_in, Nt)[1]

    state_signal = state_signal.reshape(Nt, 1)
    # combing the signal state into the whole system, Q-qubit, M1->A, M2->P,M3-S
    state_QMMM = torch.kron(state, state_signal)
    state_QMMM = state_QMMM.reshape(2, Nt, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_QMMM_after = torch.einsum('ij,abj->abi', U_BS, state_QMMM)
    state_QMMM_after = state_QMMM_after.reshape(-1, 1)

    # trace out the signal S
    rho_QMMM_after = state_QMMM_after @ dagger(state_QMMM_after)
    rho_QMMM_after = rho_QMMM_after.reshape(2, Nt, Nt, Nt, 2, Nt, Nt, Nt)
    rho_QMM_after = torch.einsum('abcd efgd->abc efg', rho_QMMM_after)

    # training the measurement
    for i in range(depth):
        rho_QMM_after = ECD_unitary_QMM(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1],
                                        psi=betas_t0_m[2 * i], theta=betas_t0_m[2 * i + 1],
                                        Nt=Nt, target=0, state=rho_QMM_after)
        rho_QMM_after = ECD_unitary_QMM(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1],
                                        psi=betas_t1_m[2 * i], theta=betas_t1_m[2 * i + 1],
                                        Nt=Nt, target=1, state=rho_QMM_after)
    rho_out = torch.einsum('abcabd->cd', rho_QMM_after)
    return rho_out, n2_input, state_in


def transduction_protocol_ECD_QMM_mixed(rho_signal, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_t0_m = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_t1_m = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_t0_p = parameters[8 * depth:10 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in preparation
    betas_t0_m = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_t1_m = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement

    # initial state for P and S
    dimension = 2 * Nt * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state
    state_in = state.clone()
    n2_input = energy_n1n2_QMM(state_in, Nt)[1]

    rho_signal = rho_signal.reshape(Nt, Nt)
    # combing the signal state into the whole system, Q-qubit, M1->A, M2->P,M3-S
    rho_QMM = state @ dagger(state)
    rho_QMMM = torch.kron(rho_QMM, rho_signal)
    rho_QMMM = rho_QMMM.reshape(2 * Nt, Nt * Nt, 2 * Nt, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    rho_QMMM_after = torch.einsum('ij,ajbl,lk->aibk', U_BS, rho_QMMM, dagger(U_BS))

    # trace out the signal S
    rho_QMMM_after = rho_QMMM_after.reshape(2, Nt, Nt, Nt, 2, Nt, Nt, Nt)
    rho_QMM_after = torch.einsum('abcd efgd->abc efg', rho_QMMM_after)

    # training the measurement
    for i in range(depth):
        rho_QMM_after = ECD_unitary_QMM(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1],
                                        psi=betas_t0_m[2 * i], theta=betas_t0_m[2 * i + 1],
                                        Nt=Nt, target=0, state=rho_QMM_after)
        rho_QMM_after = ECD_unitary_QMM(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1],
                                        psi=betas_t1_m[2 * i], theta=betas_t1_m[2 * i + 1],
                                        Nt=Nt, target=1, state=rho_QMM_after)
    rho_out = torch.einsum('abcabd->cd', rho_QMM_after)
    return rho_out, n2_input, state_in


def transduction_protocol_ECD_QM(state_signal, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t0_m = parameters[2 * depth:4 * depth]  # alphas for the  1st mode in measurement
    betas_t0_p = parameters[4 * depth:6 * depth]  # betas for the  1st mode in preparation
    betas_t0_m = parameters[6 * depth:8 * depth]  # betas for the  1st mode in measurement

    # initial state for P and S

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    # the energy of the input state
    state_in = state.clone()
    n_input = energy_n_QM(state_in, Nt)

    state_signal = state_signal.reshape(Nt, 1)
    # combing the signal state into the whole system, Q-qubit, M0->P,M1-S
    state_QMM = torch.kron(state, state_signal)
    state_QMM = state_QMM.reshape(2, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_QMM_after = torch.einsum('ij,aj->ai', U_BS, state_QMM)
    state_QMM_after = state_QMM_after.reshape(-1, 1)

    # trace out the signal S
    rho_QMM_after = state_QMM_after @ dagger(state_QMM_after)
    rho_QMM_after = rho_QMM_after.reshape(2, Nt, Nt, 2, Nt, Nt)
    rho_QM_after = torch.einsum('abc dec->ab de', rho_QMM_after)
    rho_QM_after = rho_QM_after.reshape(2 * Nt, 2 * Nt)

    # training the measurement
    for i in range(depth):
        alpha1 = alphas_t0_m[2 * i]
        alpha2 = alphas_t0_m[2 * i + 1]
        psi = betas_t0_m[2 * i]
        theta = betas_t0_m[2 * i + 1]
        # print(alpha1,alpha2,psi,theta)
        U_ECD_i = ECD_unitary(alpha1, alpha2, psi, theta, Nt)
        rho_QM_after = U_ECD_i @ rho_QM_after @ dagger(U_ECD_i)

    rho_QM_after = rho_QM_after.reshape(2, Nt, 2, Nt)
    rho_out = torch.einsum('ab ad->bd', rho_QM_after)
    return rho_out, n_input, state_in


def transduction_protocol_ECD_MM(state_signal, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_t0_m = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_t1_m = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_t0_p = parameters[8 * depth:10 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in preparation
    betas_t0_m = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_t1_m = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement

    # Q-qubit, M1->A, M2->P,M3-S
    # initial state for A and P
    dimension = 2 * Nt * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state
    state_in = state.clone()
    n2_input = energy_n1n2_QMM(state_in, Nt)[1]

    rho_in_QMM = state_in @ dagger(state_in)
    rho_in_QMM = rho_in_QMM.reshape(2, Nt, Nt, 2, Nt, Nt)
    # trace out the qubit
    rho_in_MM = torch.einsum('abc aef->bc ef', rho_in_QMM).reshape(Nt ** 2, Nt ** 2)

    state_signal = state_signal.reshape(Nt, 1)
    rho_signal = state_signal @ dagger(state_signal)

    # combing the signal state into the whole system,M1->A, M2->P,M3-S

    # the density matrix of A&P need to reshaped as Nt^2 by Nt^2 before the tensor product, otherwise the trace is not one. reason?

    rho_MMM = torch.kron(rho_in_MM, rho_signal).reshape(Nt, Nt * Nt, Nt, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    rho_MMM_after = torch.einsum('ib,ab cd, dj->ai cj', U_BS, rho_MMM, dagger(U_BS))

    # trace out the signal S
    rho_MMM_after = rho_MMM_after.reshape(Nt, Nt, Nt, Nt, Nt, Nt)

    rho_MM_after = torch.einsum('bcd fgd->bc fg', rho_MMM_after).reshape(Nt ** 2, Nt ** 2)

    # add qubit for th training of measurement
    rho_Q = torch.zeros((2, 2), dtype=torch.complex128)
    rho_Q[0, 0] = 1
    rho_QMM_after = torch.kron(rho_Q, rho_MM_after).reshape(2 * Nt ** 2, 2 * Nt ** 2)

    # training the measurement
    for i in range(depth):
        rho_QMM_after = ECD_unitary_QMM(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1],
                                        psi=betas_t0_m[2 * i], theta=betas_t0_m[2 * i + 1],
                                        Nt=Nt, target=0, state=rho_QMM_after)
        rho_QMM_after = ECD_unitary_QMM(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1],
                                        psi=betas_t1_m[2 * i], theta=betas_t1_m[2 * i + 1],
                                        Nt=Nt, target=1, state=rho_QMM_after)
    rho_out = torch.einsum('abc abd->cd', rho_QMM_after)
    return rho_out, n2_input, state_in


def transduction_protocol_ECD_QMM_EF(state_signals, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # the logical state in
    l0 = state_signals[0]
    l1 = state_signals[1]
    # Bell state of S and Q2
    state_SQ2 = (torch.kron(l0, number_state(0, 2)) + torch.kron(l1, number_state(1, 2))) / np.sqrt(2)

    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_t0_m = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_t1_m = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_t0_p = parameters[8 * depth:10 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in preparation
    betas_t0_m = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_t1_m = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement

    # Q1-ECD qubit, M1->A, M2->P,M3-S, Q2
    # initial state for A and P
    dimension = 2 * Nt * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state of Q1AP
    state_in_Q1AP = state.clone()
    n2_input = energy_n1n2_QMM(state_in_Q1AP, Nt)[1]

    state_in_Q1APSQ2 = torch.kron(state_in_Q1AP, state_SQ2).reshape(2, Nt, Nt * Nt, 2)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_in_Q1APSQ2_after = torch.einsum('ij,abjd->abid', U_BS, state_in_Q1APSQ2)
    state_in_Q1APSQ2_after = state_in_Q1APSQ2_after.reshape(2, Nt, Nt, Nt, 2)

    # training the measurement
    for i in range(depth):
        state_in_Q1APSQ2_after = ECD_unitary_QMM_plus(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1],
                                                      psi=betas_t0_m[2 * i], theta=betas_t0_m[2 * i + 1],
                                                      Nt=Nt, target=0, state=state_in_Q1APSQ2_after)
        state_in_Q1APSQ2_after = ECD_unitary_QMM_plus(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1],
                                                      psi=betas_t1_m[2 * i], theta=betas_t1_m[2 * i + 1],
                                                      Nt=Nt, target=1, state=state_in_Q1APSQ2_after)

    # trace the Q1,A and S and keep P,Q2
    state_in_Q1APSQ2_after = state_in_Q1APSQ2_after.reshape(-1, 1)
    rho_in_Q1APSQ2_after = state_in_Q1APSQ2_after @ dagger(state_in_Q1APSQ2_after)
    rho_in_Q1APSQ2_after = rho_in_Q1APSQ2_after.reshape(2, Nt, Nt, Nt, 2, 2, Nt, Nt, Nt, 2)
    rho_out_PQ2 = torch.einsum('abcde  abfdg-> cefg', rho_in_Q1APSQ2_after)
    rho_out_PQ2 = rho_out_PQ2.reshape(2 * Nt, 2 * Nt)
    ef = (dagger(state_SQ2) @ rho_out_PQ2 @ state_SQ2)[0, 0]
    return torch.real(ef), n2_input, state_in_Q1AP


def transduction_protocol_ECD_MM_EF(state_signals, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # the logical state in
    l0 = state_signals[0]
    l1 = state_signals[1]

    # Bell state of S and Q2
    state_SQ2 = (torch.kron(l0, number_state(0, 2)) + torch.kron(l1, number_state(1, 2))) / np.sqrt(2)

    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_t0_m = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_t1_m = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_t0_p = parameters[8 * depth:10 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in preparation
    betas_t0_m = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_t1_m = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement

    # Q1-ECD qubit, M1->A, M2->P,M3->S, Q2
    # initial state for Q1, A and P
    dimension = 2 * Nt * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state of Q1AP
    stateAP0 = state[0:Nt * Nt, 0]
    stateAP1 = state[Nt * Nt:2 * Nt * Nt, 0]
    pAP0 = torch.sum(torch.abs(stateAP0) ** 2)
    pAP1 = torch.sum(torch.abs(stateAP1) ** 2)
    # print(pAP0,pAP1,pAP0+pAP1)
    stateAP0 = stateAP0 / torch.sqrt(pAP0)

    state_in_AP = (stateAP0.clone()).reshape(-1, 1)
    np_input = energy_n1n2_MM(state_in_AP, Nt)[1]

    state_in_APSQ2 = torch.kron(state_in_AP, state_SQ2).reshape(Nt, Nt * Nt, 2)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    # U_BS = torch.eye(Nt*Nt, dtype=torch.complex128)
    state_in_APSQ2_after = torch.einsum('ij,bjd->bid', U_BS, state_in_APSQ2)
    state_in_APSQ2_after = state_in_APSQ2_after.reshape(Nt * Nt * Nt * 2, 1)

    state_Q1 = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q1[0, 0] = 1
    state_in_Q1APSQ2_after = torch.kron(state_Q1, state_in_APSQ2_after)
    state_in_Q1APSQ2_after = state_in_Q1APSQ2_after.reshape(2, Nt, Nt, Nt, 2)

    # training the measurement
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1], psi=betas_t0_m[2 * i],
                              theta=betas_t0_m[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1], psi=betas_t1_m[2 * i],
                              theta=betas_t1_m[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_in_Q1APSQ2_after = torch.einsum('abcd, cdghi-> abghi', U_ECD_1, state_in_Q1APSQ2_after)
        state_in_Q1APSQ2_after = torch.einsum('abcd, cgdhi-> agbhi', U_ECD_2, state_in_Q1APSQ2_after)
        """
        state_in_Q1APSQ2_after = ECD_unitary_QMM_plus(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1],
                                        psi=betas_t0_m[2 * i], theta=betas_t0_m[2 * i + 1],
                                        Nt=Nt, target=0, state=state_in_Q1APSQ2_after)
        state_in_Q1APSQ2_after = ECD_unitary_QMM_plus(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1],
                                        psi=betas_t1_m[2 * i], theta=betas_t1_m[2 * i + 1],
                                        Nt=Nt, target=1, state=state_in_Q1APSQ2_after)
        """

    # trace the Q1,A and S and keep P,Q2
    state_in_Q1APSQ2_after = state_in_Q1APSQ2_after.reshape(-1, 1)
    rho_in_Q1APSQ2_after = state_in_Q1APSQ2_after @ dagger(state_in_Q1APSQ2_after)
    rho_in_Q1APSQ2_after = rho_in_Q1APSQ2_after.reshape(2, Nt, Nt, Nt, 2, 2, Nt, Nt, Nt, 2)
    rho_out_PQ2 = torch.einsum('abcde  abfdg-> cefg', rho_in_Q1APSQ2_after)
    rho_out_PQ2 = rho_out_PQ2.reshape(2 * Nt, 2 * Nt)
    ef = (dagger(state_SQ2) @ rho_out_PQ2 @ state_SQ2)[0, 0]
    return torch.real(ef), np_input, state_in_AP


def transduction_protocol_ECD_MM_TE_EF(parameters_signals, theta, parameters, depth, Nt):
    alphas_t0_p = parameters_signals[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    betas_t0_p = parameters_signals[2 * depth:4 * depth]  # betas for the  1st mode in preparation

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the state for Q1 and P
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    # the energy of the input state of Q1AP
    stateAP0 = (state[0:Nt, 0]).reshape(-1, 1)
    stateAP1 = (state[Nt:2 * Nt, 0]).reshape(-1, 1)
    pAP0 = torch.sum(torch.abs(stateAP0) ** 2)
    pAP1 = torch.sum(torch.abs(stateAP1) ** 2)
    l0 = stateAP0 / torch.sqrt(pAP0)
    l1 = stateAP1 / torch.sqrt(pAP1)
    ns0 = energy_n_M(l0, Nt)
    ns1 = energy_n_M(l1, Nt)
    ns = (ns0 + ns1) / 2
    encoding_fidelity = (torch.abs(dagger(l0) @ l1) ** 2)[0, 0]
    state_signals = [l0, l1]
    ef, np_input, state_in_AP = transduction_protocol_ECD_MM_EF(state_signals, theta, parameters, depth, Nt)
    return ef, np_input, ns, state_in_AP, l0, l1, encoding_fidelity


def transduction_protocol_ECD_QM_EF(state_signals, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # Q1-ECD qubit, M1->P,M2-S, Q2
    # the logical state in
    l0 = state_signals[0]
    l1 = state_signals[1]
    # Bell state of S and Q2
    state_SQ2 = (torch.kron(l0, number_state(0, 2)) + torch.kron(l1, number_state(1, 2))) / np.sqrt(2)

    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t0_m = parameters[2 * depth:4 * depth]  # alphas for the  1st mode in measurement
    betas_t0_p = parameters[4 * depth:6 * depth]  # betas for the  1st mode in preparation
    betas_t0_m = parameters[6 * depth:8 * depth]  # betas for the  1st mode in measurement

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the state for Q1 and P
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    # the energy of the input state of Q1AP
    state_in_Q1P = state.clone()
    n_p = energy_n_QM(state_in_Q1P, Nt)

    state_in_Q1PSQ2 = torch.kron(state_in_Q1P, state_SQ2).reshape(2, Nt * Nt, 2)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_in_Q1PSQ2_after = torch.einsum('ij,ajd->aid', U_BS, state_in_Q1PSQ2)
    state_in_Q1PSQ2_after = state_in_Q1PSQ2_after.reshape(2, Nt, Nt, 2)

    # training the measurement
    # Q1-ECD qubit, M1->P,M2-S, Q2
    for i in range(depth):
        alpha1 = alphas_t0_m[2 * i]
        alpha2 = alphas_t0_m[2 * i + 1]
        psi = betas_t0_m[2 * i]
        theta = betas_t0_m[2 * i + 1]
        # print(alpha1,alpha2,psi,theta)
        U_ECD_i = ECD_unitary(alpha1, alpha2, psi, theta, Nt).reshape(2, Nt, 2, Nt)
        state_in_Q1PSQ2_after = torch.einsum('abcd,cdef-> abef', U_ECD_i, state_in_Q1PSQ2_after)

    # trace the Q1,S and keep P,Q2
    state_in_Q1PSQ2_after = state_in_Q1PSQ2_after.reshape(-1, 1)
    rho_in_Q1PSQ2_after = state_in_Q1PSQ2_after @ dagger(state_in_Q1PSQ2_after)
    rho_in_Q1PSQ2_after = rho_in_Q1PSQ2_after.reshape(2, Nt, Nt, 2, 2, Nt, Nt, 2)
    rho_out_PQ2 = torch.einsum('acde  afdg-> cefg', rho_in_Q1PSQ2_after)
    rho_out_PQ2 = rho_out_PQ2.reshape(2 * Nt, 2 * Nt)
    ef = (dagger(state_SQ2) @ rho_out_PQ2 @ state_SQ2)[0, 0]
    return torch.real(ef), n_p, state_in_Q1P


def transduction_protocol_ECD_QM_TE_EF(parameters_signals, theta, parameters, depth, Nt):
    alphas_t0_p = parameters_signals[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    betas_t0_p = parameters_signals[2 * depth:4 * depth]  # betas for the  1st mode in preparation

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the state for Q1 and P
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    # the energy of the input state of Q1AP
    stateAP0 = (state[0:Nt, 0]).reshape(-1, 1)
    stateAP1 = (state[Nt:2 * Nt, 0]).reshape(-1, 1)
    pAP0 = torch.sum(torch.abs(stateAP0) ** 2)
    pAP1 = torch.sum(torch.abs(stateAP1) ** 2)
    l0 = stateAP0 / torch.sqrt(pAP0)
    l1 = stateAP1 / torch.sqrt(pAP1)
    ns0 = energy_n_M(l0, Nt)
    ns1 = energy_n_M(l1, Nt)
    ns = (ns0 + ns1) / 2
    encoding_fidelity = (torch.abs(dagger(l0) @ l1) ** 2)[0, 0]
    state_signals = [l0, l1]
    ef, np_input, state_in_Q1P = transduction_protocol_ECD_QM_EF(state_signals, theta, parameters, depth, Nt)
    return ef, np_input, ns, state_in_Q1P, l0, l1, encoding_fidelity


def transduction_protocol_ECD_M_EF(state_signals, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # Q1-ECD qubit, M1->P,M2-S, Q2
    # the logical state in
    l0 = state_signals[0]
    l1 = state_signals[1]
    # Bell state of S and Q2
    state_SQ2 = (torch.kron(l0, number_state(0, 2)) + torch.kron(l1, number_state(1, 2))) / np.sqrt(2)

    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t0_m = parameters[2 * depth:4 * depth]  # alphas for the  1st mode in measurement
    betas_t0_p = parameters[4 * depth:6 * depth]  # betas for the  1st mode in preparation
    betas_t0_m = parameters[6 * depth:8 * depth]  # betas for the  1st mode in measurement

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the state for Q1 and P
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    stateP0 = state[0:Nt, 0]
    stateP1 = state[Nt:2 * Nt, 0]
    pP0 = torch.sum(torch.abs(stateP0) ** 2)
    pP1 = torch.sum(torch.abs(stateP1) ** 2)
    # print(pAP0,pAP1,pAP0+pAP1)
    stateP0 = stateP0 / torch.sqrt(pP0)
    # the energy of the input state of P
    state_in_P = (stateP0.clone()).reshape(-1, 1)
    np_input = energy_n_M(state_in_P, Nt)

    state_in_PSQ2 = torch.kron(state_in_P, state_SQ2).reshape(Nt * Nt, 2)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_in_PSQ2_after = torch.einsum('ij,jd->id', U_BS, state_in_PSQ2)
    state_in_PSQ2_after = state_in_PSQ2_after.reshape(Nt * Nt * 2, 1)

    # recombine with Q1
    state_Q1 = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q1[0, 0] = 1
    state_in_Q1PSQ2_after = torch.kron(state_Q1, state_in_PSQ2_after)
    state_in_Q1PSQ2_after = state_in_Q1PSQ2_after.reshape(2, Nt, Nt, 2)

    # training the measurement
    for i in range(depth):
        alpha1 = alphas_t0_m[2 * i]
        alpha2 = alphas_t0_m[2 * i + 1]
        psi = betas_t0_m[2 * i]
        theta = betas_t0_m[2 * i + 1]
        # print(alpha1,alpha2,psi,theta)
        U_ECD_i = ECD_unitary(alpha1, alpha2, psi, theta, Nt).reshape(2, Nt, 2, Nt)
        state_in_Q1PSQ2_after = torch.einsum('abcd,cdef-> abef', U_ECD_i, state_in_Q1PSQ2_after)

    # trace the Q1,A and S and keep P,Q2
    state_in_Q1PSQ2_after = state_in_Q1PSQ2_after.reshape(-1, 1)
    rho_in_Q1PSQ2_after = state_in_Q1PSQ2_after @ dagger(state_in_Q1PSQ2_after)
    rho_in_Q1PSQ2_after = rho_in_Q1PSQ2_after.reshape(2, Nt, Nt, 2, 2, Nt, Nt, 2)
    rho_out_PQ2 = torch.einsum('acde  afdg-> cefg', rho_in_Q1PSQ2_after)
    rho_out_PQ2 = rho_out_PQ2.reshape(2 * Nt, 2 * Nt)
    ef = (dagger(state_SQ2) @ rho_out_PQ2 @ state_SQ2)[0, 0]
    return torch.real(ef), np_input, state_in_P


def transduction_protocol_ECD_M_EF_QMMDecoding(state_signals, theta, parameters, depth, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # Q1-ECD qubit, M1->P,M2-S, Q2
    # the logical state in
    l0 = state_signals[0].reshape(-1, 1)
    l1 = state_signals[1].reshape(-1, 1)
    # Bell state of S and Q2
    state_SQ2 = (torch.kron(l0, number_state(0, 2)) + torch.kron(l1, number_state(1, 2))) / np.sqrt(2)

    # training parameters distribution
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in encoding
    alphas_t0_m = parameters[2 * depth:4 * depth]  # alphas for the  1st mode in decoding
    alphas_t1_m = parameters[4 * depth:6 * depth]  # alphas for the  2nd mode in decoding

    betas_t0_p = parameters[6 * depth:8 * depth]  # betas for the  1st mode in encoding
    betas_t0_m = parameters[8 * depth:10 * depth]  # betas for the  1st mode in decoding
    betas_t1_m = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in decoding

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the state for Q1 and P
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    stateP0 = state[0:Nt, 0]
    stateP1 = state[Nt:2 * Nt, 0]
    pP0 = torch.sum(torch.abs(stateP0) ** 2)
    pP1 = torch.sum(torch.abs(stateP1) ** 2)
    # print(pAP0,pAP1,pAP0+pAP1)
    stateP0 = stateP0 / torch.sqrt(pP0)
    # the energy of the input state of P
    state_in_P = (stateP0.clone()).reshape(-1, 1)
    np_input = energy_n_M(state_in_P, Nt)

    state_in_PSQ2 = torch.kron(state_in_P, state_SQ2).reshape(Nt * Nt, 2)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_in_PSQ2_after = torch.einsum('ij,jd->id', U_BS, state_in_PSQ2)
    state_in_PSQ2_after = state_in_PSQ2_after.reshape(Nt * Nt * 2, 1)

    # recombine with Q1 and D
    state_Q1D = torch.zeros((2 * Nt, 1), dtype=torch.complex128)
    state_Q1D[0, 0] = 1
    state_in_Q1DPSQ2_after = torch.kron(state_Q1D, state_in_PSQ2_after)
    state_in_Q1DPSQ2_after = state_in_Q1DPSQ2_after.reshape(2, Nt, Nt, Nt, 2)

    # training the decoding
    for i in range(depth):
        U_ECD_i0 = ECD_unitary(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1], psi=betas_t0_m[2 * i],
                               theta=betas_t0_m[2 * i + 1], Nt=Nt).reshape(2, Nt, 2, Nt)
        U_ECD_i1 = ECD_unitary(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1], psi=betas_t1_m[2 * i],
                               theta=betas_t1_m[2 * i + 1], Nt=Nt).reshape(2, Nt, 2, Nt)
        state_in_Q1DPSQ2_after = torch.einsum('abcd,cdefg-> abefg', U_ECD_i0, state_in_Q1DPSQ2_after)
        state_in_Q1DPSQ2_after = torch.einsum('abcd,cedfg-> aebfg', U_ECD_i1, state_in_Q1DPSQ2_after)

    # trace the Q1,D and S and keep P,Q2
    state_in_Q1DPSQ2_after = state_in_Q1DPSQ2_after.reshape(-1, 1)
    rho_in_Q1DPSQ2_after = state_in_Q1DPSQ2_after @ dagger(state_in_Q1DPSQ2_after)
    rho_in_Q1DPSQ2_after = rho_in_Q1DPSQ2_after.reshape(2, Nt, Nt, Nt, 2, 2, Nt, Nt, Nt, 2)
    rho_out_PQ2 = torch.einsum('abcde  abfdg-> cefg', rho_in_Q1DPSQ2_after)
    rho_out_PQ2 = rho_out_PQ2.reshape(2 * Nt, 2 * Nt)
    ef = (dagger(state_SQ2) @ rho_out_PQ2 @ state_SQ2)[0, 0]
    return torch.real(ef), np_input, state_in_P


def transduction_protocol_ECD_M_TE_EF_QMMDecoding(parameters_signals, theta, parameters, depth, Nt):
    alphas_t0_p = parameters_signals[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    betas_t0_p = parameters_signals[2 * depth:4 * depth]  # betas for the  1st mode in preparation

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the state for Q1 and P
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    # the energy of the input state of Q1AP
    stateAP0 = (state[0:Nt, 0]).reshape(-1, 1)
    stateAP1 = (state[Nt:2 * Nt, 0]).reshape(-1, 1)
    pAP0 = torch.sum(torch.abs(stateAP0) ** 2)
    pAP1 = torch.sum(torch.abs(stateAP1) ** 2)
    l0 = (stateAP0 / torch.sqrt(pAP0)).reshape(-1, 1)
    l1 = (stateAP1 / torch.sqrt(pAP1)).reshape(-1, 1)
    encoding_fidelity = (torch.abs(dagger(l0) @ l1) ** 2)[0, 0]
    state_signals = [l0, l1]
    ns0 = energy_n_M(l0, Nt)
    ns1 = energy_n_M(l1, Nt)
    ns = (ns0 + ns1) / 2
    state_SQ2 = (torch.kron(l0, number_state(0, 2)) + torch.kron(l1, number_state(1, 2))) / np.sqrt(2)
    state_SQ2 = state_SQ2.reshape(-1, 1)
    rho_SQ2 = (state_SQ2 @ dagger(state_SQ2)).reshape(Nt, 2, Nt, 2)
    rho_Q2 = torch.einsum("abcb-> ac", rho_SQ2)
    rho_S = torch.einsum("abad-> bd", rho_SQ2)
    entropy_SQ2 = von_neumann_entropy(rho_S)

    ef, np_input, state_in_P = transduction_protocol_ECD_M_EF_QMMDecoding(state_signals, theta, parameters, depth, Nt)
    return ef, np_input, ns, state_in_P, l0, l1, encoding_fidelity, entropy_SQ2


def transduction_protocol_ECD_M_TE_EF(parameters_signals, theta, parameters, depth, Nt):
    alphas_t0_p = parameters_signals[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    betas_t0_p = parameters_signals[2 * depth:4 * depth]  # betas for the  1st mode in preparation

    dimension = 2 * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the state for Q1 and P
    state = ECD_unitary_multi_layers(alphas_t0_p, betas_t0_p, Nt, depth) @ state

    # the energy of the input state of Q1AP
    stateAP0 = (state[0:Nt, 0]).reshape(-1, 1)
    stateAP1 = (state[Nt:2 * Nt, 0]).reshape(-1, 1)
    pAP0 = torch.sum(torch.abs(stateAP0) ** 2)
    pAP1 = torch.sum(torch.abs(stateAP1) ** 2)
    l0 = stateAP0 / torch.sqrt(pAP0)
    l1 = stateAP1 / torch.sqrt(pAP1)
    encoding_fidelity = (torch.abs(dagger(l0) @ l1) ** 2)[0, 0]
    state_signals = [l0, l1]
    ns0 = energy_n_M(l0, Nt)
    ns1 = energy_n_M(l1, Nt)
    ns = (ns0 + ns1) / 2

    ef, np_input, state_in_P = transduction_protocol_ECD_M_EF(state_signals, theta, parameters, depth, Nt)
    return ef, np_input, ns, state_in_P, l0, l1, encoding_fidelity


def transduction_protocol_ECD_MM_TMSVEF(r, theta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->Q,1->A, 2->P,3->S,4->R

    # TMSV state of S and R
    state_SR = torch.zeros(Nt * Nt, 1, dtype=torch.complex128)
    state_SR[0, 0] = 1
    U_TMS = two_mode_squeeze(r, Nt)
    state_SR = U_TMS @ state_SR

    # training parameters distribution for Q, A and P
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_t0_m = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_t1_m = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_t0_p = parameters[8 * depth:10 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in preparation
    betas_t0_m = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_t1_m = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement

    # initial state for Q1, A and P
    state = torch.zeros((2 * Nt * Nt, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state of Q1AP
    stateAP0 = state[0:Nt * Nt, 0]
    stateAP1 = state[Nt * Nt:2 * Nt * Nt, 0]
    pAP0 = torch.sum(torch.abs(stateAP0) ** 2)
    pAP1 = torch.sum(torch.abs(stateAP1) ** 2)
    # print(pAP0,pAP1,pAP0+pAP1)
    stateAP0 = stateAP0 / torch.sqrt(pAP0)

    state_in_AP = (stateAP0.clone()).reshape(-1, 1)
    np_input = energy_n1n2_MM(state_in_AP, Nt)[1]

    state_in_APSR = torch.kron(state_in_AP, state_SR).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    # U_BS = torch.eye(Nt*Nt, dtype=torch.complex128)
    state_in_APSR_after = torch.einsum('ij,bjd->bid', U_BS, state_in_APSR)
    state_in_APSR_after = state_in_APSR_after.reshape(Nt * Nt * Nt * Nt, 1)

    state_Q1 = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q1[0, 0] = 1
    state_in_Q1APSR_after = torch.kron(state_Q1, state_in_APSR_after)
    state_in_Q1APSR_after = state_in_Q1APSR_after.reshape(2, Nt, Nt, Nt, Nt)

    # training the measurement
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1], psi=betas_t0_m[2 * i],
                              theta=betas_t0_m[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1], psi=betas_t1_m[2 * i],
                              theta=betas_t1_m[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_in_Q1APSR_after = torch.einsum('abcd, cdghi-> abghi', U_ECD_1, state_in_Q1APSR_after)
        state_in_Q1APSR_after = torch.einsum('abcd, cgdhi-> agbhi', U_ECD_2, state_in_Q1APSR_after)

    # trace the Q1,A and S and keep P,Q2
    state_in_Q1APSR_after = state_in_Q1APSR_after.reshape(-1, 1)
    rho_in_Q1APSR_after = state_in_Q1APSR_after @ dagger(state_in_Q1APSR_after)
    rho_in_Q1APSR_after = rho_in_Q1APSR_after.reshape(2, Nt, Nt, Nt, Nt, 2, Nt, Nt, Nt, Nt)
    rho_out_PR = torch.einsum('abcde  abfdg-> cefg', rho_in_Q1APSR_after)
    rho_out_PR = rho_out_PR.reshape(Nt * Nt, Nt * Nt)
    ef = (dagger(state_SR) @ rho_out_PR @ state_SR)[0, 0]
    return torch.real(ef), np_input, state_in_AP


def transduction_protocol_TMS_TMSVEF(r0, eta, n, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters of TMS G ang G'
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    r = np.arcsinh(np.sqrt(n))
    G = np.cosh(r) ** 2
    Gprime = G / (G * eta + 1 - eta)
    rprime = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(Gprime))))
    theta = np.arcsin(np.sqrt(eta))

    # 0->A, 1->P,2->S,3->R

    TMS0 = two_mode_squeeze(r0, Nt)
    TMS1 = two_mode_squeeze(r, Nt)
    TMS2 = two_mode_squeeze(-rprime, Nt)

    # initial state for S and R
    state_SR = torch.zeros((Nt * Nt, 1), dtype=torch.complex128)
    state_SR[0, 0] = 1
    state_SR = TMS0 @ state_SR
    state_SR = state_SR.reshape(-1, 1)

    # initial state for A and P
    state_AP = torch.zeros((Nt * Nt, 1), dtype=torch.complex128)
    state_AP[0, 0] = 1  # initial state
    state_AP = TMS1 @ state_AP  # prepare the probe state TMS in AP
    state_AP = state_AP.reshape(-1, 1)

    # the energy of the input state
    state_in = state_AP.clone()
    np_input = energy_n1n2_MM(state_in, Nt)[0]

    # combing the signal state into the whole system
    state_APSR = torch.kron(state_AP, state_SR)

    # apply the beam splittr between S and P
    state_APSR = state_APSR.reshape(Nt, Nt * Nt, Nt)
    state_APSR = torch.einsum('ij,ajb->aib', unitary_beam_splitter(theta, Nt), state_APSR)
    state_APSR = state_APSR.reshape(-1, 1)

    # apply anti squeezing between A and P
    state_APSR = state_APSR.reshape(Nt * Nt, Nt, Nt)
    state_APSR = torch.einsum('ij,jab->iab', TMS2, state_APSR)
    state_APSR = state_APSR.reshape(-1, 1)

    # trace out the signal S and A
    rho_APSR_after = state_APSR @ dagger(state_APSR)
    rho_APSR_after = rho_APSR_after.reshape(Nt, Nt, Nt, Nt, Nt, Nt, Nt, Nt)
    rho_PR_after = torch.einsum('abcd afch->bdfh', rho_APSR_after)
    rho_PR_after = rho_PR_after.reshape(Nt * Nt, Nt * Nt)

    # calculate entanglement fidelity
    ef = (dagger(state_SR) @ rho_PR_after @ state_SR)[0, 0]

    return ef


def Coherent_Info_ECD_MM_TMSVEF(r, theta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->Q,1->A, 2->P,3->S,4->R

    # TMSV state of S and R
    state_SR = torch.zeros(Nt * Nt, 1, dtype=torch.complex128)
    state_SR[0, 0] = 1
    U_TMS = two_mode_squeeze(r, Nt)
    state_SR = U_TMS @ state_SR

    # training parameters distribution for Q, A and P
    alphas_t0_p = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_t0_m = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_t1_m = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_t0_p = parameters[8 * depth:10 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in preparation
    betas_t0_m = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_t1_m = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement

    # initial state for Q1, A and P
    state = torch.zeros((2 * Nt * Nt, 1), dtype=torch.complex128)
    state[0, 0] = 1  # initial state
    # prepare the probe state and ancilla
    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state of Q1AP
    stateAP0 = state[0:Nt * Nt, 0]
    stateAP1 = state[Nt * Nt:2 * Nt * Nt, 0]
    pAP0 = torch.sum(torch.abs(stateAP0) ** 2)
    pAP1 = torch.sum(torch.abs(stateAP1) ** 2)
    # print(pAP0,pAP1,pAP0+pAP1)
    stateAP0 = stateAP0 / torch.sqrt(pAP0)

    state_in_AP = (stateAP0.clone()).reshape(-1, 1)
    np_input = energy_n1n2_MM(state_in_AP, Nt)[1]

    state_in_APSR = torch.kron(state_in_AP, state_SR).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    # U_BS = torch.eye(Nt*Nt, dtype=torch.complex128)
    state_in_APSR_after = torch.einsum('ij,bjd->bid', U_BS, state_in_APSR)
    state_in_APSR_after = state_in_APSR_after.reshape(Nt * Nt * Nt * Nt, 1)

    state_Q1 = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q1[0, 0] = 1
    state_in_Q1APSR_after = torch.kron(state_Q1, state_in_APSR_after)
    state_in_Q1APSR_after = state_in_Q1APSR_after.reshape(2, Nt, Nt, Nt, Nt)

    # training the measurement
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_t0_m[2 * i], alpha2=alphas_t0_m[2 * i + 1], psi=betas_t0_m[2 * i],
                              theta=betas_t0_m[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_t1_m[2 * i], alpha2=alphas_t1_m[2 * i + 1], psi=betas_t1_m[2 * i],
                              theta=betas_t1_m[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_in_Q1APSR_after = torch.einsum('abcd, cdghi-> abghi', U_ECD_1, state_in_Q1APSR_after)
        state_in_Q1APSR_after = torch.einsum('abcd, cgdhi-> agbhi', U_ECD_2, state_in_Q1APSR_after)

    # trace the Q1,A and S and keep P,Q2
    state_in_Q1APSR_after = state_in_Q1APSR_after.reshape(-1, 1)
    rho_in_Q1APSR_after = state_in_Q1APSR_after @ dagger(state_in_Q1APSR_after)
    rho_in_Q1APSR_after = rho_in_Q1APSR_after.reshape(2, Nt, Nt, Nt, Nt, 2, Nt, Nt, Nt, Nt)
    rho_out_PR = torch.einsum('abcde  abfdg-> cefg', rho_in_Q1APSR_after)
    rho_out_PR = rho_out_PR.reshape(Nt * Nt, Nt * Nt)
    ef = (dagger(state_SR) @ rho_out_PR @ state_SR)[0, 0]
    return torch.real(ef), np_input, state_in_AP


def transduction_protocol_CoherentInfo_ECD_MM_EATele(theta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_a_0 = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_p_1 = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_a_1 = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement
    alphas_p_2 = parameters[8 * depth:10 * depth]  # alphas for the  1st mode in measurement
    alphas_a_2 = parameters[10 * depth:12 * depth]  # alphas for the  2nd mode in measurement
    alphas_r = parameters[12 * depth:14 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[14 * depth:16 * depth]  # alphas for mode S in preparation

    betas_p_0 = parameters[16 * depth:18 * depth]  # betas for the  1st mode in preparation
    betas_a_0 = parameters[18 * depth:20 * depth]  # betas for the  2nd mode in preparation
    betas_p_1 = parameters[20 * depth:22 * depth]  # betas for the  1st mode in measurement
    betas_a_1 = parameters[22 * depth:24 * depth]  # betas for the  2nd mode in measurement
    betas_p_2 = parameters[24 * depth:26 * depth]  # betas for the  1st mode in measurement
    betas_a_2 = parameters[26 * depth:28 * depth]  # betas for the  2nd mode in measurement
    betas_r = parameters[28 * depth:30 * depth]  # betas for mode R in preparation
    betas_s = parameters[30 * depth:32 * depth]  # betas for mode R in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    parameters_PA_0 = torch.cat([alphas_p_0, alphas_a_0, betas_p_0, betas_a_0])
    state_PA = ECD_state_generation_MM(depth, parameters_PA_0, Nt)
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS, state_PA).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)
    state_RSPA = state_RSPA.reshape(-1, 1)

    # measurement training part 1
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPAQ = torch.kron(state_RSPA, state_Q)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_1, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_2, state_RSPAQ)

    # Conditional displacement on P conditioned on S
    SUM = SUM_gate(Nt, alpha=parameters[-1])
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt * Nt, Nt, 2)
    state_RSPAQ = torch.einsum('ab, cbde-> cade', SUM, state_RSPAQ)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    # measurement training part 2
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_2[2 * i], alpha2=alphas_p_2[2 * i + 1],
                              psi=betas_p_2[2 * i], theta=betas_p_2[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_a_2[2 * i], alpha2=alphas_a_2[2 * i + 1],
                              psi=betas_a_2[2 * i], theta=betas_a_2[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_1, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_2, state_RSPAQ)

    # trace the P
    state_RSPAQ = state_RSPAQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)
    #rho_SA = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[1, 3])
    #rho_SA = rho_SA.reshape(Nt * Nt, Nt * Nt)
    #print(von_neumann_entropy(rho_P),von_neumann_entropy(rho_RP),von_neumann_entropy(rho_SA))
    return CI, ns_input, np_input, state_RS, state_PA,


def transduction_protocol_CoherentInfo_ECD_M_EATele(theta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_p_1 = parameters[2 * depth:4 * depth]  # alphas for the  1st mode in measurement
    alphas_p_2 = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_r = parameters[6 * depth:8 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[8 * depth:10 * depth]  # alphas for mode S in preparation

    betas_p_0 = parameters[10 * depth:12 * depth]  # betas for the  1st mode in preparation
    betas_p_1 = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_p_2 = parameters[14 * depth:16 * depth]  # betas for the  1st mode in measurement
    betas_r = parameters[16 * depth:18 * depth]  # betas for mode R in preparation
    betas_s = parameters[18 * depth:20 * depth]  # betas for mode R in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for P
    parameters_P_0 = torch.cat([alphas_p_0, betas_p_0])
    state_P = ECD_state_generation_M(depth, parameters_P_0, Nt)
    np_input = energy_n_M(state_P, Nt)

    state_RSP = torch.kron(state_RS, state_P).reshape(Nt, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_RSP = torch.einsum('ij,kj->ki', U_BS, state_RSP)
    state_RSP = state_RSP.reshape(-1, 1)

    # measurement training part 1
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPQ = torch.kron(state_RSP, state_Q)
    state_RSPQ = state_RSPQ.reshape(Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        state_RSPQ = torch.einsum('abcd, hgdc-> hgba', U_ECD_1, state_RSPQ)

    # Conditional displacement on P conditioned on S
    SUM = SUM_gate(Nt, alpha=parameters[-1])
    state_RSPQ = state_RSPQ.reshape(Nt, Nt * Nt, 2)
    state_RSPQ = torch.einsum('ab, cbe-> cae', SUM, state_RSPQ)
    state_RSPQ = state_RSPQ.reshape(Nt, Nt, Nt, 2)

    # measurement training part 2
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_2[2 * i], alpha2=alphas_p_2[2 * i + 1],
                              psi=betas_p_2[2 * i], theta=betas_p_2[2 * i + 1], Nt=Nt)

        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        state_RSPQ = torch.einsum('abcd, hgdc-> hgba', U_ECD_1, state_RSPQ)

    # trace the P
    state_RSPQ = state_RSPQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPQ, shape=[Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPQ, shape=[Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)
    #rho_SA = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[1, 3])
    #rho_SA = rho_SA.reshape(Nt * Nt, Nt * Nt)
    #print(von_neumann_entropy(rho_P),von_neumann_entropy(rho_RP),von_neumann_entropy(rho_SA))
    return CI, ns_input, np_input, state_RS, state_P


def transduction_protocol_CoherentInfo_ECD_MM_EATele_fixedinput(theta, ns, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    # training parameters distribution for R,S, P and A

    alphas_p_1 = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in measurement
    alphas_a_1 = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in measurement
    alphas_p_2 = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_a_2 = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement

    betas_p_1 = parameters[8 * depth:10 * depth]  # betas for the  1st mode in measurement
    betas_a_1 = parameters[10 * depth:12 * depth]  # betas for the  2nd mode in measurement
    betas_p_2 = parameters[12 * depth:14 * depth]  # betas for the  1st mode in measurement
    betas_a_2 = parameters[14 * depth:16 * depth]  # betas for the  2nd mode in measurement
    r_s, theta_s, r_p, theta_p = parameters[16 * depth:16 * depth + 4]

    # input state for RS
    ns_thermal = (ns - torch.sinh(r_s) ** 2) / (1 + 2 * torch.sinh(r_s) ** 2)
    state_RS = thermal_state_twomode_pure(ns_thermal, Nt).reshape(Nt, Nt)
    US_squeeze = squeezed_operator(r_s, theta_s, Nt)
    state_RS = torch.einsum("ab, cb-> ca", US_squeeze, state_RS)
    state_RS = state_RS.reshape(Nt * Nt, 1)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    UP_squeeze = squeezed_operator(r_p, theta_p, Nt)
    state_P = UP_squeeze @ number_state(0, Nt)
    state_PA = torch.kron(state_P, number_state(0, Nt))
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS, state_PA).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)
    state_RSPA = state_RSPA.reshape(-1, 1)

    # measurement training part 1
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPAQ = torch.kron(state_RSPA, state_Q)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_1, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_2, state_RSPAQ)

    # Conditional displacement on P conditioned on S
    SUM = SUM_gate(Nt, alpha=parameters[-1])
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt * Nt, Nt, 2)
    state_RSPAQ = torch.einsum('ab, cbde-> cade', SUM, state_RSPAQ)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    # measurement training part 2
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_2[2 * i], alpha2=alphas_p_2[2 * i + 1],
                              psi=betas_p_2[2 * i], theta=betas_p_2[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_a_2[2 * i], alpha2=alphas_a_2[2 * i + 1],
                              psi=betas_a_2[2 * i], theta=betas_a_2[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_1, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_2, state_RSPAQ)

    # trace the P
    state_RSPAQ = state_RSPAQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)
    #rho_SA = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[1, 3])
    #rho_SA = rho_SA.reshape(Nt * Nt, Nt * Nt)
    #print(von_neumann_entropy(rho_P),von_neumann_entropy(rho_RP),von_neumann_entropy(rho_SA))
    return CI, ns_input, np_input, state_RS, state_PA,


def transduction_protocol_CoherentInfo_ECD_MM_EATele_UM(theta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_a_0 = parameters[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    alphas_p_1 = parameters[4 * depth:6 * depth]  # alphas for the  1st mode in measurement
    alphas_a_1 = parameters[6 * depth:8 * depth]  # alphas for the  2nd mode in measurement
    alphas_p_2 = parameters[8 * depth:10 * depth]  # alphas for the  1st mode in measurement
    alphas_a_2 = parameters[10 * depth:12 * depth]  # alphas for the  2nd mode in measurement
    alphas_r = parameters[12 * depth:14 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[14 * depth:16 * depth]  # alphas for mode S in preparation
    alphas_s_2 = parameters[16 * depth:18 * depth]  # alphas for mode S before homodyne measurement

    betas_p_0 = parameters[18 * depth:20 * depth]  # betas for the  1st mode in preparation
    betas_a_0 = parameters[20 * depth:22 * depth]  # betas for the  2nd mode in preparation
    betas_p_1 = parameters[22 * depth:24 * depth]  # betas for the  1st mode in measurement
    betas_a_1 = parameters[24 * depth:26 * depth]  # betas for the  2nd mode in measurement
    betas_p_2 = parameters[26 * depth:28 * depth]  # betas for the  1st mode in measurement
    betas_a_2 = parameters[28 * depth:30 * depth]  # betas for the  2nd mode in measurement
    betas_r = parameters[30 * depth:32 * depth]  # betas for mode R in preparation
    betas_s = parameters[32 * depth:34 * depth]  # betas for mode S in preparation
    betas_s_2 = parameters[34 * depth:36 * depth]  # betas for mode S before homodyne measurement

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    parameters_PA_0 = torch.cat([alphas_p_0, alphas_a_0, betas_p_0, betas_a_0])
    state_PA = ECD_state_generation_MM(depth, parameters_PA_0, Nt)
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS, state_PA).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)
    state_RSPA = state_RSPA.reshape(-1, 1)

    # measurement training
    # Q2 controls S and Q1 controls PA
    state_Q1 = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q1[0, 0] = 1
    state_Q2 = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q2[0, 0] = 1
    state_RSPAQ1 = torch.kron(state_RSPA, state_Q1)
    state_RSPAQ1Q2 = torch.kron(state_RSPAQ1, state_Q2)
    state_RSPAQ1Q2 = state_RSPAQ1Q2.reshape(Nt, Nt, Nt, Nt, 2, 2)

    # measurement training  on S
    for i in range(depth):
        U_ECD = ECD_unitary(alpha1=alphas_s_2[2 * i], alpha2=alphas_s_2[2 * i + 1], psi=betas_s_2[2 * i],
                            theta=betas_s_2[2 * i + 1], Nt=Nt)

        U_ECD = U_ECD.reshape(2, Nt, 2, Nt)
        state_RSPAQ1Q2 = torch.einsum('abcd, edghic-> ebghia', U_ECD, state_RSPAQ1Q2)

    # measurement training  on PA part 1
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_RSPAQ1Q2 = torch.einsum('abcd, hgdfci-> hgbfai', U_ECD_1, state_RSPAQ1Q2)
        state_RSPAQ1Q2 = torch.einsum('abcd, hgfdci-> hgfbai', U_ECD_2, state_RSPAQ1Q2)

    # Conditional displacement on P conditioned on S
    SUM = SUM_gate(Nt, alpha=parameters[-1])
    state_RSPAQ1Q2 = state_RSPAQ1Q2.reshape(Nt, Nt * Nt, Nt, 2, 2)
    state_RSPAQ1Q2 = torch.einsum('ab, cbdef-> cadef', SUM, state_RSPAQ1Q2)
    state_RSPAQ1Q2 = state_RSPAQ1Q2.reshape(Nt, Nt, Nt, Nt, 2, 2)

    # measurement training part 2
    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_2[2 * i], alpha2=alphas_p_2[2 * i + 1],
                              psi=betas_p_2[2 * i], theta=betas_p_2[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_a_2[2 * i], alpha2=alphas_a_2[2 * i + 1],
                              psi=betas_a_2[2 * i], theta=betas_a_2[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_RSPAQ1Q2 = torch.einsum('abcd, hgdfci-> hgbfai', U_ECD_1, state_RSPAQ1Q2)
        state_RSPAQ1Q2 = torch.einsum('abcd, hgfdci-> hgfbai', U_ECD_2, state_RSPAQ1Q2)

    # trace the P
    state_RSPAQ1Q2 = state_RSPAQ1Q2.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPAQ1Q2, shape=[Nt, Nt, Nt, Nt, 2, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPAQ1Q2, shape=[Nt, Nt, Nt, Nt, 2, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)
    # rho_SA = partial_trace_torch(state_RSPAQ1Q2, shape=[Nt, Nt, Nt, Nt, 2, 2], sel=[1, 3])
    # rho_SA = rho_SA.reshape(Nt * Nt, Nt * Nt)
    # print(von_neumann_entropy(rho_P), von_neumann_entropy(rho_RP), von_neumann_entropy(rho_SA))

    return CI, ns_input, np_input, state_RS, state_PA,


def transduction_protocol_CoherentInfo_ECD_MM_EA(eta, parameters, depth, Nt,state_initial_RS=None,state_initial_PA=None):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    theta = np.arcsin(np.sqrt(eta))  # eta = sin(theta)**2

    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  mode p in preparation
    alphas_a_0 = parameters[2 * depth:4 * depth]  # alphas for the  mode a in preparation
    alphas_p_1 = parameters[4 * depth:6 * depth]  # alphas for the  mode p in measurement
    alphas_a_1 = parameters[6 * depth:8 * depth]  # alphas for the  mode a in measurement
    alphas_r = parameters[8 * depth:10 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[10 * depth:12 * depth]  # alphas for mode S in preparation

    betas_p_0 = parameters[12 * depth:14 * depth]  # betas for the  mode p in preparation
    betas_a_0 = parameters[14 * depth:16 * depth]  # betas for the  mode a in preparation
    betas_p_1 = parameters[16 * depth:18 * depth]  # betas for the  mode p in measurement
    betas_a_1 = parameters[18 * depth:20 * depth]  # betas for the  mode a in measurement
    betas_r = parameters[20 * depth:22 * depth]  # betas for mode R in preparation
    betas_s = parameters[22 * depth:24 * depth]  # betas for mode S in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt,state_initial_MM=state_initial_RS)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    parameters_PA_0 = torch.cat([alphas_p_0, alphas_a_0, betas_p_0, betas_a_0])
    state_PA = ECD_state_generation_MM(depth, parameters_PA_0, Nt,state_initial_MM=state_initial_PA )  #
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS.reshape(-1, 1), state_PA.reshape(-1, 1)).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)

    # measurement training
    state_RSPA = state_RSPA.reshape(-1, 1)
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPAQ = torch.kron(state_RSPA, state_Q)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_p = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_a = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_p = U_ECD_p.reshape(2, Nt, 2, Nt)
        U_ECD_a = U_ECD_a.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_p, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_a, state_RSPAQ)

    # calculate Coherent Information
    state_RSPAQ = state_RSPAQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)

    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, np_input, state_RS, state_PA


def transduction_protocol_CoherentInfo_ECD_M_EA(eta, parameters, depth, Nt, state_initial_RS = None, state_initial_P = None):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    theta = np.arcsin(np.sqrt(eta))  # eta = sin(theta)**2

    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  mode p in preparation
    alphas_p_1 = parameters[2 * depth:4 * depth]  # alphas for the  mode p in measurement
    alphas_r = parameters[4 * depth:6 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[6 * depth:8 * depth]  # alphas for mode S in preparation

    betas_p_0 = parameters[8 * depth:10 * depth]  # betas for the  mode p in preparation
    betas_p_1 = parameters[10 * depth:12 * depth]  # betas for the  mode p in measurement
    betas_r = parameters[12 * depth:14 * depth]  # betas for mode R in preparation
    betas_s = parameters[14 * depth:16 * depth]  # betas for mode S in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt,state_initial_MM=state_initial_RS)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    parameters_P_0 = torch.cat([alphas_p_0, betas_p_0])
    state_P = ECD_state_generation_M(depth, parameters_P_0, Nt,state_initial_M=state_initial_P )  #
    np_input = energy_n_M(state_P, Nt)

    state_RSP = torch.kron(state_RS.reshape(-1, 1), state_P.reshape(-1, 1)).reshape(Nt, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSP = torch.einsum('ij,kj->ki', U_BS, state_RSP)

    # measurement training
    state_RSP = state_RSP.reshape(-1, 1)
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPQ = torch.kron(state_RSP, state_Q)
    state_RSPQ = state_RSPQ.reshape(Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_p = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_p = U_ECD_p.reshape(2, Nt, 2, Nt)
        state_RSPQ = torch.einsum('abcd, hgdc-> hgba', U_ECD_p, state_RSPQ)

    # calculate Coherent Information
    state_RSPQ = state_RSPQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPQ, shape=[Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPQ, shape=[Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)

    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, np_input, state_RS, state_P


def generate_entangled_GKP(d, parameters, Nt):
    state = torch.zeros((Nt * Nt, 1), dtype=torch.complex128)
    delta = parameters[-2]
    kappa = parameters[-1]
    for i in range(d):
        amplitude, theta1, theta2 = parameters[3 * i:3 * (i + 1)]
        state_m1 = phase_rotation(theta2, Nt) @ GKP_state_approx(i, delta, kappa, d, Nt, slimit=5, type_torch=True)
        state_m2 = number_state(i, Nt)
        state = state + amplitude * torch.exp(1j * theta1) * torch.kron(state_m2, state_m1).reshape(-1, 1)
    state = state / torch.norm(state)
    return state


def generate_entangled_squeezedcoherent(d, parameters, Nt):
    state = torch.zeros((Nt * Nt, 1), dtype=torch.complex128)

    for i in range(d):
        amplitude, theta1, theta2, r, alpha1, alpha2 = parameters[6 * i: 6 * (i + 1)]
        state_m1 = (squeezed_operator(r, theta2, Nt) @ coherent_state_torch(alpha1 + 1j * alpha2, Nt)).reshape(-1, 1)
        state_m2 = number_state(i, Nt).reshape(-1, 1)
        state = state + amplitude * torch.exp(1j * theta1) * torch.kron(state_m1, state_m2).reshape(-1, 1)
    state = state / torch.norm(state)
    return state


def transduction_protocol_CoherentInfo_ECD_MM_EA_fixedinput(d, eta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    theta = np.arcsin(np.sqrt(eta))  # eta = sin(theta)**2

    # training parameters distribution for R,S, P and A

    alphas_p_1 = parameters[0 * depth:2 * depth]  # alphas for the  mode p in measurement
    alphas_a_1 = parameters[2 * depth:4 * depth]  # alphas for the  mode a in measurement
    betas_p_1 = parameters[4 * depth:6 * depth]  # betas for the  mode p in measurement
    betas_a_1 = parameters[6 * depth:8 * depth]  # betas for the  mode a in measurement
    parameters_RS = parameters[8 * depth:8 * depth + 3 * d + 2]
    parameters_PA = parameters[8 * depth + 3 * d + 2: 8 * depth + 3 * d + 2 + 6 * d]

    # input state for RS
    state_RS = generate_entangled_GKP(d, parameters_RS, Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    state_PA = generate_entangled_squeezedcoherent(d, parameters_PA, Nt)
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS.reshape(-1, 1), state_PA.reshape(-1, 1)).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)

    # measurement training
    state_RSPA = state_RSPA.reshape(-1, 1)
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPAQ = torch.kron(state_RSPA, state_Q)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_p = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_a = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_p = U_ECD_p.reshape(2, Nt, 2, Nt)
        U_ECD_a = U_ECD_a.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_p, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_a, state_RSPAQ)

    # calculate Coherent Information
    state_RSPAQ = state_RSPAQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)

    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, np_input, state_RS, state_PA


def transduction_protocol_CoherentInfo_ECD_MM_EA_TMSInitial(eta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    theta = np.arcsin(np.sqrt(eta))  # eta = sin(theta)**2
    G = 4 + 1
    Gprime = G / (G * eta + 1 - eta)
    r = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(G))))
    rprime = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(Gprime))))

    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  mode p in preparation
    alphas_a_0 = parameters[2 * depth:4 * depth]  # alphas for the  mode a in preparation
    alphas_p_1 = parameters[4 * depth:6 * depth]  # alphas for the  mode p in measurement
    alphas_a_1 = parameters[6 * depth:8 * depth]  # alphas for the  mode a in measurement
    alphas_r = parameters[8 * depth:10 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[10 * depth:12 * depth]  # alphas for mode S in preparation

    betas_p_0 = parameters[12 * depth:14 * depth]  # betas for the  mode p in preparation
    betas_a_0 = parameters[14 * depth:16 * depth]  # betas for the  mode a in preparation
    betas_p_1 = parameters[16 * depth:18 * depth]  # betas for the  mode p in measurement
    betas_a_1 = parameters[18 * depth:20 * depth]  # betas for the  mode a in measurement
    betas_r = parameters[20 * depth:22 * depth]  # betas for mode R in preparation
    betas_s = parameters[22 * depth:24 * depth]  # betas for mode S in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt, state_initial_MM=thermal_state_twomode_pure(4, Nt))
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    parameters_PA_0 = torch.cat([alphas_p_0, alphas_a_0, betas_p_0, betas_a_0])
    G = 4 + 1
    r = np.abs(np.arccosh(np.sqrt(G)))

    state_PA = ECD_state_generation_MM(depth, parameters_PA_0, Nt, state_initial_MM=tmsv_state_analytical(-r, Nt))  #
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS.reshape(-1, 1), state_PA.reshape(-1, 1)).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)

    # apply the anti-squeezer
    state_RSPA = state_RSPA.reshape(Nt, Nt, Nt * Nt)
    U_S2 = two_mode_squeeze(-rprime, Nt)
    state_RSPA = torch.einsum('ij,kdj->kdi', U_S2, state_RSPA)

    # measurement training
    state_RSPA = state_RSPA.reshape(-1, 1)
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPAQ = torch.kron(state_RSPA, state_Q)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_p = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_a = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_p = U_ECD_p.reshape(2, Nt, 2, Nt)
        U_ECD_a = U_ECD_a.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_p, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_a, state_RSPAQ)

    # calculate Coherent Information
    state_RSPAQ = state_RSPAQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)

    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, np_input, state_RS, state_PA


def transduction_protocol_CoherentInfo_ECD_MM_EA_test1(n_p, eta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    theta = np.arcsin(np.sqrt(eta))  # eta = sin(theta)**2
    G = n_p + 1
    Gprime = G / (G * eta + 1 - eta)
    r = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(G))))
    rprime = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(Gprime))))

    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  mode p in preparation
    alphas_a_0 = parameters[2 * depth:4 * depth]  # alphas for the  mode a in preparation
    alphas_p_1 = parameters[4 * depth:6 * depth]  # alphas for the  mode p in measurement
    alphas_a_1 = parameters[6 * depth:8 * depth]  # alphas for the  mode a in measurement
    alphas_r = parameters[8 * depth:10 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[10 * depth:12 * depth]  # alphas for mode S in preparation

    betas_p_0 = parameters[12 * depth:14 * depth]  # betas for the  mode p in preparation
    betas_a_0 = parameters[14 * depth:16 * depth]  # betas for the  mode a in preparation
    betas_p_1 = parameters[16 * depth:18 * depth]  # betas for the  mode p in measurement
    betas_a_1 = parameters[18 * depth:20 * depth]  # betas for the  mode a in measurement
    betas_r = parameters[20 * depth:22 * depth]  # betas for mode R in preparation
    betas_s = parameters[22 * depth:24 * depth]  # betas for mode R in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    state_PA = torch.zeros(Nt * Nt, 1, dtype=torch.complex128)
    state_PA[0, 0] = 1
    U_S1 = two_mode_squeeze(r, Nt)
    state_PA = U_S1 @ state_PA
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS.reshape(-1, 1), state_PA.reshape(-1, 1)).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)

    # measurement training
    state_RSPA = state_RSPA.reshape(-1, 1)
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPAQ = torch.kron(state_RSPA, state_Q)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_p = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_a = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_p = U_ECD_p.reshape(2, Nt, 2, Nt)
        U_ECD_a = U_ECD_a.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_p, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_a, state_RSPAQ)

    # calculate Coherent Information
    state_RSPAQ = state_RSPAQ.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[2])
    rho_RP = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, np_input, state_RS, state_PA


def transduction_protocol_CoherentInfo_ECD_MM_EA_test2(n_p, eta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A,4->Q
    theta = np.arcsin(np.sqrt(eta))  # eta = sin(theta)**2
    G = n_p + 1
    Gprime = G / (G * eta + 1 - eta)
    r = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(G))))
    rprime = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(Gprime))))

    # training parameters distribution for R,S, P and A
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  mode p in preparation
    alphas_a_0 = parameters[2 * depth:4 * depth]  # alphas for the  mode a in preparation
    alphas_p_1 = parameters[4 * depth:6 * depth]  # alphas for the  mode p in measurement
    alphas_a_1 = parameters[6 * depth:8 * depth]  # alphas for the  mode a in measurement
    alphas_r = parameters[8 * depth:10 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[10 * depth:12 * depth]  # alphas for mode S in preparation

    betas_p_0 = parameters[12 * depth:14 * depth]  # betas for the  mode p in preparation
    betas_a_0 = parameters[14 * depth:16 * depth]  # betas for the  mode a in preparation
    betas_p_1 = parameters[16 * depth:18 * depth]  # betas for the  mode p in measurement
    betas_a_1 = parameters[18 * depth:20 * depth]  # betas for the  mode a in measurement
    betas_r = parameters[20 * depth:22 * depth]  # betas for mode R in preparation
    betas_s = parameters[22 * depth:24 * depth]  # betas for mode R in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input state for PA
    parameters_PA_0 = torch.cat([alphas_p_0, alphas_a_0, betas_p_0, betas_a_0])
    state_PA = ECD_state_generation_MM(depth, parameters_PA_0, Nt)
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS.reshape(-1, 1), state_PA.reshape(-1, 1)).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)

    # apply anti-squeezer
    state_RSPA = state_RSPA.reshape(Nt, Nt, Nt * Nt)
    U_S2 = two_mode_squeeze(-rprime, Nt)
    state_RSPA = torch.einsum('ij,kdj->kdi', U_S2, state_RSPA)

    # calculate Coherent Information
    state_RSPA = state_RSPA.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPA, shape=[Nt, Nt, Nt, Nt], sel=[2])
    rho_RP = partial_trace_torch(state_RSPA, shape=[Nt, Nt, Nt, Nt], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, np_input, state_RS, state_PA


def transduction_protocol_CoherentInfo_TMS(n_p, eta, parameters, depth, Nt):
    # 0->R,1->S,2->P, 3->A

    theta = np.arcsin(np.sqrt(eta))  # eta = sin(theta)**2
    G = n_p + 1
    Gprime = G / (G * eta + 1 - eta)
    r = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(G))))
    rprime = torch.abs(torch.arccosh(torch.sqrt(torch.tensor(Gprime))))

    # training parameters distribution for R,S, P
    alphas_r = parameters[0 * depth:2 * depth]  # alphas for mode R in preparation
    alphas_s = parameters[2 * depth:4 * depth]  # alphas for mode S in preparation
    betas_r = parameters[4 * depth:6 * depth]  # betas for mode R in preparation
    betas_s = parameters[6 * depth:8 * depth]  # betas for mode R in preparation

    # input state for RS
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt)
    # state_RS = thermal_state_twomode_pure(n_p,Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # input TMSV state for PA
    state_PA = torch.zeros(Nt * Nt, 1, dtype=torch.complex128)
    state_PA[0, 0] = 1
    U_S1 = two_mode_squeeze(r, Nt)
    state_PA = U_S1 @ state_PA
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    state_RSPA = torch.kron(state_RS.reshape(-1, 1), state_PA.reshape(-1, 1)).reshape(Nt, Nt * Nt, Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)

    # Apply the anti-squeezer
    state_RSPA = state_RSPA.reshape(Nt, Nt, Nt * Nt)
    U_S2 = two_mode_squeeze(-rprime, Nt)
    state_RSPA = torch.einsum('ij,kdj->kdi', U_S2, state_RSPA)

    # calculate Coherent Information
    state_RSPA = state_RSPA.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSPA, shape=[Nt, Nt, Nt, Nt], sel=[2])
    rho_RP = partial_trace_torch(state_RSPA, shape=[Nt, Nt, Nt, Nt], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, np_input, state_RS, state_PA


def transduction_protocol_CoherentInfo_Tele(n_p, eta, parameters, depth, Nt):
    # 0->R,1->S,2->A, 3->P
    r = np.arcsinh(np.sqrt(n_p))
    theta = np.arcsin(np.sqrt(eta))
    U_BS = unitary_beam_splitter(-theta, Nt)

    state_A = squeezed_vacuum_state_numerical(r, torch.tensor(0), Nt)
    state_P = squeezed_vacuum_state_numerical(r, torch.tensor(np.pi), Nt)
    state_RS = ECD_state_generation_MM(depth, parameters, Nt, state_initial_MM=thermal_state_twomode_pure(n_p, Nt))
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # apply beam splitter between A and P
    state_AP = torch.kron(state_A, state_P)
    state_AP = U_BS @ state_AP

    # apply beam splitter between S and A
    state_RSAP = torch.kron(state_RS, state_AP).reshape(Nt, Nt * Nt, Nt)
    state_RSAP = torch.einsum('ij,kjd->kid', U_BS, state_RSAP)

    # Conditional q displacement
    SUM_AP_q = SUM_gate(Nt, 1 / np.sqrt(eta))
    state_RSAP = state_RSAP.reshape(Nt, Nt, Nt * Nt)
    state_RSAP = torch.einsum('ij,kdj->kdi', SUM_AP_q, state_RSAP)

    # Conditional p displacement
    SUM_SP_p = SUM_gate_var(Nt, 1 / np.sqrt(1 - eta))
    state_RSPA = torch.einsum('abcd->abdc', state_RSAP.reshape(Nt, Nt, Nt, Nt))
    state_RSPA = state_RSPA.reshape(Nt, Nt * Nt, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', SUM_SP_p, state_RSPA)
    state_RSAP = torch.einsum('abcd->abdc', state_RSPA.reshape(Nt, Nt, Nt, Nt))

    # calculate Coherent Information
    state_RSAP = state_RSAP.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSAP, shape=[Nt, Nt, Nt, Nt], sel=[3])
    rho_RP = partial_trace_torch(state_RSAP, shape=[Nt, Nt, Nt, Nt], sel=[0, 3])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_RP)

    return CI, ns_input, state_RS


def transduction_protocol_CoherentInfo_AQT(eta, parameters, depth, Nt):
    # 0->R,1->S,2->A, 3->P
    #r = np.arcsinh(np.sqrt(n_p))
    theta = np.arcsin(np.sqrt(eta))
    U_BS = unitary_beam_splitter(-theta, Nt)

    # state_P = squeezed_vacuum_state_numerical(r, torch.tensor(0), Nt)
    alphas_r = parameters[0 * depth:2 * depth]  # alphas for the  mode r
    alphas_s = parameters[2 * depth:4 * depth]  # alphas for the  mode s
    alphas_p = parameters[4 * depth:6 * depth]  # alphas for the  mode p
    betas_r = parameters[6 * depth:8 * depth]  # alphas for the  mode r
    betas_s = parameters[8 * depth:10 * depth]  # alphas for the  mode s
    betas_p = parameters[10 * depth:12 * depth]  # alphas for the  mode p
    parameters_P = torch.cat([alphas_p, betas_p])
    parameters_RS = torch.cat([alphas_r, alphas_s, betas_r, betas_s])

    # state preparation
    state_P = ECD_state_generation_M(depth, parameters_P, Nt)
    state_RS = ECD_state_generation_MM(depth, parameters_RS, Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]
    np_input = energy_n_M(state_P, Nt)

    # apply beam splitter between S and P
    state_RSP = torch.kron(state_RS, state_P).reshape(Nt, Nt * Nt)
    state_RSP = torch.einsum('ij,kj->ki', U_BS, state_RSP)

    # Conditional q displacement
    SUM_SP_q = SUM_gate(Nt, parameters[-1])
    state_RSP = state_RSP.reshape(Nt, Nt * Nt)
    state_RSP = torch.einsum('ij,kj->ki', SUM_SP_q, state_RSP)

    # calculate Coherent Information
    state_RSP = state_RSP.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSP, shape=[Nt, Nt, Nt], sel=[2])
    rho_S = partial_trace_torch(state_RSP, shape=[Nt, Nt, Nt], sel=[1])
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_S)

    return CI, np_input, state_P, ns_input, state_RS


def transduction_protocol_CoherentInfo_AQT_v2(eta, n_s, n_p, Nt):
    # 0->R,1->S,2->A, 3->P
    #r = np.arcsinh(np.sqrt(n_p))
    theta = np.arcsin(np.sqrt(eta))
    U_BS = unitary_beam_splitter(-theta, Nt)
    r = np.arcsinh(np.sqrt(n_p))

    # state preparation

    state_P =  squeezed_vacuum_state_numerical(torch.tensor(r), phi=torch.tensor(0), Nt=30)
    state_RS = thermal_state_twomode_pure(torch.tensor(n_s),Nt)
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]
    np_input = energy_n_M(state_P, Nt)

    # apply beam splitter between S and P
    state_RSP = torch.kron(state_RS, state_P).reshape(Nt, Nt * Nt)
    state_RSP = torch.einsum('ij,kj->ki', U_BS, state_RSP)

    # Conditional q displacement
    SUM_SP_q = SUM_gate(Nt)
    state_RSP = state_RSP.reshape(Nt, Nt * Nt)
    state_RSP = torch.einsum('ij,kj->ki', SUM_SP_q, state_RSP)

    # calculate Coherent Information
    state_RSP = state_RSP.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSP, shape=[Nt, Nt, Nt], sel=[2])
    rho_S = partial_trace_torch(state_RSP, shape=[Nt, Nt, Nt], sel=[1])
    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_S)

    return CI, np_input, state_P, ns_input, state_RS


def transduction_protocol_CoherentInfo_GKP(eta, d1, d2, j2, parameters, Nt):
    theta = np.arcsin(np.sqrt(eta))
    delta1, kappa1, r_hex1, phi_hex1 = parameters[0:4]
    delta2, kappa2, r_hex2, phi_hex2 = parameters[4:8]

    state_RS = torch.zeros(Nt * Nt, 1)
    for i in range(d1):
        state_R = number_state(i, Nt)
        state_S = hex_gkp_qudit(Nt, delta1, kappa1, d1, i, r_hex1, phi_hex1)
        state_RS = state_RS + torch.kron(state_R, state_S).reshape(-1,1)
    state_RS = state_RS / torch.norm(state_RS)
    state_P = hex_gkp_qudit(Nt, delta2, kappa2, d2, j2, r_hex2, phi_hex2)

    ns_input = energy_n1n2_MM(state_RS, Nt)[1]
    np_input = energy_n_M(state_P, Nt)

    state_RSP = torch.kron(state_RS.reshape(-1, 1), state_P.reshape(-1, 1)).reshape(Nt, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSP = torch.einsum('ij,kj->ki', U_BS, state_RSP)

    # calculate Coherent Information
    state_RSP = state_RSP.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSP, shape=[Nt, Nt, Nt, ], sel=[2])
    rho_S = partial_trace_torch(state_RSP, shape=[Nt, Nt, Nt], sel=[1])

    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_S)

    return CI, ns_input, np_input, state_RS, state_P


def transduction_protocol_CoherentInfo_GKP2(eta, d1, d2, j2, parameters, Nt, NR=10):
    theta = np.arcsin(np.sqrt(eta))
    delta1, r_hex1, phi1_hex, phi2_hex = parameters[0:4]
    delta2, r_hex2, phi3_hex, phi4_hex  = parameters[4:8]

    state_RS = torch.zeros(NR * Nt, 1)
    for i in range(d1):
        state_R = number_state(i, NR)
        state_S = GKP_hex_lattice_approximate(Nt, d1, i, delta1, r_hex1, phi1_hex, phi2_hex, 5)
        state_RS = state_RS + torch.kron(state_R, state_S).reshape(-1,1)
    state_RS = state_RS / torch.norm(state_RS)

    state_P = GKP_hex_lattice_approximate(Nt, d2, j2, delta2, r_hex2, phi3_hex, phi4_hex, 5)

    rho_S = partial_trace_torch(state_RS,shape=[NR,Nt],sel=[1])
    ns_input = energy_n_M(rho_S, Nt)
    np_input = energy_n_M(state_P, Nt)

    state_RSP = torch.kron(state_RS.reshape(-1, 1), state_P.reshape(-1, 1)).reshape(NR, Nt * Nt)

    # apply the beam splittr between S and P
    U_BS = unitary_beam_splitter(-theta, Nt)
    state_RSP = torch.einsum('ij,kj->ki', U_BS, state_RSP)

    # calculate Coherent Information
    state_RSP = state_RSP.reshape(-1, 1)
    rho_P = partial_trace_torch(state_RSP, shape=[NR, Nt, Nt], sel=[2])
    rho_S = partial_trace_torch(state_RSP, shape=[NR, Nt, Nt], sel=[1])

    CI = von_neumann_entropy(rho_P) - von_neumann_entropy(rho_S)

    return CI, ns_input, np_input, state_RS, state_P


def transduction_protocol_TMS_mixed_v2(rho_signal, eta, r, rprime, Nt):
    """

    :param state: the state we prepare the send in the Signal part (S)
    :param theta: the angle of beam splitter
    :param parameters: the parameters of TMS G ang G'
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """

    theta = np.arcsin(np.sqrt(eta))
    TMS0 = two_mode_squeeze(r, Nt)
    TMS1 = two_mode_squeeze(rprime, Nt)
    # initial state for A and P
    dimension = Nt * Nt
    state_PA = torch.zeros((dimension, 1), dtype=torch.complex128)
    state_PA[0, 0] = 1  # initial state

    # prepare the probe state TMS
    state_PA = TMS0 @ state_PA
    rho_PA = state_PA @ dagger(state_PA)

    # the energy of the input state
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    rho_S = rho_signal.reshape(Nt, Nt)
    # combing the signal state into the whole system, M1->S, M2->P,M2->A
    rho_SPA = torch.kron(rho_S, rho_PA)

    # apply the beam splittr between S and P
    rho_SPA = rho_SPA.reshape(Nt * Nt, Nt, Nt * Nt, Nt)
    U_BS = unitary_beam_splitter(-theta, Nt)
    rho_SPA = torch.einsum('ij,jalb,lk->iakb', U_BS, rho_SPA, dagger(U_BS))

    """
    rho_APS = rho_APS.reshape(Nt, Nt * Nt, Nt, Nt * Nt)
    U_BS = unitary_beam_splitter(theta, Nt)
    rho_APS = torch.einsum('ij,ajbl,lk->aibk', U_BS, rho_APS, dagger(U_BS))
    rho_S_save = torch.einsum('abc dbc->ad', rho_SPA.reshape(Nt, Nt, Nt, Nt, Nt, Nt))
    rho_P_save = torch.einsum('bac bdc->ad', rho_SPA.reshape(Nt, Nt, Nt, Nt, Nt, Nt))
    """

    # trace out the signal S
    rho_SPA = rho_SPA.reshape(Nt, Nt, Nt, Nt, Nt, Nt)
    rho_PA = torch.einsum('dbc dfg->bc fg', rho_SPA)

    # apply the anti-squeezer
    rho_PA = rho_PA.reshape(Nt * Nt, Nt * Nt)
    rho_PA = TMS1 @ rho_PA @ dagger(TMS1)
    rho_PA = rho_PA.reshape(Nt, Nt, Nt, Nt)

    rho_P = torch.einsum('abcb->ac', rho_PA)
    return rho_P, np_input


def transduction_protocol_TMS_mixed(rho_signal, eta, r, rprime, Nt):
    theta = np.arcsin(np.sqrt(eta))
    TMS0 = two_mode_squeeze(r, Nt)
    TMS1 = two_mode_squeeze(rprime, Nt)
    # initial state for A and P
    dimension = Nt * Nt
    state_AP = torch.zeros((dimension, 1), dtype=torch.complex128)
    state_AP[0, 0] = 1  # initial state

    # prepare the probe state TMS
    state_AP = TMS0 @ state_AP
    rho_AP = state_AP @ dagger(state_AP)

    # the energy of the input state
    np_input = energy_n1n2_MM(state_AP, Nt)[0]

    rho_S = rho_signal.reshape(Nt, Nt)
    # combing the signal state into the whole system, M1->A, M2->P,M2->S
    rho_APS = torch.kron(rho_AP, rho_S)

    # apply the beam splittr between S and P
    rho_APS = rho_APS.reshape(Nt, Nt * Nt, Nt, Nt * Nt)
    U_BS = unitary_beam_splitter(theta, Nt)
    rho_APS = torch.einsum('ij,ajbl,lk->aibk', U_BS, rho_APS, dagger(U_BS))

    """
    rho_S_save = torch.einsum('bca bcd->ad', rho_APS.reshape(Nt, Nt, Nt, Nt, Nt, Nt))
    rho_P_save = torch.einsum('bac bdc->ad', rho_APS.reshape(Nt, Nt, Nt, Nt, Nt, Nt))
    """

    rho_APS = rho_APS.reshape(Nt, Nt, Nt, Nt, Nt, Nt)

    # trace out the signal S
    rho_APS = rho_APS.reshape(Nt, Nt, Nt, Nt, Nt, Nt)
    rho_AP = torch.einsum('bcd fgd->bc fg', rho_APS)

    # apply the anti-squeezer
    rho_AP = rho_AP.reshape(Nt * Nt, Nt * Nt)
    rho_AP = TMS1 @ rho_AP @ dagger(TMS1)
    rho_AP = rho_AP.reshape(Nt, Nt, Nt, Nt)

    rho_P = torch.einsum('babc->ac', rho_AP)
    return rho_P, np_input


#####################################################################################################################
# Calculate Entanglement Fidelity with TMSV state as fixed input in RS
#####################################################################################################################
def transduction_protocol_TMSVEF_ECD_MM(r, theta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 3->A, 4->Q

    # training parameters distribution for Q, A and P
    alphas_a_0 = parameters[0 * depth:2 * depth]  # alphas for the  mode a in ECD0
    alphas_p_0 = parameters[2 * depth:4 * depth]  # alphas for the  mode p in ECD0
    alphas_a_1 = parameters[4 * depth:6 * depth]  # alphas for the  mode a in ECD1
    alphas_p_1 = parameters[6 * depth:8 * depth]  # alphas for the  mode p in ECD1

    betas_a_0 = parameters[8 * depth:10 * depth]  # betas for the  mode a in ECD0
    betas_p_0 = parameters[10 * depth:12 * depth]  # betas for the  mode p in ECD0
    betas_a_1 = parameters[12 * depth:14 * depth]  # betas for the  mode a in ECD1
    betas_p_1 = parameters[14 * depth:16 * depth]  # betas for the  mode p in ECD1

    # TMSV state of S and R
    state_RS = torch.zeros(Nt * Nt, 1, dtype=torch.complex128)
    state_RS[0, 0] = 1
    U_TMS = two_mode_squeeze(r, Nt)
    state_RS = U_TMS @ state_RS
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # initial state for  P and A
    parameters_PA_0 = torch.cat([alphas_p_0, alphas_a_0, betas_p_0, betas_a_0])
    state_PA = ECD_state_generation_MM(depth, parameters_PA_0, Nt)
    np_input = energy_n1n2_MM(state_PA, Nt)[0]

    # apply the beam splittr between S and P
    state_RSPA = torch.kron(state_RS, state_PA).reshape(Nt, Nt * Nt, Nt)
    U_BS = unitary_beam_splitter(theta, Nt)
    state_RSPA = torch.einsum('ij,kjd->kid', U_BS, state_RSPA)
    state_RSPA = state_RSPA.reshape(-1, 1)

    # measurement training
    state_Q = torch.zeros((2, 1), dtype=torch.complex128)
    state_Q[0, 0] = 1
    state_RSPAQ = torch.kron(state_RSPA, state_Q)
    state_RSPAQ = state_RSPAQ.reshape(Nt, Nt, Nt, Nt, 2)

    for i in range(depth):
        U_ECD_1 = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_2 = ECD_unitary(alpha1=alphas_a_1[2 * i], alpha2=alphas_a_1[2 * i + 1], psi=betas_a_1[2 * i],
                              theta=betas_a_1[2 * i + 1], Nt=Nt)
        U_ECD_1 = U_ECD_1.reshape(2, Nt, 2, Nt)
        U_ECD_2 = U_ECD_2.reshape(2, Nt, 2, Nt)
        state_RSPAQ = torch.einsum('abcd, hgdfc-> hgbfa', U_ECD_1, state_RSPAQ)
        state_RSPAQ = torch.einsum('abcd, hgfdc-> hgfba', U_ECD_2, state_RSPAQ)

    # calculate the entanglement fidelity
    state_RSPAQ = state_RSPAQ.reshape(-1, 1)
    rho_RP = partial_trace_torch(state_RSPAQ, shape=[Nt, Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    ef = (dagger(state_RS) @ rho_RP @ state_RS)[0, 0]

    return torch.real(ef), ns_input, np_input, state_RS, state_PA


def transduction_protocol_TMSVEF_ECD_QM(r, theta, parameters, depth, Nt):
    """
    :param r: squeezing parameter for TMS
    :param theta: the angle of beam splitter
    :param parameters: the parameters needed to train the ECD circuit
    :param depth:ECD circuit depth,
    :param Nt:truncation dimension of the boson
    :return: the final state in probe  (P)
    """
    # 0->R, 1->S,2->P, 4->Q

    # training parameters distribution for Q, A and P
    alphas_p_0 = parameters[0 * depth:2 * depth]  # alphas for the  mode p in ECD0
    alphas_p_1 = parameters[2 * depth:4 * depth]  # alphas for the  mode p in ECD1
    betas_p_0 = parameters[4 * depth:6 * depth]  # betas for the  mode p in ECD0
    betas_p_1 = parameters[6 * depth:8 * depth]  # betas for the  mode p in ECD1

    # TMSV state of S and R
    state_RS = torch.zeros(Nt * Nt, 1, dtype=torch.complex128)
    state_RS[0, 0] = 1
    U_TMS = two_mode_squeeze(r, Nt)
    state_RS = U_TMS @ state_RS
    ns_input = energy_n1n2_MM(state_RS, Nt)[1]

    # initial state for  P and Q
    parameters_P_0 = torch.cat([alphas_p_0, betas_p_0, ])
    state_QP = ECD_state_generation_QM(depth, parameters_P_0, Nt).reshape(2, Nt)
    state_PQ = torch.einsum('ij>ji', state_QP)
    np_input = energy_n_QM(state_QP, Nt)

    # apply the beam splittr between S and P
    state_RSPQ = torch.kron(state_RS, state_PQ).reshape(Nt, Nt * Nt, 2)
    U_BS = unitary_beam_splitter(theta, Nt)
    state_RSPQ = torch.einsum('ij,kjd->kid', U_BS, state_RSPQ)

    # measurement training
    state_RSPQ = state_RSPQ.reshape(Nt, Nt, Nt, 2)
    for i in range(depth):
        U_ECD_p = ECD_unitary(alpha1=alphas_p_1[2 * i], alpha2=alphas_p_1[2 * i + 1], psi=betas_p_1[2 * i],
                              theta=betas_p_1[2 * i + 1], Nt=Nt)
        U_ECD_p = U_ECD_p.reshape(2, Nt, 2, Nt)
        state_RSPQ = torch.einsum('abcd, hgdc-> hgba', U_ECD_p, state_RSPQ)

    # calculate the entanglement fidelity
    state_RSPQ = state_RSPQ.reshape(-1, 1)
    rho_RP = partial_trace_torch(state_RSPQ, shape=[Nt, Nt, Nt, 2], sel=[0, 2])
    rho_RP = rho_RP.reshape(Nt * Nt, Nt * Nt)
    ef = (dagger(state_RS) @ rho_RP @ state_RS)[0, 0]

    return torch.real(ef), ns_input, np_input, state_RS, state_QP
