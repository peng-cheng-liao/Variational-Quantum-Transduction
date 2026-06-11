import inspect
import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QTorch.Transduction import (
    transduction_protocol_CoherentInfo_ECD_MM_EA,
    transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise,
)


def _parameters(depth, scale=0.02, seed=19):
    torch.manual_seed(seed)
    return (scale * torch.randn(24 * depth, dtype=torch.float64)).requires_grad_(True)


def _assert_close(name, actual, expected, atol=1e-8, rtol=1e-8):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name} mismatch: {actual} vs {expected}")


def _assert_density_matrix(name, rho, trace_tol=1e-8, eig_tol=1e-8):
    _assert_close(f"{name} trace", torch.trace(rho).real, torch.tensor(1.0, dtype=rho.real.dtype), atol=trace_tol)
    _assert_close(f"{name} Hermiticity", rho, rho.conj().T, atol=trace_tol)

    min_eval = torch.linalg.eigvalsh(rho).real.min()
    if min_eval < -eig_tol:
        raise AssertionError(f"{name} is not positive semidefinite: min eigenvalue={min_eval}")


def test_noiseless_equivalence():
    depth = 1
    Nt = 3
    eta = 0.25
    parameters = _parameters(depth)

    old = transduction_protocol_CoherentInfo_ECD_MM_EA(eta, parameters, depth, Nt)[0]
    new = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
        eta,
        parameters,
        depth,
        Nt,
        kappa_o=1.0,
        n_o=0.0,
        kappa_m=1.0,
        n_m=0.0,
    )[0]

    _assert_close("noiseless coherent information", new, old, atol=1e-10, rtol=1e-10)


def test_noisy_debug_reduced_density_matrices():
    depth = 1
    Nt = 2
    eta = 0.25
    parameters = _parameters(depth, seed=23)

    CI, _, _, _, _, debug = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
        eta,
        parameters,
        depth,
        Nt,
        initial_p_thermal_nbar=0.0,
        kappa_o=1.0,
        n_o=0.0,
        kappa_m=0.92,
        n_m=0.0,
        env_cutoff_o=Nt,
        env_cutoff_m=Nt,
        return_debug=True,
    )

    if not torch.isfinite(CI):
        raise AssertionError("coherent information is not finite")
    if debug["branch_count"] <= 0:
        raise AssertionError("no Kraus branches were accumulated")

    _assert_density_matrix("rho_P", debug["rho_P"])
    _assert_density_matrix("rho_RP", debug["rho_RP"])


def test_zero_initial_thermal_branch_matches_noiseless_with_debug():
    depth = 1
    Nt = 3
    eta = 0.25
    parameters = _parameters(depth, seed=29)

    old = transduction_protocol_CoherentInfo_ECD_MM_EA(
        eta,
        parameters,
        depth,
        Nt,
    )[0]
    new = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
        eta,
        parameters,
        depth,
        Nt,
        initial_p_thermal_nbar=0.0,
        kappa_o=1.0,
        n_o=0.0,
        kappa_m=1.0,
        n_m=0.0,
        return_debug=True,
    )[0]

    _assert_close("zero-thermal branch coherent information", new, old, atol=1e-10, rtol=1e-10)


def test_initial_thermal_branch_trace_preservation():
    depth = 1
    Nt = 3
    eta = 0.25
    parameters = _parameters(depth, seed=30)

    _, _, _, _, state_PA, debug = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
        eta,
        parameters,
        depth,
        Nt,
        initial_p_thermal_nbar=0.03,
        kappa_o=0.99,
        n_o=0.0,
        kappa_m=0.99,
        n_m=0.0,
        return_debug=True,
    )

    if state_PA is not None:
        raise AssertionError("state_PA_return should be None for a mixed initial thermal P state")
    if debug["model"] != "initial_P_thermal_branches_plus_output_pure_loss":
        raise AssertionError("unexpected debug model label")

    _assert_density_matrix("thermal rho_P", debug["rho_P"], trace_tol=1e-7)
    _assert_density_matrix("thermal rho_RP", debug["rho_RP"], trace_tol=1e-7)


def test_backward_pass():
    depth = 1
    Nt = 2
    eta = 0.25
    parameters = _parameters(depth, seed=31)

    CI = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
        eta,
        parameters,
        depth,
        Nt,
        initial_p_thermal_nbar=0.02,
        kappa_o=1.0,
        n_o=0.0,
        kappa_m=0.95,
        n_m=0.0,
        env_cutoff_o=Nt,
        env_cutoff_m=Nt,
    )[0]
    (-CI).backward()

    if parameters.grad is None:
        raise AssertionError("missing gradient")
    if not torch.isfinite(parameters.grad).all():
        raise AssertionError("gradient contains NaN or Inf")


def test_production_path_does_not_build_full_density_matrix():
    source = inspect.getsource(transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise)
    forbidden = [
        "rho_RSPA = state_RSPA @ dagger(state_RSPA)",
        "rho_RSPAQ = torch.kron",
        "apply_thermal_noisy_transducer(",
        "thermal_loss_kraus_operators(",
    ]
    for text in forbidden:
        if text in source:
            raise AssertionError(f"production thermal path still contains full-density code: {text}")


if __name__ == "__main__":
    test_noiseless_equivalence()
    test_noisy_debug_reduced_density_matrices()
    test_zero_initial_thermal_branch_matches_noiseless_with_debug()
    test_initial_thermal_branch_trace_preservation()
    test_backward_pass()
    test_production_path_does_not_build_full_density_matrix()
    print("thermal-noise memory-saving smoke tests passed")
