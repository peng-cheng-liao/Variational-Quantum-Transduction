import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QTorch.Basic import apply_thermal_noisy_transducer, thermal_probs
from QTorch.Transduction import (
    transduction_protocol_CoherentInfo_ECD_MM_EA,
    transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise,
)


def _assert_close(name, actual, expected, atol=1e-8, rtol=1e-8):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name} mismatch: {actual} vs {expected}")


def test_thermal_probs():
    probs = thermal_probs(0.0, 4, dtype=torch.float64)
    _assert_close(
        "vacuum thermal probabilities",
        probs,
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=probs.dtype),
    )
    probs = thermal_probs(torch.tensor(0.05, dtype=torch.float32), 5)
    _assert_close("thermal probability normalization", probs.sum(), torch.tensor(1.0, dtype=probs.dtype))


def test_noiseless_protocol_equivalence():
    torch.manual_seed(7)
    depth = 1
    Nt = 3
    eta = 0.25
    parameters = (0.05 * torch.randn(24 * depth, dtype=torch.float64)).requires_grad_(True)

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


def test_noisy_transducer_density_properties():
    Nt = 3
    dims = [Nt, Nt, Nt]
    state = torch.zeros((Nt ** 3, 1), dtype=torch.complex128)
    state[0, 0] = 1.0
    rho = state @ state.conj().T

    rho_out = apply_thermal_noisy_transducer(
        rho,
        eta=0.25,
        dims=dims,
        kappa_o=0.97,
        n_o=0.01,
        kappa_m=0.90,
        n_m=0.05,
        env_cutoff_o=Nt,
        env_cutoff_m=Nt,
    )

    _assert_close("trace", torch.trace(rho_out), torch.tensor(1.0, dtype=torch.complex128))
    _assert_close("Hermiticity", rho_out, rho_out.conj().T)

    min_eval = torch.linalg.eigvalsh(rho_out).real.min()
    if min_eval < -1e-8:
        raise AssertionError(f"rho_out is not positive semidefinite: min eigenvalue={min_eval}")


def test_noisy_protocol_backward():
    torch.manual_seed(11)
    depth = 1
    Nt = 3
    eta = 0.25
    parameters = (0.02 * torch.randn(24 * depth, dtype=torch.float64)).requires_grad_(True)

    CI, _, _, _, _ = transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise(
        eta,
        parameters,
        depth,
        Nt,
        initial_p_thermal_nbar=0.02,
        kappa_o=0.99,
        n_o=0.0,
        kappa_m=0.95,
        n_m=0.0,
        env_cutoff_o=Nt,
        env_cutoff_m=Nt,
    )
    loss = -CI
    loss.backward()

    if parameters.grad is None:
        raise AssertionError("missing gradient")
    if not torch.isfinite(parameters.grad).all():
        raise AssertionError("gradient contains NaN or Inf")


if __name__ == "__main__":
    test_thermal_probs()
    test_noiseless_protocol_equivalence()
    test_noisy_transducer_density_properties()
    test_noisy_protocol_backward()
    print("thermal-noise VQT smoke tests passed")
