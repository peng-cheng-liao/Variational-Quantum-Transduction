import math
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QTorch.Transduction import gaussian_thermal_noise_capacity_lower_bound


def _g(n):
    if n == 0:
        return 0.0
    return (n + 1.0) * math.log2(n + 1.0) - n * math.log2(n)


def _assert_close(name, actual, expected, atol=1e-8, rtol=1e-8):
    if not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{name} mismatch: {actual} vs {expected}")


def test_pure_loss_asymptotic_limit():
    large_ns = 1.0e6
    transmissive = gaussian_thermal_noise_capacity_lower_bound(
        large_ns,
        eta=0.75,
        kappa_p=1.0,
        n_p_env=0.0,
        num_grid=501,
    )
    expected = math.log2(0.75 / 0.25)
    _assert_close("pure-loss asymptotic lower bound", transmissive, expected, atol=2e-5, rtol=2e-5)

    antidegradable = gaussian_thermal_noise_capacity_lower_bound(
        large_ns,
        eta=0.5,
        kappa_p=1.0,
        n_p_env=0.0,
        num_grid=501,
    )
    _assert_close("antidegradable pure-loss lower bound", antidegradable, 0.0, atol=1e-8, rtol=0.0)


def test_identity_finite_energy_limit():
    n_s = 2.5
    value = gaussian_thermal_noise_capacity_lower_bound(
        n_s,
        eta=1.0,
        kappa_p=1.0,
        n_p_env=0.0,
    )
    _assert_close("identity finite-energy lower bound", value, _g(n_s), atol=1e-12, rtol=1e-12)


def test_complete_loss_limit():
    value_eta = gaussian_thermal_noise_capacity_lower_bound(
        3.0,
        eta=0.0,
        kappa_p=1.0,
        n_p_env=0.0,
    )
    details_kappa = gaussian_thermal_noise_capacity_lower_bound(
        3.0,
        eta=0.8,
        kappa_p=0.0,
        n_p_env=0.2,
        return_details=True,
    )
    _assert_close("eta complete-loss lower bound", value_eta, 0.0)
    _assert_close("kappa complete-loss lower bound", details_kappa["capacity_lower_bound"], 0.0)
    _assert_close("kappa complete-loss bath occupancy", details_kappa["N_B_eff"], 0.2)


def test_s_output_noise_independence_and_details():
    clean_s = gaussian_thermal_noise_capacity_lower_bound(
        4.0,
        eta=0.8,
        kappa_s=1.0,
        n_s_env=0.0,
        kappa_p=0.95,
        n_p_env=0.02,
        return_details=True,
    )
    noisy_s = gaussian_thermal_noise_capacity_lower_bound(
        4.0,
        eta=0.8,
        kappa_s=0.4,
        n_s_env=10.0,
        kappa_p=0.95,
        n_p_env=0.02,
        return_details=True,
    )

    _assert_close(
        "S-output noise independence",
        clean_s["capacity_lower_bound"],
        noisy_s["capacity_lower_bound"],
        atol=1e-12,
        rtol=1e-12,
    )
    _assert_close("effective transmissivity", clean_s["T_eff"], 0.8 * 0.95)
    expected_N_B = (1.0 - 0.95) * 0.02 / (1.0 - 0.8 * 0.95)
    _assert_close("effective bath occupancy", clean_s["N_B_eff"], expected_N_B)
    if clean_s["s_noise_affects_single_output_channel"]:
        raise AssertionError("S-output noise should not affect the single-output channel")


def test_thermal_noise_monotonicity_sanity_check():
    low_noise = gaussian_thermal_noise_capacity_lower_bound(
        5.0,
        eta=0.8,
        kappa_p=0.9,
        n_p_env=0.0,
        num_grid=701,
    )
    high_noise = gaussian_thermal_noise_capacity_lower_bound(
        5.0,
        eta=0.8,
        kappa_p=0.9,
        n_p_env=0.5,
        num_grid=701,
    )
    if high_noise > low_noise + 1e-10:
        raise AssertionError(f"thermal noise increased lower bound: {high_noise} > {low_noise}")


def test_validation():
    invalid_kwargs = [
        {"n_s": -1.0, "eta": 0.5},
        {"n_s": 1.0, "eta": 1.1},
        {"n_s": 1.0, "eta": 0.5, "kappa_s": -0.1},
        {"n_s": 1.0, "eta": 0.5, "n_s_env": -0.1},
        {"n_s": 1.0, "eta": 0.5, "kappa_p": 1.1},
        {"n_s": 1.0, "eta": 0.5, "n_p_env": -0.1},
        {"n_s": 1.0, "eta": 0.5, "num_grid": 1},
    ]
    for kwargs in invalid_kwargs:
        try:
            gaussian_thermal_noise_capacity_lower_bound(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


if __name__ == "__main__":
    test_pure_loss_asymptotic_limit()
    test_identity_finite_energy_limit()
    test_complete_loss_limit()
    test_s_output_noise_independence_and_details()
    test_thermal_noise_monotonicity_sanity_check()
    test_validation()
    print("Gaussian capacity lower-bound smoke tests passed")
