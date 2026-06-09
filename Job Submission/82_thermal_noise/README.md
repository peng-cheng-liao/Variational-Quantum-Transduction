# Job 82 Thermal Noise

This folder is the thermal-noise extension of `Job Submission/82`.

Numerical mode implemented here:

- Mode 2: noise-aware retraining. `train_ci.py` optimizes the VQT parameters directly with `transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise`.

Mode 1, fixed-protocol robustness, is not launched here. It can be added by loading noiseless optimized parameters from `Job Submission/82` and evaluating the same thermal protocol without optimizer steps.

Noise scan controls:

- `SCAN_MODE=nm` scans `N_M_LIST=(0.00 0.01 0.03 0.05 0.10)` with `KAPPA_M=1.00`.
- `SCAN_MODE=km` scans `KAPPA_M_LIST=(1.00 0.99 0.97 0.95 0.90)` with `N_M=0.00`.
- `KAPPA_O=1.00` and `N_O=0.00` by default.

The training script encodes `eta`, `n_s`, `n_p`, `kappa_o`, `n_o`, `kappa_m`, `n_m`, cutoff, and seed in result file prefixes.
