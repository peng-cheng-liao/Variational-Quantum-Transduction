# Job 97: eta-uncertainty robust coherent information

Job 97 is a local Mac post-evaluation workflow for calibration mismatch in the transduction efficiency. It does not train new VQT or GKP parameters and does not need an HPC submission script.

The diagnostic uses a nominal design efficiency `eta0`, a true physical efficiency `eta_t`, and endpoint uncertainty `eta_t = eta0 +/- delta`. For VQT, GKP, and TMS-EA, protocol settings are chosen at `eta0`, frozen, and evaluated at the two true endpoints. The reported average robust coherent information is

```text
I_avg(eta0, delta) = [I(eta0 + delta | eta0) + I(eta0 - delta | eta0)] / 2.
```

Schemes:

- `VQT`: loads the main non-adaptive VQT selected parameters from `Data_HPC/84`.
- `GKP`: loads corrected selected GKP parameters from `Data_HPC/94/Data-Download` when available, with fallbacks in the calculation script.
- `TMS-EA`: freezes the anti-squeezer gain calibrated at nominal `eta0` and evaluates the mismatched Gaussian covariance matrix at true `eta_t`.
- `QT`: uses the energy-constrained pure-loss capacity formula at the true transmissivity because there is no finite nominal parameter to freeze.

Run the calculation from the repository root:

```bash
python3 "Job Submission/97/calculate_eta_uncertainty_robust_ci.py"
```

Useful options:

```bash
python3 "Job Submission/97/calculate_eta_uncertainty_robust_ci.py" --schemes TMS-EA QT
python3 "Job Submission/97/calculate_eta_uncertainty_robust_ci.py" --deltas 0.01 0.02
python3 "Job Submission/97/calculate_eta_uncertainty_robust_ci.py" --etas 0.05 0.10 0.15
python3 "Job Submission/97/calculate_eta_uncertainty_robust_ci.py" --output Data_HPC/97
```

Outputs are written to `Data_HPC/97` by default:

- `robust_ci_summary.csv`
- `robust_ci_summary.json`
- `config.json`

After data are available, run the plot script:

```bash
python3 Paperfig/CI_eta_uncertainty_robust.py
```

The plot script reads `Data_HPC/97/robust_ci_summary.csv` and saves:

- `Figs/CI_eta_uncertainty_robust.jpg`
- `Figs/CI_eta_uncertainty_robust.pdf`
