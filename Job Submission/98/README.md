# Job 98: fixed-parameter VQT eta-uncertainty scan

Job 98 evaluates the VQT scheme only. It freezes the run-84 VQT parameter set selected at `eta0 = 0.30` and evaluates coherent information at shifted physical efficiencies `eta0 - delta` and `eta0 + delta`.

No retraining or re-optimization is performed for shifted eta values.

Run from the repository root:

```bash
python3 "Job Submission/98/calculate_fixed_eta0_vqt_uncertainty.py"
```

Default outputs are written to `Data_HPC/98`:

- `vqt_eta_uncertainty_fixed_eta0_0p30.csv`
- `vqt_eta_uncertainty_fixed_eta0_0p30.json`
- `vqt_eta_uncertainty_fixed_eta0_0p30.npz`
- `config.json`
- `README.md`

The delta grid is `0.01, 0.02, ..., 0.10`. The `minus` branch is `CI(eta0 - delta)` and the `plus` branch is `CI(eta0 + delta)`, both using the same fixed eta0 parameter set.

Additional delta ranges can be saved under the same output directory without overwriting existing files by using a distinct output stem:

```bash
python3 "Job Submission/98/calculate_fixed_eta0_vqt_uncertainty.py" \
  --deltas 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 \
  --output-stem vqt_eta_uncertainty_fixed_eta0_0p30_delta_0p11_0p20
```
