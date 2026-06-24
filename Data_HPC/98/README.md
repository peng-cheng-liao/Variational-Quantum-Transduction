# Job 98 Data

Fixed-parameter VQT eta-uncertainty evaluation at `eta0 = 0.30`.

- The parameter set is selected from `Data_HPC/84/eta=0.30/parameters_best_feasible.npy`.
- The selected run-84 seed is recorded in `config.json` and in the CSV source fields.
- The parameter set is not reoptimized for shifted eta values.
- Deltas are `0.01, 0.02, ..., 0.10`.
- `minus` means `CI(eta0 - delta)` evaluated with the fixed eta0 parameter set.
- `plus` means `CI(eta0 + delta)` evaluated with the fixed eta0 parameter set.
