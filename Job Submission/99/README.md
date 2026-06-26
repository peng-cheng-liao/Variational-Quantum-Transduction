# Job 99: Initial-P/A thermal noisy VQT evaluation

Job 99 mirrors the self-contained Job 95 workflow with local `QTorch`, local
run-84 parameter files, a Slurm array script, and text/config outputs. It uses:

```text
transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise
```

with separate initial thermal occupations for microwave-side modes P and A:

```text
initial_p_thermal_nbar = n_P^th
initial_a_thermal_nbar = n_A^th
```

The output/environment occupations stay fixed to pure loss:

```text
n_o = n_m = n_a = 0.0
```

Do not use `n_a` for the initial thermal state of A.

## Cases

Eta grid for cases 1 and 2:

```text
eta = 0.05, 0.10, ..., 0.95
```

Tau-A grid for cases 3 and 4:

```text
tau_A = 1.00, 0.99, 0.98, ..., 0.80
```

Four cases are written under independent subfolders in `Data_HPC/99`:

```text
case1_eta_scan_nthP_0_nthA_0_tauA_0p90
case2_eta_scan_nthP_0p1_nthA_0p1_tauA_0p90
case3_tauA_scan_eta_0p60_nthP_0_nthA_0
case4_tauA_scan_eta_0p60_nthP_0p1_nthA_0p1
```

Fixed settings:

```text
kappa_S = kappa_P = 0.99
n_o = n_m = n_a = 0.0
```

## Commands

Run from this folder:

```bash
cd "Job Submission/99"
```

List cases:

```bash
python calculate_noisy_vqt_99.py --list-cases
```

Dry run for case 0 and scan index 0:

```bash
python calculate_noisy_vqt_99.py --dry-run --case-index 0 --scan-index 0
```

One recompute task:

```bash
python calculate_noisy_vqt_99.py --case-index 0 --scan-index 0 --recompute
```

Submit the full 80-task Slurm array:

```bash
sbatch submit_noisy_vqt_99.sbatch
```

Summarize outputs:

```bash
python process_noisy_vqt_99.py
```

## Runtime Notes

The full scan has 80 tasks:

```text
19 eta points + 19 eta points + 21 tau_A points + 21 tau_A points
```

The Slurm array maps each `SLURM_ARRAY_TASK_ID` to an explicit
`case_index` and `scan_index`, then calls `calculate_noisy_vqt_99.py`.
