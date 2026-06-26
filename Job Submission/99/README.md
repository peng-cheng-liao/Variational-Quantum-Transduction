# Job 99: Initial-P/A thermal noisy VQT eta scans

Job 99 is a self-contained HPC job for noisy VQT coherent-information
evaluation with local `QTorch`, local run-84 parameter files, a Slurm array
script, and local text/config outputs. It uses:

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

## Scan

All cases use the same loss setting:

```text
tau_s = tau_p = tau_A = 0.99
kappa_o = kappa_m = kappa_a = 0.99
```

Eta grid:

```text
eta = 0.05, 0.10, ..., 0.95
```

Three initial thermal settings are evaluated in the active run:

```text
n_P^th = n_A^th = 0.03
n_P^th = n_A^th = 0.05
n_P^th = n_A^th = 0.07
```

Outputs are written under the job-local data folder:

```text
Job Submission/99/Data
```

On HPC this is:

```text
/home1/liaopeng/QuantumTransduction/99/Data
```

Existing data in `Data` from previous Job 99 scans are intentionally
preserved and should not be deleted. New runs use distinct case subfolders.

Case subfolders:

```text
Data/nthP_0p03_nthA_0p03_tauAll_0p99/eta=0.05/
Data/nthP_0p05_nthA_0p05_tauAll_0p99/eta=0.05/
Data/nthP_0p07_nthA_0p07_tauAll_0p99/eta=0.05/
```

Each eta folder contains:

```text
best_feasible_ci.txt
noise_config.json
source_parameter_file.txt
```

## Commands

Run local checks from this folder:

```bash
cd "Job Submission/99"
```

List cases:

```bash
python calculate_noisy_vqt_99.py --list-cases
```

Dry run for case 0 and scan index 0:

```bash
python calculate_noisy_vqt_99.py --dry-run --case-index 0 --scan-index 0 --output-root Data
```

One recompute task:

```bash
python calculate_noisy_vqt_99.py --case-index 0 --scan-index 0 --output-root Data --recompute
```

Summarize outputs:

```bash
python process_noisy_vqt_99.py
```

## HPC Submission

The Slurm script is intended for this HPC deployment path:

```text
/home1/liaopeng/QuantumTransduction/99
```

Submit from HPC with:

```bash
cd /home1/liaopeng/QuantumTransduction/99
mkdir -p logs Data
sbatch submit_noisy_vqt_99.sbatch
```

The script uses absolute HPC paths for `--chdir`, stdout, stderr, and
`--output-root`, so raw outputs are saved to:

```text
/home1/liaopeng/QuantumTransduction/99/Data
```

Slurm stdout/stderr files are written to:

```text
/home1/liaopeng/QuantumTransduction/99/logs
```

## Runtime Notes

The full scan has 57 tasks:

```text
3 thermal settings x 19 eta points
```

The Slurm array maps each `SLURM_ARRAY_TASK_ID` to an explicit `case_index`
and `scan_index`, then calls `calculate_noisy_vqt_99.py`.
