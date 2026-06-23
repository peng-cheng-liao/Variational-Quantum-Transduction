# Job 95: Self-contained auxiliary-loss noisy VQT sweep

Job 95 mirrors the structure of `Job Submission/92`. It is self-contained with a
local `QTorch` copy, local run-84 parameter files copied from Job 92, a Slurm
array script, and Job-92-style text/config outputs.

The calculation evaluates the run-84 non-adaptive VQT-EA parameters using:

```text
transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise
```

with the updated auxiliary-loss arguments for mode A.

## Scan

Eta grid:

```text
eta = 0.05, 0.10, ..., 0.95
```

Auxiliary-loss sweep:

```text
kappa_A in {1.0, 0.95, 0.9}
```

Fixed noise settings:

```text
n_th^P = 0.1
kappa_S = kappa_P = 0.99
n_o = n_m = n_a = 0.0
```

Outputs are written under:

```text
Data_HPC/95
```

## Commands

Run from this folder:

```bash
cd "Job Submission/95"
```

List setup presets:

```bash
python calculate_noisy_vqt_95.py --list-setups
```

Dry run for setup 0 and eta index 0:

```bash
python calculate_noisy_vqt_95.py --dry-run --eta-index 0 --setup-index 0
```

One recompute task:

```bash
python calculate_noisy_vqt_95.py --setup-index 0 --eta-index 0 --recompute
```

Submit the full 57-task Slurm array:

```bash
sbatch submit_noisy_vqt_95.sbatch
```

Summarize outputs:

```bash
python process_noisy_vqt_95.py
```

## Runtime Notes

The full scan has 57 tasks: 3 auxiliary-loss setup presets times 19 eta values.

For `kappa_A=1.0`, A-loss should be identity-like and cheap. For
`kappa_A=0.95` and `kappa_A=0.9`, A-loss adds Kraus branching and may be slower.

The branch count scales roughly as:

```text
N_thermal * N_Kraus(S) * N_Kraus(P) * N_Kraus(A)
```
