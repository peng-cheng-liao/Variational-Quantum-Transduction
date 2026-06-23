# Job 95: Auxiliary-loss noisy VQT sweep

This local job mirrors `Job Submission/92` for the run-84 non-adaptive VQT-EA
parameters, but evaluates the repository-level
`transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise` function with an
explicit auxiliary-loss sweep.

The default scan uses the same eta grid as Job 92:

```text
eta = 0.05, 0.10, ..., 0.95
```

For every eta, Job 95 evaluates:

```text
kappa_a = 1.0, 0.95, 0.9
```

Fixed noise settings:

```text
initial_p_thermal_nbar = 0.1
kappa_o = 0.99
kappa_m = 0.99
n_o = n_m = n_a = 0.0
```

Inputs are read from `Job Submission/92/parameters`, preserving Job 92's local
parameter-loading style. Outputs are written under `Data_HPC/95`.

This folder intentionally uses the repository `QTorch` package directly and
does not copy a local `QTorch` folder or add an sbatch script for local Mac
runs.

## Commands

Dry run:

```bash
python "Job Submission/95/calculate_95.py" --dry-run
```

One-task smoke test:

```bash
python "Job Submission/95/calculate_95.py" --eta-index 0 --kappa-a 1.0 --max-items 1 --nt 8 --num-threads 4 --overwrite
```

Full intended local scan:

```bash
python "Job Submission/95/calculate_95.py" --num-threads 4
```

Summarize outputs:

```bash
python "Job Submission/95/process_95.py"
```

Run one eta for all auxiliary-loss values:

```bash
python "Job Submission/95/calculate_95.py" --eta-index 0 --num-threads 4
```

Run one auxiliary-loss value for all eta:

```bash
python "Job Submission/95/calculate_95.py" --kappa-a 0.95 --num-threads 4
```

## Runtime Notes

The full scan is heavier than Job 92 by roughly a factor of three because it
sweeps three `kappa_a` values.

If the updated thermal-noise function includes A-loss Kraus branching, branch
count scales approximately as:

```text
N_thermal * N_Kraus(S) * N_Kraus(P) * N_Kraus(A)
```

For `kappa_a=1.0`, the A-loss branch should be identity-like and cheap. For
`kappa_a=0.95` or `kappa_a=0.9`, A-loss can noticeably increase runtime.

On a MacBook Air with 8GB RAM, start with the smoke-test command before
launching the full scan.
