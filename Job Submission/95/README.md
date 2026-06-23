# Job 95: Local noisy VQT evaluation with auxiliary loss

This local job evaluates the run-84 non-adaptive VQT-EA parameters with the updated
`transduction_protocol_CoherentInfo_ECD_MM_EA_thermal_noise` function from the
repository-level `QTorch/Transduction.py`.

Differences from `Job Submission/92`:

- uses the repository `QTorch` package directly;
- does not include an sbatch script;
- does not copy a local `QTorch` folder;
- reads existing run-84 parameter files from `Data_HPC/84`;
- writes outputs under `Data_HPC/95`;
- explicitly includes pure loss on auxiliary mode A with `kappa_a` and `n_a`.

The default noisy setup mirrors job 92's main noisy case and adds auxiliary loss:

- `initial_p_thermal_nbar = 0.1`
- `kappa_o = 0.99`
- `kappa_m = 0.99`
- `kappa_a = 0.99`
- `n_o = n_m = n_a = 0.0`

Job 92 also compared lower initial thermal occupations and a noiseless reference.
By default this local script mirrors that full four-preset sweep. Use `--setup`
to restrict a run to one preset.

## Commands

Dry run:

```bash
python "Job Submission/95/calculate_95.py" --dry-run
```

One-task smoke test:

```bash
python "Job Submission/95/calculate_95.py" --eta-index 0 --max-items 1 --nt 8 --num-threads 4 --overwrite
```

Intended local run for the full job-92-style sweep:

```bash
python "Job Submission/95/calculate_95.py" --num-threads 4
```

Run only the default noisy setup:

```bash
python "Job Submission/95/calculate_95.py" --setup noisy --num-threads 4
```

Summarize outputs:

```bash
python "Job Submission/95/process_95.py"
```

List all noise presets:

```bash
python "Job Submission/95/calculate_95.py" --list-setups
```

Run a lower-thermal preset:

```bash
python "Job Submission/95/calculate_95.py" --setup noisy_nPth_0p01 --num-threads 4
```

## Local runtime guidance

A full `Nt=30` calculation with thermal branches and pure-loss Kraus branches on
S, P, and A can be expensive on an 8GB MacBook Air. The branch count scales
approximately as:

```text
N_thermal * N_Kraus(S) * N_Kraus(P) * N_Kraus(A)
```

Start with `--nt 8` or `--nt 10`, `--eta-index 0`, and `--max-items 1` before
launching the full calculation. Full production results may take many hours to
multiple days depending on the eta count, retained Kraus branches, thermal
branches, and thread count.
