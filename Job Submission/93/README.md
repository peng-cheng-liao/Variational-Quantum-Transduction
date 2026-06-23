# Job 93: Noisy GKP coherent information rerun

This self-contained HPC job evaluates noisy coherent information for corrected
GKP parameters copied from `Data_HPC/94/Data-Download`, using the job-local
`QTorch.Transduction.transduction_protocol_CoherentInfo_GKP2_thermal_noise`.

The default eta grid is `0.05, 0.10, ..., 0.95`. Each selected parameter vector
has length 8:

```text
delta1, r_hex1, phi1_hex, phi2_hex, delta2, r_hex2, phi3_hex, phi4_hex
```

## Source data

The source selection is summarized in `Data_HPC/94/Data-Download/selection_summary.tsv`.
Each local `parameters/eta=*/source_info.json` records the selected source
`score`, `d1`, `d2`, and `j2`.

Job 93 evaluates with the selected source protocol settings:

```text
d1 = source_d1
d2 = source_d2
j2 = source_j2
NR = d1
```

`noise_config.json` records both the source metadata and the actual evaluated
`d1`, `d2`, `j2`, `Nt`, and `NR`, with `uses_source_protocol_settings: true`.

## Noise setup

This rerun evaluates only the noisy GKP-QT setup:

```text
initial_p_thermal_nbar = 0.1
kappa_o = 0.99
kappa_m = 0.99
n_o = 0.0
n_m = 0.0
```

There is no auxiliary mode A in the GKP scheme, so no A-loss parameter is
included.

## Output folders

Outputs are written under:

```text
Data/noisy_nPth=0p1_kS=0p99_kP=0p99/eta=*/
```

Existing outputs are reused by default. Use `--recompute` only when you want to
overwrite an eta output with corrected metadata and regenerated values.

## Slurm

Submit the 19-task array from this folder:

```bash
cd "Job Submission/93"
sbatch submit_noisy_gkp_93.sbatch
```

The sbatch script also supports submission from the repository root:

```bash
sbatch "Job Submission/93/submit_noisy_gkp_93.sbatch"
```

For a local metadata dry run:

```bash
python calculate_noisy_gkp_93.py --setup noisy --eta 0.25 --dry-run
```

Aggregate completed results:

```bash
python process_noisy_gkp_93.py
```
