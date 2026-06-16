# Job 93: Noisy GKP coherent information

This self-contained HPC job evaluates noisy coherent information for selected
GKP parameters from `Data_HPC/64_v2_2`, using the job-local
`QTorch.Transduction.transduction_protocol_CoherentInfo_GKP2_thermal_noise`.

The default eta grid is `0.05, 0.10, ..., 0.95`. Each selected parameter vector
has length 8:

```text
delta1, r_hex1, phi1_hex, phi2_hex, delta2, r_hex2, phi3_hex, phi4_hex
```

## Source data

The source selection is summarized in `Data_HPC/64_v2_2/selection_summary.tsv`.
Each local `parameters/eta=*/source_info.json` records the selected source
`score`, `d1`, `d2`, and `j2`.

Job 93 now evaluates with the selected source protocol settings:

```text
d1 = source_d1
d2 = source_d2
j2 = source_j2
NR = d1
```

`noise_config.json` records both the source metadata and the actual evaluated
`d1`, `d2`, `j2`, and `NR`, with `uses_source_protocol_settings: true`.

## Output folders

Outputs are written under `Data/<setup-output-subdir>/eta=*/`:

- `noisy`: `Data/noisy_nPth=0p1_kS=0p99_kP=0p99/eta=*/`
- `noisy_nPth_0p01`: `Data/noisy_nPth=0p01_kS=0p99_kP=0p99/eta=*/`
- `noisy_nPth_0p001`: `Data/noisy_nPth=0p001_kS=0p99_kP=0p99/eta=*/`
- `noiseless_reference`: `Data/noiseless_nPth=0_kS=1_kP=1/eta=*/`

Existing outputs are reused by default. Use `--recompute` only when you want to
overwrite an eta output with corrected metadata and regenerated values.

## Zero-noise validation

Validate one eta against the selected noiseless score:

```bash
python calculate_noisy_gkp_93.py --setup noiseless_reference --eta 0.30 --validate-zero-noise --recompute
```

Validate all etas:

```bash
python calculate_noisy_gkp_93.py --setup noiseless_reference --all-etas --validate-zero-noise --recompute
```

Validation evaluates with `initial_p_thermal_nbar=0`, `kappa_o=1`,
`kappa_m=1`, `n_o=0`, and `n_m=0`, prints the eta, source score, recomputed
score, absolute error, and `(d1,d2,j2,NR)`, and exits nonzero if any error is
larger than `1e-5`.

## Slurm

Submit the full four-setup array from this folder:

```bash
sbatch submit_noisy_gkp_93.sbatch
```

For a local metadata dry run:

```bash
python calculate_noisy_gkp_93.py --setup noisy --eta 0.25 --dry-run
```

Aggregate completed results:

```bash
python process_noisy_gkp_93.py
```
