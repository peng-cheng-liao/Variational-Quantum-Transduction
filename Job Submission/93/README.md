# Job 93: Noisy GKP coherent information

This self-contained job evaluates noisy coherent information for GKP parameters
copied from `Data_HPC/64_v2_2`, using
`QTorch.Transduction.transduction_protocol_CoherentInfo_GKP2_thermal_noise`.

The noise presets match job 92:

- `noisy`: `initial_p_nbar=0.1`, `kappa_o=kappa_m=0.99`
- `noisy_nPth_0p01`: `initial_p_nbar=0.01`, `kappa_o=kappa_m=0.99`
- `noisy_nPth_0p001`: `initial_p_nbar=0.001`, `kappa_o=kappa_m=0.99`
- `noiseless_reference`: `initial_p_nbar=0`, `kappa_o=kappa_m=1`

The default eta grid is `0.05, 0.10, ..., 0.95`.

Each parameter vector has length 8:

`delta1, r_hex1, phi1_hex, phi2_hex, delta2, r_hex2, phi3_hex, phi4_hex`.

The source metadata in `parameters/eta=*/source_info.json` records the selected
`d1`, `d2`, and `j2` from `Data_HPC/64_v2_2`. The production evaluator uses
the job-64 GKP constants `d1=2`, `d2=1`, `j2=0`, `Nt=30`, and `NR=2`, and
stores the source metadata values in `noise_config.json` for audit.

Example local dry run:

```bash
python calculate_noisy_gkp_93.py --setup noisy --eta 0.25 --dry-run
```

Submit the full array from this folder:

```bash
sbatch submit_noisy_gkp_93.sbatch
```

Aggregate completed results:

```bash
python process_noisy_gkp_93.py
```
