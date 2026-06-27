# Job 93: Noisy GKP eta scans for Job 99 comparison

This self-contained HPC job evaluates noisy GKP coherent information using the
corrected local parameters copied from Job 94 in `parameters/eta=*`. It is
matched to the Job 99 VQT eta-scan cases, but it does not compute or scan any
auxiliary A mode. VQT A-mode values are recorded only as metadata.

The eta grid is:

```text
0.05, 0.10, ..., 0.95
```

Each selected parameter vector has length 8:

```text
delta1, r_hex1, phi1_hex, phi2_hex, delta2, r_hex2, phi3_hex, phi4_hex
```

## Source Data

Each local `parameters/eta=*/source_info.json` records the selected Job 94
source `score`, `d1`, `d2`, and `j2`. Job 93 evaluates with those source
protocol settings:

```text
d1 = source_d1
d2 = source_d2
j2 = source_j2
NR = d1
```

`noise_config.json` records both the source metadata and the actual evaluated
`d1`, `d2`, `j2`, `Nt`, and `NR`, with `uses_source_protocol_settings: true`.

## GKP Settings

The computation has seven unique shared GKP thermal settings:

```text
initial_p_thermal_nbar = 0.0, 0.001, 0.01, 0.03, 0.05, 0.07, 0.1
kappa_o = 0.99
kappa_m = 0.99
n_o = 0.0
n_m = 0.0
```

There is no auxiliary mode A in the GKP protocol, so the protocol call includes
no A-loss or A-thermal argument. Metadata records:

```text
gkp_has_no_auxiliary_mode_A = true
gkp_independent_of_tau_A = true
matched_vqt_run_id = 99
```

## Output Layout

New outputs are kept separate from prior Job 93 data:

```text
Data/gkp_eta_scans_nPth_grid_tauSP_0p99/
```

Shared compute outputs are written once per unique GKP setting:

```text
Data/gkp_eta_scans_nPth_grid_tauSP_0p99/shared/nPth_<tag>_kS_0p99_kP_0p99/eta=*/
```

VQT-compatible case folders are materialized separately:

```text
Data/gkp_eta_scans_nPth_grid_tauSP_0p99/cases/<vqt_case_id>/eta=*/
```

Materialized case folders contain regular small files, not symlinks, and reuse
the shared CI value with case-specific metadata. They are not overwritten unless
`--overwrite` is passed to the processing script or `--overwrite-materialized`
is passed to the calculation script.

## Job 99 Eta Cases

The materialized VQT-compatible eta cases are:

```text
case1_eta_scan_nthP_0_nthA_0_tauA_0p90
nthP_0_nthA_0_tauAll_0p99
case2_eta_scan_nthP_0p1_nthA_0p1_tauA_0p90
nthP_0p001_nthA_0p001_tauAll_0p99
nthP_0p01_nthA_0p01_tauAll_0p99
nthP_0p03_nthA_0p03_tauAll_0p99
nthP_0p05_nthA_0p05_tauAll_0p99
nthP_0p07_nthA_0p07_tauAll_0p99
nthP_0p1_nthA_0p1_tauAll_0p99
```

The `case2_*` and `nthP_0p1_*` cases share the same GKP setting. The two
zero-thermal cases also share the same GKP setting.

## Slurm

Submit the 133-task array on Discovery:

```bash
cd /home1/liaopeng/QuantumTransduction/93
sbatch submit_noisy_gkp_93.sbatch
```

Array mapping:

```text
SETTING_INDEX = SLURM_ARRAY_TASK_ID / 19
ETA_INDEX = SLURM_ARRAY_TASK_ID % 19
```

The default HPC output root is:

```text
/home1/liaopeng/QuantumTransduction/93/Data/gkp_eta_scans_nPth_grid_tauSP_0p99
```

Existing `best_feasible_ci.txt` files are treated as cache hits. To recompute:

```bash
sbatch --export=ALL,RECOMPUTE=1 submit_noisy_gkp_93.sbatch
```

## Local Checks

List settings and cases:

```bash
python calculate_noisy_gkp_93.py --list-settings
```

Dry-run selected array tasks:

```bash
python calculate_noisy_gkp_93.py --setting-index 0 --eta-index 0 --dry-run
python calculate_noisy_gkp_93.py --setting-index 6 --eta-index 18 --dry-run
```

Materialize VQT-compatible case folders after shared outputs are complete:

```bash
python process_noisy_gkp_93.py --materialize-cases
```

Write summaries without materializing:

```bash
python process_noisy_gkp_93.py
```

The processing script writes:

```text
noise_ci_summary_93.tsv
noise_ci_summary_93.json
```
