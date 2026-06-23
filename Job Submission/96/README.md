# Job 96: depth-convergence scan for non-adaptive VQT-EA

Job 96 mirrors the Job 84 two-stage coherent-information training workflow, but
uses depth as the scan axis at fixed eta. The scripts and local `QTorch` copy
live in the home job folder, while large raw outputs are written to scratch.

## Fixed Configuration

```text
eta = 0.30
Nt = 30
ns = np = 2
objective = transduction_protocol_CoherentInfo_ECD_MM_EA
depths = 2, 4, 6, 8, 10, 12, 14, 16, 18
seeds = 0, 1, ..., 199
```

The parameter vector is initialized independently per depth and seed with
length `24 * depth`. Depth 20 is not trained in Job 96; it is read only as the
Job 84 reference during processing.

## HPC Paths

Run scripts from:

```text
/home1/liaopeng/QuantumTransduction/96
```

Raw stage-1 and stage-2 outputs are stored in scratch:

```text
/scratch1/liaopeng/QuantumTransduction/96/Data/runs
/scratch1/liaopeng/QuantumTransduction/96/Data/training_parameters
```

Compact selected outputs are copied to:

```text
/home1/liaopeng/QuantumTransduction/96/Data_Download
```

The Job 84 depth-20 reference on HPC is read from:

```text
/home1/liaopeng/QuantumTransduction/84
```

Path verification on 2026-06-23 confirmed that `/home1/liaopeng/QuantumTransduction`,
`/home1/liaopeng/QuantumTransduction/96`, `Data_Download`, and the scratch
`Data/runs` and `Data/training_parameters` folders exist or can be created and
written. `df -h /scratch1` reported about 902T available on `/scratch1`.
`quota -s` produced no usable quota output. Job 84 eta=0.30 raw reference files
were found under `/home1/liaopeng/QuantumTransduction/84/runs` and
`/home1/liaopeng/QuantumTransduction/84/training_parameters`.

## Workflow

Run from the home job folder on HPC:

```bash
cd /home1/liaopeng/QuantumTransduction/96
```

Stage 1 trains every depth and seed for 5000 steps:

```bash
sbatch stage1.sbatch
```

Stage 1 array mapping:

```text
SLURM_ARRAY_TASK_ID = depth_index * 200 + seed
depth_index in 0..8 -> depth in [2,4,6,8,10,12,14,16,18]
seed in 0..199
array = 0-1799
```

After stage 1 finishes, select the top 20 seeds per depth by
`best_feasible_ci.txt`:

```bash
sbatch select_top.sbatch
```

Stage 2 must wait for `select_top.sbatch`. It resumes selected seeds from
`best_feasible` to 20000 steps:

```bash
sbatch stage2.sbatch
```

Stage 2 array mapping:

```text
SLURM_ARRAY_TASK_ID = depth_index * 20 + top_seed_rank
depth_index in 0..8 -> depth in [2,4,6,8,10,12,14,16,18]
top_seed_rank in 0..19 from scratch top20_seeds.txt
array = 0-179
```

Process the final best feasible outputs after stage 2 finishes:

```bash
sbatch process_depth_scan_96.sbatch
```

Scratch raw data may be purged by the cluster, so process and copy selected
outputs to `Data_Download` after stage 2 completes.

The processor writes:

```text
Data_Download/selection_summary.tsv
Data_Download/depth_scan_with_job84_reference.tsv
```

Each selected depth folder contains:

```text
best_feasible_ci.txt
parameters_best_feasible.npy
source_info.json
state_RS_best_feasible.npy  # if present
state_PA_best_feasible.npy  # if present
```

The processor does not copy full checkpoints or CI/ns/np histories to
`Data_Download`.

## Local Checks

Dry-run one training task without running optimization:

```bash
DEPTH=2 SEED=0 ETA=0.30 \
RUN_DIR=/scratch1/liaopeng/QuantumTransduction/96/Data/runs/test_dry \
CKPT_DIR=/scratch1/liaopeng/QuantumTransduction/96/Data/training_parameters/test_dry \
python train_ci.py --dry-run
```

Dry-run processing with explicit HPC-style paths:

```bash
python process_depth_scan_96.py \
  --runs-root /scratch1/liaopeng/QuantumTransduction/96/Data/runs \
  --parameters-root /scratch1/liaopeng/QuantumTransduction/96/Data/training_parameters \
  --output-root /home1/liaopeng/QuantumTransduction/96/Data_Download \
  --dry-run
```
