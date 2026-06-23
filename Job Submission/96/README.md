# Job 96: depth-convergence scan for non-adaptive VQT-EA

Job 96 mirrors the Job 84 two-stage coherent-information training workflow, but
uses depth as the scan axis at fixed eta.

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

## Workflow

Run from this folder on HPC:

```bash
cd "Job Submission/96"
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

Stage 2 resumes those selected seeds from `best_feasible` to 20000 steps:

```bash
sbatch stage2.sbatch
```

Stage 2 array mapping:

```text
SLURM_ARRAY_TASK_ID = depth_index * 20 + top_seed_rank
depth_index in 0..8 -> depth in [2,4,6,8,10,12,14,16,18]
top_seed_rank in 0..19 from top20_seeds.txt
array = 0-179
```

Process the final best feasible outputs:

```bash
sbatch process_depth_scan_96.sbatch
```

The processor writes:

```text
Data_HPC/96/selection_summary.tsv
Data_HPC/96/depth_scan_with_job84_reference.tsv
```

Each selected depth folder contains:

```text
best_feasible_ci.txt
parameters_best_feasible.npy
source_info.json
```

## Local Checks

Dry-run one training task without running optimization:

```bash
DEPTH=2 SEED=0 python train_ci.py --dry-run
```

Dry-run processing without writing files:

```bash
python process_depth_scan_96.py --dry-run
```
