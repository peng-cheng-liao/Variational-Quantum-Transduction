# Job 94

Fixed rerun of `Job Submission/64_v2` for GKP coherent-information training.

Key changes:

- Saves best feasible parameters and states before `optimizer.step()` changes `x`.
- Selects best snapshots using feasible masked CI, with `ns <= 2.05` and `np <= 2.05` during training.
- Uses unambiguous raw filenames such as `best_parameters.npy`, `best_state_RS.npy`, and `best_state_P.npy`.
- Processes only candidates with `best_parameters.npy`; `parameters.npy` is produced only in `Data-Download` as the selected best parameter file.

Submit from this folder:

```bash
cd "Job Submission/94"
sbatch --array=0-4999 --export=ALL,OFFSET=0 submit_gkp_94.sbatch
sbatch --array=0-4999 --export=ALL,OFFSET=5000 submit_gkp_94.sbatch
sbatch --array=0-4249 --export=ALL,OFFSET=10000 submit_gkp_94.sbatch
```

Process results:

```bash
python process_gkp_94.py --overwrite
```

## Parameter recomputation validation

Use `verify_gkp94_parameters.py` to recompute coherent information from each
selected `parameters.npy` file and compare it with the processed best feasible
CI and selection summary score.

From repo root:

```bash
python "Job Submission/94/verify_gkp94_parameters.py" \
  --data-root Data_HPC/94/Data-Download-partial-10000 \
  --num-threads 1 \
  --strict
```

From HPC folder `94`:

```bash
python verify_gkp94_parameters.py \
  --data-root Data-Download-partial-10000 \
  --num-threads 1 \
  --strict
```

For a few etas:

```bash
python verify_gkp94_parameters.py \
  --data-root Data-Download-partial-10000 \
  --etas 0.05 0.30 0.65 \
  --num-threads 1
```

By default the TSV report is written to
`<data-root>/parameter_recompute_validation.tsv`, and the JSON summary is
written to `<data-root>/parameter_recompute_validation.json`.
