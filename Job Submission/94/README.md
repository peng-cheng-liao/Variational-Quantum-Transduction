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
