#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --job-name=GKP_match
#SBATCH --output=logs/gkp_%A_%a.out
#SBATCH --error=logs/gkp_%A_%a.err
#SBATCH --array=0-54

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs

# ---- thread caps for NumPy (MKL/OpenMP/OpenBLAS) + PyTorch ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# -----------------------
# Sweep config: d1 in {1..10}, i in {0..d1-1}
# Total tasks = sum_{d1=1}^{10} d1 = 55  => array 0-54
# Map SLURM_ARRAY_TASK_ID -> (d1, i)
# -----------------------
TASK_ID="${SLURM_ARRAY_TASK_ID}"

D1=1
OFFSET=0
while true; do
  NEXT=$(( OFFSET + D1 ))
  if [ "${TASK_ID}" -lt "${NEXT}" ]; then
    I=$(( TASK_ID - OFFSET ))
    break
  fi
  OFFSET="${NEXT}"
  D1=$(( D1 + 1 ))
done

export D1
export I

echo "Running task:"
echo "  TASK_ID = ${TASK_ID}"
echo "  D1      = ${D1}"
echo "  I       = ${I}"

# Run your script (recommended: modify the python code to read D1/I from env vars)
# e.g., in python:
#   d1 = int(os.environ["D1"])
#   i  = int(os.environ["I"])
python gkp_state_match.py
