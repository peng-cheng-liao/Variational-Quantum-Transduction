#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --job-name=GKP94_CI
#SBATCH --output=logs/gkp94_%A_%a.out
#SBATCH --error=logs/gkp94_%A_%a.err
#SBATCH --array=0-4999

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs Data Data-Download

# ---- thread caps for NumPy (MKL/OpenMP/OpenBLAS) + PyTorch ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# -----------------------
# Global task indexing: chunked arrays via OFFSET
# -----------------------
TOTAL_TASKS=14250
OFFSET="${OFFSET:-0}"
TASK_ID=$(( OFFSET + SLURM_ARRAY_TASK_ID ))

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "TASK_ID=$TASK_ID >= TOTAL_TASKS=$TOTAL_TASKS, exiting."
  exit 0
fi

# -----------------------
# Sweep config
# -----------------------
ETAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
D1S=(2 3 4 5 6)
D2S=(1 2 3 4 5)

# Total combos:
#   eta: 19
#   randomization: 10 (0..9)
#   d1: 5 (2..6)
#   (d2,j2): sum_{d2=1..5} d2 = 15
#   total = 19*10*5*15 = 14250 -> TASK_ID 0..14249

PER_ETA=$(( 10 * 5 * 15 ))
ETA_IDX=$(( TASK_ID / PER_ETA ))
REM1=$(( TASK_ID % PER_ETA ))

PER_RAND=$(( 5 * 15 ))
RAND_IDX=$(( REM1 / PER_RAND ))
REM2=$(( REM1 % PER_RAND ))

PER_D1=15
D1_IDX=$(( REM2 / PER_D1 ))
REM3=$(( REM2 % PER_D1 ))

ETA="${ETAS[$ETA_IDX]}"
RANDOMIZATION="${RAND_IDX}"
D1="${D1S[$D1_IDX]}"

if   (( REM3 < 1 )); then D2=1; J2=$(( REM3 - 0 ))
elif (( REM3 < 3 )); then D2=2; J2=$(( REM3 - 1 ))
elif (( REM3 < 6 )); then D2=3; J2=$(( REM3 - 3 ))
elif (( REM3 < 10 )); then D2=4; J2=$(( REM3 - 6 ))
else                      D2=5; J2=$(( REM3 - 10 ))
fi

export ETA="${ETA}"
export ETA_TAG="${ETA}"
export RANDOMIZATION="${RANDOMIZATION}"
export D1="${D1}"
export D2="${D2}"
export J2="${J2}"

echo "Running GKP job 94:"
echo "  SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "  OFFSET              = $OFFSET"
echo "  TASK_ID             = $TASK_ID"
echo "  ETA                 = $ETA"
echo "  RANDOMIZATION       = $RANDOMIZATION"
echo "  D1                  = $D1"
echo "  D2                  = $D2"
echo "  J2                  = $J2"

python train_ci_94.py
