#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --job-name=CI_s1_eta
#SBATCH --output=logs/s1_%A_%a.out
#SBATCH --error=logs/s1_%A_%a.err
#SBATCH --array=0-3799

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs runs training_parameters

# ---- thread caps for NumPy (MKL/OpenMP/OpenBLAS) + PyTorch ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# -----------------------
# Sweep config
# -----------------------
ETAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ETA_IDX=$(( TASK_ID / 200 ))
SEED=$(( TASK_ID % 200 ))

ETA="${ETAS[$ETA_IDX]}"

export ETA="${ETA}"
export ETA_TAG="${ETA}"          # keep folder naming stable (0.10 not 0.1)
export SEED="${SEED}"

# Stage-1: short runs
export MAX_STEPS=5000
export RESUME=0
export RESUME_FROM=latest

# Explicit dirs to guarantee consistent layout
export RUN_DIR="runs/eta=${ETA_TAG}_depth=20_Nt=30/seed_${SEED}"
export CKPT_DIR="training_parameters/eta=${ETA_TAG}_depth=20_Nt=30/seed_${SEED}"

echo "Running stage1:"
echo "  ETA      = $ETA"
echo "  SEED     = $SEED"
echo "  MAX_STEPS= $MAX_STEPS"
echo "  RUN_DIR  = $RUN_DIR"
echo "  CKPT_DIR = $CKPT_DIR"
echo "  TASK_ID  = $TASK_ID"

python train_ci.py
