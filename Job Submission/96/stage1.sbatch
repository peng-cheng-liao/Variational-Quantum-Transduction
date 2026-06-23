#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --job-name=CI96_s1_depth
#SBATCH --output=/home1/liaopeng/QuantumTransduction/96/logs/s1_%A_%a.out
#SBATCH --error=/home1/liaopeng/QuantumTransduction/96/logs/s1_%A_%a.err
#SBATCH --array=0-1799

set -euo pipefail

HOME_JOB_DIR="/home1/liaopeng/QuantumTransduction/96"
SCRATCH_DATA_ROOT="/scratch1/liaopeng/QuantumTransduction/96/Data"
SCRATCH_RUNS_ROOT="${SCRATCH_DATA_ROOT}/runs"
SCRATCH_PARAMS_ROOT="${SCRATCH_DATA_ROOT}/training_parameters"
DOWNLOAD_DIR="${HOME_JOB_DIR}/Data_Download"

mkdir -p "${HOME_JOB_DIR}/logs"
mkdir -p "${SCRATCH_RUNS_ROOT}" "${SCRATCH_PARAMS_ROOT}"
mkdir -p "${DOWNLOAD_DIR}"
cd "${HOME_JOB_DIR}"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

# ---- thread caps for NumPy (MKL/OpenMP/OpenBLAS) + PyTorch ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# -----------------------
# Sweep config: eta fixed at 0.30, depth in 2,4,...,18, seeds 0..199.
# Total tasks = 9 depths * 200 seeds = 1800.
# -----------------------
ETA="0.30"
DEPTHS=(2 4 6 8 10 12 14 16 18)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
DEPTH_IDX=$(( TASK_ID / 200 ))
SEED=$(( TASK_ID % 200 ))

DEPTH="${DEPTHS[$DEPTH_IDX]}"
DEPTH_TAG="$(printf "%02d" "${DEPTH}")"

export ETA="${ETA}"
export ETA_TAG="${ETA}"
export DEPTH="${DEPTH}"
export DEPTH_TAG="${DEPTH_TAG}"
export SEED="${SEED}"

# Stage-1: short runs
export MAX_STEPS=5000
export RESUME=0
export RESUME_FROM=latest

# Explicit dirs to guarantee consistent layout
export RUN_DIR="${SCRATCH_RUNS_ROOT}/eta=${ETA_TAG}_depth=${DEPTH_TAG}_Nt=30/seed_${SEED}"
export CKPT_DIR="${SCRATCH_PARAMS_ROOT}/eta=${ETA_TAG}_depth=${DEPTH_TAG}_Nt=30/seed_${SEED}"

echo "Running stage1:"
echo "  ETA      = $ETA"
echo "  DEPTH    = $DEPTH"
echo "  SEED     = $SEED"
echo "  MAX_STEPS= $MAX_STEPS"
echo "  RUN_DIR  = $RUN_DIR"
echo "  CKPT_DIR = $CKPT_DIR"
echo "  DOWNLOAD_DIR = $DOWNLOAD_DIR"
echo "  TASK_ID  = $TASK_ID"

python train_ci.py
