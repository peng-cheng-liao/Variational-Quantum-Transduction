#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --job-name=CI_th_s1
#SBATCH --output=logs/th_s1_%A_%a.out
#SBATCH --error=logs/th_s1_%A_%a.err
#SBATCH --array=0-1899

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
N_M_LIST=(0.00 0.01 0.03 0.05 0.10)
KAPPA_M_LIST=(1.00 0.99 0.97 0.95 0.90)
SEEDS_PER_POINT=20
SCAN_MODE="${SCAN_MODE:-nm}"  # nm or km

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ETA_IDX=$(( TASK_ID / (SEEDS_PER_POINT * 5) ))
NOISE_IDX=$(( (TASK_ID / SEEDS_PER_POINT) % 5 ))
SEED=$(( TASK_ID % SEEDS_PER_POINT ))

ETA="${ETAS[$ETA_IDX]}"
KAPPA_O="1.00"
N_O="0.00"

if [[ "$SCAN_MODE" == "km" ]]; then
  KAPPA_M="${KAPPA_M_LIST[$NOISE_IDX]}"
  N_M="0.00"
else
  KAPPA_M="1.00"
  N_M="${N_M_LIST[$NOISE_IDX]}"
fi

export ETA="${ETA}"
export ETA_TAG="${ETA}"          # keep folder naming stable (0.10 not 0.1)
export SEED="${SEED}"
export KAPPA_O N_O KAPPA_M N_M
export NS_CONSTRAINT=2
export NP_CONSTRAINT=2
export DEPTH=20
export NT=30

# Stage-1: short runs
export MAX_STEPS=5000
export RESUME=0
export RESUME_FROM=latest

# Explicit dirs to guarantee consistent layout
RUN_ROOT="runs/thermal_eta=${ETA_TAG}_depth=20_Nt=30_ko=${KAPPA_O}_no=${N_O}_km=${KAPPA_M}_nm=${N_M}"
CKPT_ROOT="training_parameters/thermal_eta=${ETA_TAG}_depth=20_Nt=30_ko=${KAPPA_O}_no=${N_O}_km=${KAPPA_M}_nm=${N_M}"
export RUN_DIR="${RUN_ROOT}/seed_${SEED}"
export CKPT_DIR="${CKPT_ROOT}/seed_${SEED}"

echo "Running stage1:"
echo "  ETA      = $ETA"
echo "  SCAN_MODE= $SCAN_MODE"
echo "  KAPPA_O  = $KAPPA_O"
echo "  N_O      = $N_O"
echo "  KAPPA_M  = $KAPPA_M"
echo "  N_M      = $N_M"
echo "  SEED     = $SEED"
echo "  MAX_STEPS= $MAX_STEPS"
echo "  RUN_DIR  = $RUN_DIR"
echo "  CKPT_DIR = $CKPT_DIR"
echo "  TASK_ID  = $TASK_ID"

python train_ci.py
