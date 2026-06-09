#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=10:00:00
#SBATCH --job-name=CI_th_s2
#SBATCH --output=logs/th_s2_%A_%a.out
#SBATCH --error=logs/th_s2_%A_%a.err
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

ETAS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
N_M_LIST=(0.00 0.01 0.03 0.05 0.10)
KAPPA_M_LIST=(1.00 0.99 0.97 0.95 0.90)
TOPK=20
SCAN_MODE="${SCAN_MODE:-nm}"  # nm or km

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ETA_IDX=$(( TASK_ID / (TOPK * 5) ))
NOISE_IDX=$(( (TASK_ID / TOPK) % 5 ))
RANK=$(( TASK_ID % TOPK ))

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

RUN_ROOT="runs/thermal_eta=${ETA}_depth=20_Nt=30_ko=${KAPPA_O}_no=${N_O}_km=${KAPPA_M}_nm=${N_M}"
CKPT_ROOT="training_parameters/thermal_eta=${ETA}_depth=20_Nt=30_ko=${KAPPA_O}_no=${N_O}_km=${KAPPA_M}_nm=${N_M}"
TOP_FILE="${RUN_ROOT}/top${TOPK}_seeds.txt"

if [[ ! -f "$TOP_FILE" ]]; then
  echo "ERROR: missing $TOP_FILE (did you run select_top.sbatch?)"
  exit 1
fi

SEED="$(sed -n "$((RANK+1))p" "$TOP_FILE")"

if [[ -z "${SEED}" ]]; then
  echo "ERROR: could not read seed for rank=$RANK from $TOP_FILE"
  exit 1
fi

export ETA="${ETA}"
export ETA_TAG="${ETA}"
export SEED="${SEED}"
export KAPPA_O N_O KAPPA_M N_M
export NS_CONSTRAINT=2
export NP_CONSTRAINT=2
export DEPTH=20
export NT=30

# Stage-2: resume to full length
export MAX_STEPS=20000
export RESUME=1
export RESUME_FROM=best_feasible

export RUN_DIR="${RUN_ROOT}/seed_${SEED}"
export CKPT_DIR="${CKPT_ROOT}/seed_${SEED}"

echo "Running stage2:"
echo "  ETA        = $ETA"
echo "  SCAN_MODE  = $SCAN_MODE"
echo "  KAPPA_O    = $KAPPA_O"
echo "  N_O        = $N_O"
echo "  KAPPA_M    = $KAPPA_M"
echo "  N_M        = $N_M"
echo "  SEED       = $SEED (rank $RANK for eta)"
echo "  MAX_STEPS  = $MAX_STEPS"
echo "  RESUME_FROM= $RESUME_FROM"
echo "  RUN_DIR    = $RUN_DIR"
echo "  CKPT_DIR   = $CKPT_DIR"
echo "  TOP_FILE   = $TOP_FILE"
echo "  TASK_ID    = $TASK_ID"

python train_ci.py
