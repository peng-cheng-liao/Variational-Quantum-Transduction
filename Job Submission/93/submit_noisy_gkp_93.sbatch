#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=noisy_gkp_93
#SBATCH --output=logs/noisy_gkp_93_%A_%a.out
#SBATCH --error=logs/noisy_gkp_93_%A_%a.err
#SBATCH --array=0-132

# Usage on Discovery:
#   cd /home1/liaopeng/QuantumTransduction/93
#   sbatch submit_noisy_gkp_93.sbatch
#
# Task mapping:
#   SETTING_INDEX = SLURM_ARRAY_TASK_ID / 19
#   ETA_INDEX     = SLURM_ARRAY_TASK_ID % 19
#
# This computes seven shared GKP settings across 19 eta values. Existing
# best_feasible_ci.txt files are reused unless RECOMPUTE=1 is exported.

set -euo pipefail

HPC_JOB_DIR="/home1/liaopeng/QuantumTransduction/93"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "${HPC_JOB_DIR}/calculate_noisy_gkp_93.py" ]; then
  SCRIPT_DIR="${HPC_JOB_DIR}"
elif [ -f "${SUBMIT_DIR}/calculate_noisy_gkp_93.py" ]; then
  SCRIPT_DIR="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/Job Submission/93/calculate_noisy_gkp_93.py" ]; then
  SCRIPT_DIR="${SUBMIT_DIR}/Job Submission/93"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs

VENV_PATH="/home1/liaopeng/QuantumTransduction/venvs/job93_py39"
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: Missing Job 93 Python virtual environment: ${VENV_PATH}" >&2
  echo "Create it with the setup commands documented in README.md before submitting." >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SCRIPT_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

python - <<'PY'
import sys
print("Python executable:", sys.executable)
import datetime
print("datetime ok")
import numpy
print("numpy:", numpy.__version__)
import torch
print("torch:", torch.__version__)
PY

NUM_ETAS=19
TASK_ID="${SLURM_ARRAY_TASK_ID}"
SETTING_INDEX=$(( TASK_ID / NUM_ETAS ))
ETA_INDEX=$(( TASK_ID % NUM_ETAS ))
OUTPUT_ROOT="${OUTPUT_ROOT:-/home1/liaopeng/QuantumTransduction/93/Data/gkp_eta_scans_nPth_grid_tauSP_0p99}"

echo "Running noisy GKP run 93:"
echo "  SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "  SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "  setting_index       = ${SETTING_INDEX}"
echo "  eta_index           = ${ETA_INDEX}"
echo "  output_root         = ${OUTPUT_ROOT}"
echo "  recompute           = ${RECOMPUTE:-0}"
echo "  materialize_cases   = ${MATERIALIZE_CASES:-0}"

EXTRA_ARGS=()
if [ "${RECOMPUTE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--recompute)
fi
if [ "${MATERIALIZE_CASES:-0}" = "1" ]; then
  EXTRA_ARGS+=(--materialize-cases)
fi

python -u calculate_noisy_gkp_93.py \
  --setting-index "${SETTING_INDEX}" \
  --eta-index "${ETA_INDEX}" \
  --output-root "${OUTPUT_ROOT}" \
  --num-threads "${SLURM_CPUS_PER_TASK}" \
  "${EXTRA_ARGS[@]}"

# For fast convergence tests only, add:
#   --max-initial-thermal-branches 3 --max-kraus-terms 3 --initial-thermal-prob-tol 1e-5 --kraus-prob-tol 1e-8
