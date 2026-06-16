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
#SBATCH --array=0-75

# Usage from this folder:
#   Full run for all four setups and all etas, reusing existing outputs:
#     sbatch submit_noisy_gkp_93.sbatch
#   Full rerun for all four setups and all etas, overwriting existing outputs:
#     sbatch --export=ALL,RECOMPUTE=1 submit_noisy_gkp_93.sbatch
#   Only noiseless validation, all etas:
#     python -u calculate_noisy_gkp_93.py --setup noiseless_reference --all-etas --validate-zero-noise --recompute
#   One eta/debug run:
#     python -u calculate_noisy_gkp_93.py --setup noiseless_reference --eta 0.30 --validate-zero-noise --recompute --num-threads 1
#
# The array writes to ./Data/<setup-output-subdir>/eta=*/. Existing outputs are
# reused unless calculate_noisy_gkp_93.py is run explicitly with --recompute.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "${SUBMIT_DIR}/calculate_noisy_gkp_93.py" ]; then
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SCRIPT_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

NUM_ETAS=19
SETUPS=(noisy noisy_nPth_0p01 noisy_nPth_0p001 noiseless_reference)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SETUP_IDX=$(( TASK_ID / NUM_ETAS ))
ETA_IDX=$(( TASK_ID % NUM_ETAS ))
SETUP="${SETUPS[$SETUP_IDX]}"

echo "Running noisy GKP run 93:"
echo "  SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "  SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "  setup               = ${SETUP}"
echo "  setup_index         = ${SETUP_IDX}"
echo "  eta_index           = ${ETA_IDX}"
echo "  output_root         = Data"
echo "  recompute           = ${RECOMPUTE:-0}"

EXTRA_ARGS=()
if [ "${RECOMPUTE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--recompute)
fi

python -u calculate_noisy_gkp_93.py \
  --setup-index "${SETUP_IDX}" \
  --eta-index "${ETA_IDX}" \
  --num-threads "${SLURM_CPUS_PER_TASK}" \
  "${EXTRA_ARGS[@]}"

# For fast convergence tests only, add:
#   --max-initial-thermal-branches 3 --max-kraus-terms 3 --initial-thermal-prob-tol 1e-5 --kraus-prob-tol 1e-8
