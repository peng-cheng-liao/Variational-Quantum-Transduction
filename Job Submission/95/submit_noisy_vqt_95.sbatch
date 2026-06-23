#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=noisy_vqt_95
#SBATCH --output=logs/noisy_vqt_95_%A_%a.out
#SBATCH --error=logs/noisy_vqt_95_%A_%a.err
#SBATCH --array=0-56

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

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
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

NUM_ETAS=19
SETUPS=(kappaA_1p00 kappaA_0p95 kappaA_0p90)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SETUP_IDX=$(( TASK_ID / NUM_ETAS ))
ETA_IDX=$(( TASK_ID % NUM_ETAS ))
SETUP="${SETUPS[$SETUP_IDX]}"

echo "Running noisy VQT run 95:"
echo "  SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "  SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "  setup               = ${SETUP}"
echo "  setup_index         = ${SETUP_IDX}"
echo "  eta_index           = ${ETA_IDX}"
echo "  output_root         = ../../Data_HPC/95"

python calculate_noisy_vqt_95.py --setup-index "${SETUP_IDX}" --eta-index "${ETA_IDX}"
