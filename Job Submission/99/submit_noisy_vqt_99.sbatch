#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=noisy_vqt_99
#SBATCH --chdir=/home1/liaopeng/QuantumTransduction/99
#SBATCH --output=/home1/liaopeng/QuantumTransduction/99/logs/noisy_vqt_99_%A_%a.out
#SBATCH --error=/home1/liaopeng/QuantumTransduction/99/logs/noisy_vqt_99_%A_%a.err
#SBATCH --array=0-62

set -euo pipefail

JOB_DIR="/home1/liaopeng/QuantumTransduction/99"
cd "${JOB_DIR}"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs Data

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${JOB_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

# Explicit task map:
#   case 0: tau_A scan, eta=0.30, n_P^th=n_A^th=0.01, tau_S=tau_P=0.99, 21 tau_A points
#   case 1: tau_A scan, eta=0.50, n_P^th=n_A^th=0.01, tau_S=tau_P=0.99, 21 tau_A points
#   case 2: tau_A scan, eta=0.70, n_P^th=n_A^th=0.01, tau_S=tau_P=0.99, 21 tau_A points
CASE_LENGTHS=(21 21 21)
CASE_NAMES=(
  eta_0p30_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan
  eta_0p50_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan
  eta_0p70_nthP_0p01_nthA_0p01_tauSP_0p99_tauA_scan
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
REMAINING="${TASK_ID}"
CASE_IDX=-1
SCAN_IDX=-1
for IDX in 0 1 2; do
  LENGTH="${CASE_LENGTHS[$IDX]}"
  if (( REMAINING < LENGTH )); then
    CASE_IDX="${IDX}"
    SCAN_IDX="${REMAINING}"
    break
  fi
  REMAINING=$(( REMAINING - LENGTH ))
done

if (( CASE_IDX < 0 )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

echo "Running noisy VQT run 99:"
echo "  SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "  SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "  job_dir             = ${JOB_DIR}"
echo "  case                = ${CASE_NAMES[$CASE_IDX]}"
echo "  case_index          = ${CASE_IDX}"
echo "  scan_index          = ${SCAN_IDX}"
echo "  output_root         = ${JOB_DIR}/Data"

python calculate_noisy_vqt_99.py --case-index "${CASE_IDX}" --scan-index "${SCAN_IDX}" --output-root "${JOB_DIR}/Data"
