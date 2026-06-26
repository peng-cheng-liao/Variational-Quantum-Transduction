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
#SBATCH --array=0-56

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
#   case 0: eta scan, n_P^th=n_A^th=0.1, tau_S=tau_P=tau_A=0.99, 19 eta points
#   case 1: eta scan, n_P^th=n_A^th=0.01, tau_S=tau_P=tau_A=0.99, 19 eta points
#   case 2: eta scan, n_P^th=n_A^th=0.001, tau_S=tau_P=tau_A=0.99, 19 eta points
CASE_LENGTHS=(19 19 19)
CASE_NAMES=(
  nthP_0p1_nthA_0p1_tauAll_0p99
  nthP_0p01_nthA_0p01_tauAll_0p99
  nthP_0p001_nthA_0p001_tauAll_0p99
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
