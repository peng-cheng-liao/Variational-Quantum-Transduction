#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=noisy_vqt_99
#SBATCH --output=logs/noisy_vqt_99_%A_%a.out
#SBATCH --error=logs/noisy_vqt_99_%A_%a.err
#SBATCH --array=0-79

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# Explicit task map:
#   case 0: eta scan, n_P^th=n_A^th=0, tau_A=0.90, 19 eta points
#   case 1: eta scan, n_P^th=n_A^th=0.1, tau_A=0.90, 19 eta points
#   case 2: tau_A scan, eta=0.60, n_P^th=n_A^th=0, 21 tau_A points
#   case 3: tau_A scan, eta=0.60, n_P^th=n_A^th=0.1, 21 tau_A points
CASE_LENGTHS=(19 19 21 21)
CASE_NAMES=(
  case1_eta_scan_nthP_0_nthA_0_tauA_0p90
  case2_eta_scan_nthP_0p1_nthA_0p1_tauA_0p90
  case3_tauA_scan_eta_0p60_nthP_0_nthA_0
  case4_tauA_scan_eta_0p60_nthP_0p1_nthA_0p1
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
REMAINING="${TASK_ID}"
CASE_IDX=-1
SCAN_IDX=-1
for IDX in 0 1 2 3; do
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
echo "  script_dir          = ${SCRIPT_DIR}"
echo "  case                = ${CASE_NAMES[$CASE_IDX]}"
echo "  case_index          = ${CASE_IDX}"
echo "  scan_index          = ${SCAN_IDX}"
echo "  output_root         = ../../Data_HPC/99"

python calculate_noisy_vqt_99.py --case-index "${CASE_IDX}" --scan-index "${SCAN_IDX}"
