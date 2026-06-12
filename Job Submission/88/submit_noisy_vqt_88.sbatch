#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=noisy_vqt_88
#SBATCH --output=logs/noisy_vqt_88_%A_%a.out
#SBATCH --error=logs/noisy_vqt_88_%A_%a.err
#SBATCH --array=0-18

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

echo "Running noisy VQT run 88:"
echo "  SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "  SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo "  output_dir          = ../../Data_HPC/88"
echo "  initial_p_nbar      = 0.1"
echo "  kappa_o             = 0.99"
echo "  kappa_m             = 0.99"

python calculate_noisy_vqt_88.py --eta-index "${SLURM_ARRAY_TASK_ID}"
