#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:20:00
#SBATCH --job-name=CI96_process
#SBATCH --output=/home1/liaopeng/QuantumTransduction/96/logs/process_%j.out
#SBATCH --error=/home1/liaopeng/QuantumTransduction/96/logs/process_%j.err

set -euo pipefail

HOME_JOB_DIR="/home1/liaopeng/QuantumTransduction/96"
SCRATCH_DATA_ROOT="/scratch1/liaopeng/QuantumTransduction/96/Data"
SCRATCH_RUNS_ROOT="${SCRATCH_DATA_ROOT}/runs"
SCRATCH_PARAMS_ROOT="${SCRATCH_DATA_ROOT}/training_parameters"
DOWNLOAD_DIR="${HOME_JOB_DIR}/Data_Download"
REFERENCE_ROOT="/home1/liaopeng/QuantumTransduction/84"

mkdir -p "${HOME_JOB_DIR}/logs"
mkdir -p "${SCRATCH_RUNS_ROOT}" "${SCRATCH_PARAMS_ROOT}"
mkdir -p "${DOWNLOAD_DIR}"
cd "${HOME_JOB_DIR}"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

python process_depth_scan_96.py \
  --runs-root "${SCRATCH_RUNS_ROOT}" \
  --parameters-root "${SCRATCH_PARAMS_ROOT}" \
  --output-root "${DOWNLOAD_DIR}" \
  --reference-root "${REFERENCE_ROOT}"
