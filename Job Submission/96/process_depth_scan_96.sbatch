#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:20:00
#SBATCH --job-name=CI96_process
#SBATCH --output=logs/process_%j.out
#SBATCH --error=logs/process_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs

python process_depth_scan_96.py
