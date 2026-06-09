#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:20:00
#SBATCH --job-name=CI_th_select
#SBATCH --output=logs/th_select_%j.out
#SBATCH --error=logs/th_select_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs runs

python - << 'PY'
import os
from pathlib import Path

ETAS = ['0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45',
        '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95']
N_M_LIST = ['0.00', '0.01', '0.03', '0.05', '0.10']
KAPPA_M_LIST = ['1.00', '0.99', '0.97', '0.95', '0.90']
SEEDS = range(20)
DEPTH = 20
NT = 30
TOPK = 20
KAPPA_O = '1.00'
N_O = '0.00'
SCAN_MODE = os.environ.get('SCAN_MODE', 'nm').strip().lower()

base_runs = Path("runs")

def read_float(p: Path):
    try:
        return float(p.read_text().strip())
    except Exception:
        return None

for eta in ETAS:
    for noise_idx in range(5):
        if SCAN_MODE == 'km':
            kappa_m = KAPPA_M_LIST[noise_idx]
            n_m = '0.00'
        else:
            kappa_m = '1.00'
            n_m = N_M_LIST[noise_idx]

        run_root = base_runs / (
            f"thermal_eta={eta}_depth={DEPTH}_Nt={NT}"
            f"_ko={KAPPA_O}_no={N_O}_km={kappa_m}_nm={n_m}"
        )
        rows = []
        missing = 0

        for seed in SEEDS:
            p = run_root / f"seed_{seed}" / "best_feasible_ci.txt"
            v = read_float(p)
            if v is None:
                missing += 1
                continue
            rows.append((v, seed))

        rows.sort(key=lambda t: t[0], reverse=True)
        top = rows[:TOPK]

        run_root.mkdir(parents=True, exist_ok=True)
        out_path = run_root / f"top{TOPK}_seeds.txt"
        with out_path.open("w") as f:
            for v, seed in top:
                f.write(f"{seed}\n")

        print(
            f"eta={eta} kappa_m={kappa_m} n_m={n_m}: "
            f"found {len(rows)} / {len(SEEDS)} seeds with best_feasible_ci.txt "
            f"(missing {missing})"
        )
        print(f"  wrote: {out_path}")
        for rank, (v, seed) in enumerate(top, 1):
            print(f"    {rank:2d}) seed={seed:3d}  best_feasible_ci={v:.8f}")
PY
