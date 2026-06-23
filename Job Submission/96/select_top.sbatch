#!/bin/bash
#SBATCH --account=qzhuang_922
#SBATCH --partition=main,epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:20:00
#SBATCH --job-name=CI96_select_depth
#SBATCH --output=logs/select_%j.out
#SBATCH --error=logs/select_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load python/3.9.12

mkdir -p logs runs

python - << 'PY'
from pathlib import Path

ETA = '0.30'
DEPTHS = [2, 4, 6, 8, 10, 12, 14, 16, 18]
SEEDS = range(200)
NT = 30
TOPK = 20

base_runs = Path("runs")

def read_float(p: Path):
    try:
        return float(p.read_text().strip())
    except Exception:
        return None

for depth in DEPTHS:
    depth_tag = f"{depth:02d}"
    run_root = base_runs / f"eta={ETA}_depth={depth_tag}_Nt={NT}"
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

    print(f"depth={depth}: found {len(rows)} / 200 seeds with best_feasible_ci.txt (missing {missing})")
    print(f"  wrote: {out_path}")
    for rank, (v, seed) in enumerate(top, 1):
        print(f"    {rank:2d}) seed={seed:3d}  best_feasible_ci={v:.8f}")
PY
