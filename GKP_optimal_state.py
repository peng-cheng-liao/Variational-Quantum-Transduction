import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import torch

from QTorch.Transduction import transduction_protocol_CoherentInfo_GKP2


DATA_ROOT = Path("Data_HPC/64-v2_2")
NT = 30


def as_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value))


def eta_folder_name(eta):
    return f"eta={float(eta):.2f}"


def load_eta_metadata(eta_dir):
    info_path = eta_dir / "source_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {info_path}")
    return json.loads(info_path.read_text())


def verify_eta(eta_dir, atol):
    info = load_eta_metadata(eta_dir)
    eta = float(info["eta"])
    d1 = int(info["d1"])
    d2 = int(info["d2"])
    j2 = int(info["j2"])
    best_index = int(info["best_index"])
    recorded_score = float(info["score"])

    parameters = np.load(eta_dir / "parameters.npy")
    ci_list = np.load(eta_dir / "ci_list.npy")
    ns_list = np.load(eta_dir / "ns_list.npy")
    np_list = np.load(eta_dir / "np_list.npy")

    with torch.no_grad():
        ci, ns_input, np_input, _, _ = transduction_protocol_CoherentInfo_GKP2(
            eta, d1, d2, j2, parameters, NT, NR=d1
        )

    ci_value = as_float(ci)
    ns_value = as_float(ns_input)
    np_value = as_float(np_input)
    saved_ci = float(ci_list[best_index])
    saved_ns = float(ns_list[best_index])
    saved_np = float(np_list[best_index])

    return {
        "eta": info["eta"],
        "d1": d1,
        "d2": d2,
        "j2": j2,
        "best_index": best_index,
        "ci_value": ci_value,
        "saved_ci": saved_ci,
        "ci_abs_diff": abs(ci_value - saved_ci),
        "score": recorded_score,
        "score_abs_diff": abs(ci_value - recorded_score),
        "ns_value": ns_value,
        "saved_ns": saved_ns,
        "ns_abs_diff": abs(ns_value - saved_ns),
        "np_value": np_value,
        "saved_np": saved_np,
        "np_abs_diff": abs(np_value - saved_np),
        "ok": (
            abs(ci_value - saved_ci) <= atol
            and abs(ci_value - recorded_score) <= atol
            and abs(ns_value - saved_ns) <= atol
            and abs(np_value - saved_np) <= atol
        ),
    }


def iter_eta_dirs(data_root, requested_etas, verify_all):
    if verify_all:
        for eta_dir in sorted(data_root.glob("eta=*")):
            if eta_dir.is_dir():
                yield eta_dir
        return

    if requested_etas:
        for eta in requested_etas:
            yield data_root / eta_folder_name(eta)
        return

    yield data_root / eta_folder_name(0.30)


def print_result(result):
    status = "OK" if result["ok"] else "FAIL"
    print(
        f"{status} eta={result['eta']} "
        f"d1={result['d1']} d2={result['d2']} j2={result['j2']} "
        f"best_index={result['best_index']}"
    )
    print(
        "  CI: "
        f"computed={result['ci_value']:.17g} "
        f"saved={result['saved_ci']:.17g} "
        f"score={result['score']:.17g} "
        f"diff_saved={result['ci_abs_diff']:.3e} "
        f"diff_score={result['score_abs_diff']:.3e}"
    )
    print(
        "  ns/np: "
        f"computed=({result['ns_value']:.17g}, {result['np_value']:.17g}) "
        f"saved=({result['saved_ns']:.17g}, {result['saved_np']:.17g}) "
        f"diff=({result['ns_abs_diff']:.3e}, {result['np_abs_diff']:.3e})"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify transduction_protocol_CoherentInfo_GKP2 against selected "
            "Data_HPC/64-v2_2 GKP outputs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--eta",
        type=float,
        action="append",
        help="Eta value to verify. Can be passed multiple times. Defaults to eta=0.30.",
    )
    parser.add_argument("--all", action="store_true", help="Verify every eta folder.")
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if args.all and args.eta:
        raise ValueError("Use either --all or one or more --eta values, not both.")

    results = []
    for eta_dir in iter_eta_dirs(args.data_root, args.eta, args.all):
        if not eta_dir.exists():
            raise FileNotFoundError(f"Eta folder does not exist: {eta_dir}")
        result = verify_eta(eta_dir, args.atol)
        print_result(result)
        results.append(result)

    failures = [result["eta"] for result in results if not result["ok"]]
    print(f"Verified eta folders: {len(results)}")
    if failures:
        raise SystemExit(f"Verification failed for eta: {', '.join(failures)}")
    print("All requested verifications passed.")


if __name__ == "__main__":
    main()
