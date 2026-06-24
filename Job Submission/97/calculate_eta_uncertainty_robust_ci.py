#!/usr/bin/env python3
"""Post-evaluate eta-uncertainty robust coherent information for Job 97."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "Data_HPC"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

DEFAULT_DELTAS = [0.01, 0.03, 0.05]
DEFAULT_ETAS = np.around(np.arange(0.05, 1.0, 0.05), 2).tolist()
DEFAULT_SCHEMES = ["VQT", "GKP", "TMS-EA", "QT"]

NS_CONSTRAINT = 2.0
NP_CONSTRAINT = 2.0
VQT_RUN_ID = 84
VQT_DEPTH = 20
VQT_NT = 30
GKP_NT = 30
GKP_ROOT_CANDIDATES = [
    DATA_ROOT / "94" / "Data-Download",
    DATA_ROOT / "94" / "Data-Download-partial-10000",
    DATA_ROOT / "64-v2_2",
]

OUTPUT_FIELDS = [
    "scheme",
    "eta0",
    "delta",
    "eta_minus",
    "eta_plus",
    "ci_minus",
    "ci_plus",
    "ci_avg",
    "ci_nominal",
    "ci_loss_avg",
    "ci_worst",
    "source",
    "status",
    "notes",
]


def eta_label(eta):
    return f"{float(eta):.2f}"


def eta_folder(eta):
    return f"eta={eta_label(eta)}"


def finite_or_nan(value):
    if value is None:
        return math.nan
    value = float(value)
    return value if math.isfinite(value) else math.nan


def format_float(value):
    value = finite_or_nan(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.17g}"


def read_float(path):
    return float(path.read_text().strip())


def g_entropy(nbar):
    nbar = np.asarray(nbar, dtype=float)
    nbar = np.maximum(nbar, 0.0)
    out = np.zeros_like(nbar, dtype=float)
    mask = nbar > 0.0
    out[mask] = (
        (nbar[mask] + 1.0) * np.log2(nbar[mask] + 1.0)
        - nbar[mask] * np.log2(nbar[mask])
    )
    if out.shape == ():
        return float(out)
    return out


def pure_loss_energy_ci(eta, n_s=NS_CONSTRAINT):
    eta = float(np.clip(eta, 0.0, 1.0))
    return max(0.0, g_entropy(eta * n_s) - g_entropy((1.0 - eta) * n_s))


def tms_ea_matched_ci(eta, n_s=NS_CONSTRAINT, n_p=NP_CONSTRAINT):
    gain = n_p + 1.0
    eta_ea = 1.0 / (1.0 + (1.0 - eta) / (eta * gain))
    return pure_loss_energy_ci(eta_ea, n_s)


def symplectic_entropy_two_mode(a, b, k):
    delta = a * a + b * b - 2.0 * k * k
    det_v = (a * b - k * k) ** 2
    disc = delta * delta - 4.0 * det_v
    if disc < 0.0 and disc > -1e-10:
        disc = 0.0
    if disc < 0.0:
        return math.nan, "negative_symplectic_discriminant"

    sqrt_disc = math.sqrt(disc)
    nu_plus_sq = (delta + sqrt_disc) / 2.0
    nu_minus_sq = (delta - sqrt_disc) / 2.0
    if nu_plus_sq < -1e-10 or nu_minus_sq < -1e-10:
        return math.nan, "negative_symplectic_eigenvalue"

    nu_plus = math.sqrt(max(nu_plus_sq, 0.0))
    nu_minus = math.sqrt(max(nu_minus_sq, 0.0))
    if nu_plus < 0.5 - 1e-8 or nu_minus < 0.5 - 1e-8:
        return math.nan, f"unphysical_symplectic_eigenvalues={nu_plus:.8g},{nu_minus:.8g}"

    entropy = g_entropy(nu_plus - 0.5) + g_entropy(nu_minus - 0.5)
    return entropy, ""


def tms_ea_mismatched_ci(eta0, eta_t, n_s=NS_CONSTRAINT, n_p=NP_CONSTRAINT):
    gain = n_p + 1.0
    gain_prime = gain / (gain - (1.0 - eta0) * (gain - 1.0))

    alpha = (
        math.sqrt(gain_prime * (1.0 - eta_t) * gain)
        - math.sqrt((gain_prime - 1.0) * (gain - 1.0))
    )
    beta = (
        math.sqrt(gain_prime * (1.0 - eta_t) * (gain - 1.0))
        - math.sqrt((gain_prime - 1.0) * gain)
    )
    c_gain = math.sqrt(gain_prime * eta_t)

    # Vacuum covariance is I/2. For real coefficients,
    # q_out=c q_S+alpha q_P+beta q_A and p_out=c p_S+alpha p_P-beta p_A.
    a = n_s + 0.5
    b = c_gain * c_gain * (n_s + 0.5) + 0.5 * (alpha * alpha + beta * beta)
    k = c_gain * math.sqrt(n_s * (n_s + 1.0))
    joint_entropy, problem = symplectic_entropy_two_mode(a, b, k)
    if problem:
        return math.nan, problem

    raw_ci = g_entropy(b - 0.5) - joint_entropy
    ci = max(0.0, raw_ci)
    note = "frozen TMS-EA calibration from eta0; covariance Gaussian CI"
    if raw_ci < 0.0:
        note += f"; raw_ci={raw_ci:.8g} clipped_to_zero"

    if abs(eta_t - eta0) < 1e-12:
        matched = tms_ea_matched_ci(eta0, n_s, n_p)
        if abs(ci - matched) > 1e-8:
            note += f"; nominal_check_delta={ci - matched:.3e}"
    return ci, note


def import_torch_protocols():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import torch
    from QTorch.Transduction import (
        transduction_protocol_CoherentInfo_ECD_MM_EA,
        transduction_protocol_CoherentInfo_GKP2,
    )

    torch.set_num_threads(int(os.environ.get("JOB97_TORCH_THREADS", "1")))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch, transduction_protocol_CoherentInfo_ECD_MM_EA, transduction_protocol_CoherentInfo_GKP2


def tensor_scalar(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


class ProtocolEvaluator:
    def __init__(self):
        self._torch = None
        self._vqt_protocol = None
        self._gkp_protocol = None

    def _ensure_torch(self):
        if self._torch is None:
            (
                self._torch,
                self._vqt_protocol,
                self._gkp_protocol,
            ) = import_torch_protocols()

    def evaluate_vqt(self, eta0, eta_t):
        self._ensure_torch()
        param_path = DATA_ROOT / str(VQT_RUN_ID) / eta_folder(eta0) / "parameters_best_feasible.npy"
        saved_ci_path = DATA_ROOT / str(VQT_RUN_ID) / eta_folder(eta0) / "best_feasible_ci.txt"
        if not param_path.exists():
            return math.nan, math.nan, f"missing parameter file: {param_path}", "missing_parameter"

        parameters = np.load(param_path)
        parameter_tensor = self._torch.as_tensor(parameters, dtype=self._torch.float64)
        with self._torch.no_grad():
            ci, ns_in, np_in, _state_rs, _state_pa = self._vqt_protocol(
                float(eta_t), parameter_tensor, VQT_DEPTH, VQT_NT
            )
        notes = [f"frozen VQT parameters from run {VQT_RUN_ID} eta0={eta_label(eta0)}"]
        ns_value = tensor_scalar(ns_in)
        np_value = tensor_scalar(np_in)
        if ns_value > NS_CONSTRAINT + 1e-5 or np_value > NP_CONSTRAINT + 1e-5:
            notes.append(f"energy_check ns={ns_value:.8g} np={np_value:.8g}")
        stored = read_float(saved_ci_path) if saved_ci_path.exists() else math.nan
        return tensor_scalar(ci), stored, "; ".join(notes), "ok"

    def evaluate_gkp(self, source, summary_by_eta, eta0, eta_t):
        self._ensure_torch()
        row = summary_by_eta.get(eta_label(eta0))
        if row is None:
            return math.nan, math.nan, f"missing selected row for eta0={eta_label(eta0)}", "missing_parameter"

        eta_dir = source / eta_folder(eta0)
        param_path = eta_dir / "parameters.npy"
        if not param_path.exists():
            return math.nan, math.nan, f"missing parameter file: {param_path}", "missing_parameter"

        try:
            d1 = int(row["d1"])
            d2 = int(row["d2"])
            j2 = int(row["j2"])
            score = float(row["score"])
        except (KeyError, ValueError) as exc:
            return math.nan, math.nan, f"bad selection metadata: {exc}", "invalid_metadata"

        parameters = np.load(param_path)
        parameter_tensor = self._torch.as_tensor(parameters, dtype=self._torch.float64)
        with self._torch.no_grad():
            ci, ns_in, np_in, _state_rs, _state_p = self._gkp_protocol(
                float(eta_t), d1, d2, j2, parameter_tensor, GKP_NT, NR=d1
            )
        notes = [
            f"frozen GKP parameters from {source.relative_to(REPO_ROOT)} eta0={eta_label(eta0)}",
            f"d1={d1} d2={d2} j2={j2}",
        ]
        ns_value = tensor_scalar(ns_in)
        np_value = tensor_scalar(np_in)
        if ns_value > NS_CONSTRAINT + 1e-5 or np_value > NP_CONSTRAINT + 1e-5:
            notes.append(f"energy_check ns={ns_value:.8g} np={np_value:.8g}")
        return tensor_scalar(ci), score, "; ".join(notes), "ok"


def load_gkp_summary(source):
    summary_path = source / "selection_summary.tsv"
    if not summary_path.exists():
        return None
    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        eta_label(row["eta"]): row
        for row in rows
        if row.get("status", "").startswith("selected")
    }


def resolve_gkp_source():
    for source in GKP_ROOT_CANDIDATES:
        summary = load_gkp_summary(source)
        if summary:
            return source, summary
    return None, {}


def make_summary_row(scheme, eta0, delta, ci_minus, ci_plus, ci_nominal, source, status, notes):
    ci_minus = finite_or_nan(ci_minus)
    ci_plus = finite_or_nan(ci_plus)
    ci_nominal = finite_or_nan(ci_nominal)
    if math.isfinite(ci_minus) and math.isfinite(ci_plus):
        ci_avg = 0.5 * (ci_minus + ci_plus)
        ci_worst = min(ci_minus, ci_plus)
    else:
        ci_avg = math.nan
        ci_worst = math.nan
    ci_loss_avg = ci_nominal - ci_avg if math.isfinite(ci_nominal) and math.isfinite(ci_avg) else math.nan

    return {
        "scheme": scheme,
        "eta0": eta_label(eta0),
        "delta": eta_label(delta),
        "eta_minus": eta_label(np.clip(eta0 - delta, 0.0, 1.0)),
        "eta_plus": eta_label(np.clip(eta0 + delta, 0.0, 1.0)),
        "ci_minus": format_float(ci_minus),
        "ci_plus": format_float(ci_plus),
        "ci_avg": format_float(ci_avg),
        "ci_nominal": format_float(ci_nominal),
        "ci_loss_avg": format_float(ci_loss_avg),
        "ci_worst": format_float(ci_worst),
        "source": source,
        "status": status,
        "notes": notes,
    }


def evaluate_scheme_delta_eta(evaluator, scheme, eta0, delta, gkp_source, gkp_summary):
    eta_minus = float(np.clip(eta0 - delta, 0.0, 1.0))
    eta_plus = float(np.clip(eta0 + delta, 0.0, 1.0))
    notes = []
    status = "ok"

    if scheme == "QT":
        ci_minus = pure_loss_energy_ci(eta_minus)
        ci_plus = pure_loss_energy_ci(eta_plus)
        ci_nominal = pure_loss_energy_ci(eta0)
        source = "pure-loss energy-constrained formula"
        notes.append("true-eta capacity benchmark; no frozen nominal parameters")

    elif scheme == "TMS-EA":
        ci_minus, note_minus = tms_ea_mismatched_ci(eta0, eta_minus)
        ci_plus, note_plus = tms_ea_mismatched_ci(eta0, eta_plus)
        ci_nominal, note_nominal = tms_ea_mismatched_ci(eta0, eta0)
        source = "Gaussian covariance model with frozen nominal anti-squeezer"
        notes.extend([note_minus, note_plus, note_nominal])
        if not (math.isfinite(ci_minus) and math.isfinite(ci_plus) and math.isfinite(ci_nominal)):
            status = "invalid_covariance"

    elif scheme == "VQT":
        ci_minus, stored_ci, note_minus, status_minus = evaluator.evaluate_vqt(eta0, eta_minus)
        ci_plus, _stored_ci2, note_plus, status_plus = evaluator.evaluate_vqt(eta0, eta_plus)
        ci_nominal, stored_nominal, note_nominal, status_nominal = evaluator.evaluate_vqt(eta0, eta0)
        source = f"Data_HPC/{VQT_RUN_ID}"
        notes.extend([note_minus, note_plus, note_nominal])
        if status_minus != "ok" or status_plus != "ok" or status_nominal != "ok":
            status = status_minus if status_minus != "ok" else status_plus
        elif math.isfinite(stored_ci) and abs(ci_nominal - stored_nominal) > 1e-6:
            notes.append(f"nominal_recompute_warning stored={stored_nominal:.10g} recomputed={ci_nominal:.10g}")

    elif scheme == "GKP":
        if gkp_source is None:
            ci_minus = ci_plus = ci_nominal = math.nan
            source = "none"
            status = "missing_parameter"
            notes.append("no GKP selection_summary.tsv source found")
        else:
            ci_minus, stored_ci, note_minus, status_minus = evaluator.evaluate_gkp(gkp_source, gkp_summary, eta0, eta_minus)
            ci_plus, _stored_ci2, note_plus, status_plus = evaluator.evaluate_gkp(gkp_source, gkp_summary, eta0, eta_plus)
            ci_nominal, stored_nominal, note_nominal, status_nominal = evaluator.evaluate_gkp(gkp_source, gkp_summary, eta0, eta0)
            source = str(gkp_source.relative_to(REPO_ROOT))
            notes.extend([note_minus, note_plus, note_nominal])
            if status_minus != "ok" or status_plus != "ok" or status_nominal != "ok":
                status = status_minus if status_minus != "ok" else status_plus
            elif math.isfinite(stored_ci) and abs(ci_nominal - stored_nominal) > 1e-6:
                notes.append(f"nominal_recompute_warning stored={stored_nominal:.10g} recomputed={ci_nominal:.10g}")

    else:
        raise ValueError(f"unknown scheme: {scheme}")

    unique_notes = []
    for note in notes:
        if note and note not in unique_notes:
            unique_notes.append(note)
    return make_summary_row(
        scheme,
        eta0,
        delta,
        ci_minus,
        ci_plus,
        ci_nominal,
        source,
        status,
        "; ".join(unique_notes),
    )


def write_outputs(rows, output_dir, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "robust_ci_summary.csv"
    json_path = output_dir / "robust_ci_summary.json"
    config_path = output_dir / "config.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate eta-uncertainty robust coherent information."
    )
    parser.add_argument("--deltas", nargs="+", type=float, default=DEFAULT_DELTAS)
    parser.add_argument("--etas", nargs="+", type=float, default=DEFAULT_ETAS)
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "97")
    parser.add_argument("--schemes", nargs="+", choices=DEFAULT_SCHEMES, default=DEFAULT_SCHEMES)
    return parser.parse_args()


def main():
    args = parse_args()
    etas = [round(float(eta), 2) for eta in args.etas]
    deltas = [round(float(delta), 2) for delta in args.deltas]
    schemes = list(args.schemes)

    gkp_source, gkp_summary = resolve_gkp_source()
    if "GKP" in schemes:
        if gkp_source is None:
            print("Warning: no GKP selected-parameter source found")
        else:
            print(f"GKP source: {gkp_source.relative_to(REPO_ROOT)}")

    evaluator = ProtocolEvaluator()
    rows = []
    for scheme in schemes:
        for delta in deltas:
            for eta0 in etas:
                print(f"[Job97] scheme={scheme} delta={delta:.2f} eta0={eta0:.2f}")
                row = evaluate_scheme_delta_eta(
                    evaluator, scheme, eta0, delta, gkp_source, gkp_summary
                )
                rows.append(row)

    config = {
        "deltas": deltas,
        "etas": etas,
        "schemes": schemes,
        "n_s": NS_CONSTRAINT,
        "n_p": NP_CONSTRAINT,
        "vqt_run_id": VQT_RUN_ID,
        "vqt_depth": VQT_DEPTH,
        "vqt_nt": VQT_NT,
        "gkp_nt": GKP_NT,
        "gkp_source": None if gkp_source is None else str(gkp_source.relative_to(REPO_ROOT)),
        "output_dir": str(args.output),
        "notes": "Local Mac post-evaluation job; no training is performed.",
    }
    write_outputs(rows, args.output, config)


if __name__ == "__main__":
    main()
