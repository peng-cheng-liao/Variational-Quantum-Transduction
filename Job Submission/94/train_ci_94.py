#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from QTorch.Transduction import transduction_protocol_CoherentInfo_GKP2


torch.autograd.set_detect_anomaly(True)

BASE_DIR = Path(__file__).resolve().parent

Nt = 30
DEFAULT_N = 5000
disp = True
disp_frequency = 100
save_frequency = 10
ns_constraint = 2.0
np_constraint = 2.0
energy_tol = 0.05
n_coefficient = 10
learning_rate = 0.001


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_int(name, default):
    return int(os.environ.get(name, default))


eta = env_float("ETA", "0.10")
eta_source = os.environ.get("ETA_TAG", os.environ.get("ETA", "0.10"))
randomization = env_int("RANDOMIZATION", "0")
d1 = env_int("D1", "2")
d2 = env_int("D2", "1")
j2 = env_int("J2", "0")
NR = d1

dirpath = BASE_DIR / "Data" / f"eta={eta_source}_d1={d1}_d2={d2}"
filename_prefix = f"j2={j2}_randomization={randomization}_"
filepath = dirpath / filename_prefix


def penalty_energy(n, n_constraint, ncoefficient):
    n_constraint_tensor = torch.as_tensor(
        n_constraint, dtype=n.dtype, device=n.device
    )
    ncoefficient_tensor = torch.as_tensor(
        ncoefficient, dtype=n.dtype, device=n.device
    )
    return ncoefficient_tensor * (n - n_constraint_tensor) ** 2


def target(x):
    CI, ns_input, np_input, state_RS, state_P = transduction_protocol_CoherentInfo_GKP2(
        eta, d1, d2, j2, x, Nt, NR
    )
    loss = (
        -CI
        + penalty_energy(ns_input, ns_constraint, n_coefficient)
        + penalty_energy(np_input, np_constraint, n_coefficient)
    )
    return loss, CI, ns_input, np_input, state_RS, state_P


def save_array(path_stem, value):
    np.save(str(path_stem), value)


def save_text(path, value):
    Path(path).write_text(f"{value}\n")


def save_histories(ci_list, ns_list, np_list):
    save_array(filepath.with_name(filename_prefix + "ci_list"), np.asarray(ci_list))
    save_array(filepath.with_name(filename_prefix + "ns_list"), np.asarray(ns_list))
    save_array(filepath.with_name(filename_prefix + "np_list"), np.asarray(np_list))


def save_best_snapshot(x, state_RS, state_P, nitern, ci_value, ns_value, np_value):
    save_array(
        filepath.with_name(filename_prefix + "best_parameters"),
        x.detach().cpu().numpy(),
    )
    save_array(
        filepath.with_name(filename_prefix + "best_state_RS"),
        state_RS.detach().cpu().numpy(),
    )
    save_array(
        filepath.with_name(filename_prefix + "best_state_P"),
        state_P.detach().cpu().numpy(),
    )
    save_text(filepath.with_name(filename_prefix + "best_index.txt"), nitern)
    save_text(filepath.with_name(filename_prefix + "best_feasible_ci.txt"), ci_value)
    save_text(filepath.with_name(filename_prefix + "best_ns.txt"), ns_value)
    save_text(filepath.with_name(filename_prefix + "best_np.txt"), np_value)


def minimize_target(num_steps=DEFAULT_N, do_save=True):
    dirpath.mkdir(parents=True, exist_ok=True)
    time1 = datetime.now()

    x = torch.rand(8, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)

    ci_list = []
    ns_list = []
    np_list = []
    best_feasible_ci = -np.inf
    best_index = None

    for nitern in range(num_steps):
        optimizer.zero_grad()
        loss, CI, ns_input, np_input, state_RS, state_P = target(x)

        ci_value = float(CI.detach().cpu())
        ns_value = float(ns_input.detach().cpu())
        np_value = float(np_input.detach().cpu())

        ci_list.append(ci_value)
        ns_list.append(ns_value)
        np_list.append(np_value)

        feasible = (
            ns_value <= ns_constraint + energy_tol
            and np_value <= np_constraint + energy_tol
        )

        if do_save and feasible and ci_value > best_feasible_ci:
            best_feasible_ci = ci_value
            best_index = nitern
            print(
                "save best feasible data",
                "iteration:", nitern,
                "ci:", ci_value,
                "ns:", ns_value,
                "np:", np_value,
            )
            save_best_snapshot(
                x, state_RS, state_P, nitern, ci_value, ns_value, np_value
            )

        loss.backward()
        optimizer.step()

        if optimizer.param_groups[0]["lr"] > 1e-6:
            scheduler.step()

        if do_save and (nitern % save_frequency == 0 or nitern == num_steps - 1):
            save_histories(ci_list, ns_list, np_list)

        if nitern > 1 and nitern % disp_frequency == 0 and disp:
            print(
                "Nt:", Nt,
                "eta:", eta,
                "ns_constraint:", ns_constraint,
                "np_constraint:", np_constraint,
                "iteration:", nitern,
                "ci:", ci_value,
                "ns:", ns_value,
                "np:", np_value,
                "best_feasible_ci:", best_feasible_ci,
                "best_index:", best_index,
                "learning rate:", optimizer.param_groups[0]["lr"],
                "running time:", datetime.now() - time1,
                "remaining time:", (datetime.now() - time1) * (num_steps - nitern) / nitern,
            )

    if do_save:
        save_histories(ci_list, ns_list, np_list)
        save_array(
            filepath.with_name(filename_prefix + "final_parameters"),
            x.detach().cpu().numpy(),
        )

    return ci_list, ns_list, np_list, best_feasible_ci, best_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GKP coherent-information protocol for HPC job 94."
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.environ.get("NUM_STEPS", DEFAULT_N)),
        help="Number of optimizer steps. Defaults to NUM_STEPS or 5000.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training without writing output files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    minimize_target(num_steps=args.num_steps, do_save=not args.no_save)


if __name__ == "__main__":
    main()
