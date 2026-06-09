import os
import numpy as np
import torch
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

from QTorch.Transduction import transduction_protocol_CoherentInfo_ECD_MM_EA

num_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
torch.set_num_threads(num_cpu)
torch.set_num_interop_threads(1)

# -----------------------
# Fixed experiment config
# -----------------------
statetype = "_CoherentInfo_EA_"
Nt = 30
eta = float(os.environ.get("ETA", "0.25"))
depth = 20

disp = True
disp_frequency = 100
save = True
save_frequency = 10

ns_constraint = 3
np_constraint = 3
n_coefficient = 10.0


STD_DEV_ALPHAS_DEFAULT = 0.01
STD_DEV_ALPHAS_ENV = "STD_DEV_ALPHAS"   # stage1.sbatch exports this
std_dev_alphas = float(os.environ.get(STD_DEV_ALPHAS_ENV, STD_DEV_ALPHAS_DEFAULT))

#state_RS = np.load(f"Input_state/Nonadaptive_VQT_EA_ns=np=3_eta={eta}_state_RS.npy")
#state_PA = np.load(f"Input_state/Nonadaptive_VQT_EA_ns=np=3_eta={eta}_state_PA.npy")
#state_initial_RS = torch.tensor(state_RS)
#state_initial_PA = torch.tensor(state_PA)

# Penalty schedule:
#   P = 0  -> always on
#   P > 0  -> OFF for steps 0..P-1, ON for P..2P-1, OFF for 2P..3P-1, ...
P = 0

# -----------------------
# Early-abort config
# -----------------------
# If current CI is < EARLY_ABORT_CI_THRESH at (or after) EARLY_ABORT_STEP, abort training.
EARLY_ABORT_STEP = 2000
EARLY_ABORT_CI_THRESH = 0.05

# -----------------------
# Optimizer / scheduler config
# -----------------------
LR_INIT = 0.05
STEP_SIZE = 1000
GAMMA = 0.5
LR_MIN = 1e-4

# -----------------------
# LR bump-on-plateau config
# -----------------------

# Bump is only ACTIVE after LR_BUMP_START_STEP steps.
LR_BUMP_START_STEP = 10000
# If best-CI improvement over last LR_PLATEAU_WINDOW steps is < LR_PLATEAU_IMPROV_THRESH,
# then multiply LR by LR_BUMP_RATIO once (with cooldown), and keep decaying with the same StepLR schedule.
LR_PLATEAU_WINDOW = 1000
LR_PLATEAU_IMPROV_THRESH = 0.05
LR_BUMP_RATIO = 2.0
LR_BUMP_COOLDOWN_STEPS = 2000  # prevent repeated bumps too frequently; set 0 to allow frequent bumps
LR_MAX = 0.05  # safety cap; set to None to disable

# feasibility tolerance
FEAS_TOL = 0.01


def penalty_energy(n, n_constraint, ncoefficient):
    n_constraint = torch.tensor(n_constraint, dtype=torch.float64)
    ncoefficient = torch.tensor(ncoefficient, dtype=torch.float64)
    return ncoefficient * (n - n_constraint) ** 2


def target(x, apply_penalty: bool):
    CI, ns_input, np_input, state_RS, state_PA = (
        transduction_protocol_CoherentInfo_ECD_MM_EA(
            eta, x, depth, Nt
        )
    )

    if apply_penalty:
        pen = (penalty_energy(ns_input, ns_constraint, n_coefficient)
               + penalty_energy(np_input, np_constraint, n_coefficient))
    else:
        pen = torch.zeros((), dtype=CI.dtype, device=CI.device)

    loss = -CI + pen
    return loss, CI, ns_input, np_input, state_RS, state_PA


def _build_best_so_far(ci_list):
    best = -1e99
    out = []
    for v in ci_list:
        if v > best:
            best = v
        out.append(best)
    return out


def _maybe_bump_lr(optimizer, nitern, best_ci_so_far, last_bump_iter):
    """
    Uses best_ci_so_far[k] = max_{t<=k} CI(t).
    Plateau test: improvement over last LR_PLATEAU_WINDOW steps is:
        best_ci_so_far[n] - best_ci_so_far[n - LR_PLATEAU_WINDOW]
    If < thresh -> bump LR by ratio, keep StepLR schedule unchanged.
    """
    # bump not active until a warmup period is over
    if nitern < LR_BUMP_START_STEP:
        return last_bump_iter, False, 0.0

    if LR_PLATEAU_WINDOW <= 0:
        return last_bump_iter, False, 0.0

    if nitern < LR_PLATEAU_WINDOW:
        return last_bump_iter, False, 0.0

    if LR_BUMP_COOLDOWN_STEPS > 0 and (nitern - last_bump_iter) < LR_BUMP_COOLDOWN_STEPS:
        return last_bump_iter, False, 0.0

    improv = best_ci_so_far[nitern] - best_ci_so_far[nitern - LR_PLATEAU_WINDOW]
    if improv >= LR_PLATEAU_IMPROV_THRESH:
        return last_bump_iter, False, float(improv)

    # bump
    for pg in optimizer.param_groups:
        new_lr = pg["lr"] * LR_BUMP_RATIO
        if LR_MAX is not None:
            new_lr = min(new_lr, LR_MAX)
        pg["lr"] = new_lr

    return nitern, True, float(improv)


def _save_latest(run_dir, filepath_prefix, ckpt_latest_path, payload, ci_list, ns_list, np_list):
    # NPY lists
    np.save(filepath_prefix + "ci_list.npy", np.array(ci_list))
    np.save(filepath_prefix + "ns_list.npy", np.array(ns_list))
    np.save(filepath_prefix + "np_list.npy", np.array(np_list))
    # CKPT
    torch.save(payload, ckpt_latest_path)


def main():
    # -----------------------
    # Controlled via ENV vars
    # -----------------------
    seed = int(os.environ.get("SEED", "0"))
    max_steps = int(os.environ.get("MAX_STEPS", "20000"))
    eta_tag = os.environ.get("ETA_TAG")
    if eta_tag is None:
        eta_tag = f"{eta:.6g}"


  
    run_dir = os.environ.get("RUN_DIR",f"runs/eta={eta_tag}_depth={depth}_Nt={Nt}/seed_{seed}")
    # -----------------------
    # Top-level training_parameters directory
    # -----------------------
    ckpt_dir = os.environ.get("CKPT_DIR",f"training_parameters/eta={eta_tag}_depth={depth}_Nt={Nt}/seed_{seed}")

    resume = (os.environ.get("RESUME", "0") == "1")
    resume_from = os.environ.get("RESUME_FROM", "latest").strip().lower()
    # RESUME_FROM: latest | best | best_feasible

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -----------------------
    # File paths
    # -----------------------
    filepath_prefix = os.path.join(run_dir, "data_")

    ckpt_latest_path = os.path.join(ckpt_dir, "ckpt_latest.pt")
    ckpt_best_path = os.path.join(ckpt_dir, "ckpt_best.pt")
    ckpt_best_feas_path = os.path.join(ckpt_dir, "ckpt_best_feasible.pt")

    best_ci_path = os.path.join(run_dir, "best_ci.txt")
    best_feas_ci_path = os.path.join(run_dir, "best_feasible_ci.txt")

    # Best feasible physics objects (kept with checkpoints)
    rs_best_feas_path = os.path.join(ckpt_dir, "state_RS_best_feasible.npy")
    pa_best_feas_path = os.path.join(ckpt_dir, "state_PA_best_feasible.npy")
    par_best_feas_path = os.path.join(ckpt_dir, "parameters_best_feasible.npy")

    # Decide which checkpoint to resume from
    if resume_from == "best":
        resume_path = ckpt_best_path
    elif resume_from == "best_feasible":
        resume_path = ckpt_best_feas_path
    else:
        resume_path = ckpt_latest_path

    time1 = datetime.now()

    # -----------------------
    # Init or resume
    # -----------------------
    if resume and os.path.exists(resume_path):
        payload = torch.load(resume_path, map_location="cpu")

        x = payload["x"].detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=LR_INIT)
        optimizer.load_state_dict(payload["optimizer"])

        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        scheduler.load_state_dict(payload["scheduler"])

        ci_list = list(payload["ci_list"])
        ns_list = list(payload["ns_list"])
        np_list = list(payload["np_list"])

        start_iter = int(payload["nitern"]) + 1

        best_ci = float(payload.get("best_ci", -1e99))
        best_feas_ci = float(payload.get("best_feasible_ci", -1e99))

        best_ci_so_far = list(payload.get("best_ci_so_far", []))
        if len(best_ci_so_far) != len(ci_list):
            best_ci_so_far = _build_best_so_far(ci_list)

        last_bump_iter = int(payload.get("last_bump_iter", -10**18))

        print(
            f"[RESUME] seed={seed} from={resume_from} start_iter={start_iter} "
            f"best_ci={best_ci:.8f} best_feas_ci={best_feas_ci:.8f}\n"
            f"         run_dir={run_dir}\n"
            f"         ckpt_dir={ckpt_dir}\n"
            f"         P={P} (penalty period)\n"
            f"         early_abort: step>={EARLY_ABORT_STEP} and CI<{EARLY_ABORT_CI_THRESH}\n"
            f"         lr_bump_start={LR_BUMP_START_STEP}\n"
            f"         plateau_window={LR_PLATEAU_WINDOW} plateau_thresh={LR_PLATEAU_IMPROV_THRESH} "
            f"bump_ratio={LR_BUMP_RATIO} cooldown={LR_BUMP_COOLDOWN_STEPS}"
        )

    else:
        rng = np.random.default_rng(seed)
        

        alphas = rng.standard_normal(12 * depth) * std_dev_alphas
        betas = rng.uniform(-np.pi, np.pi, 12 * depth)
        x = np.concatenate((alphas, betas))
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)

        optimizer = torch.optim.Adam([x], lr=LR_INIT)
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

        ci_list, ns_list, np_list = [], [], []
        best_ci_so_far = []
        start_iter = 0

        best_ci = -1e99
        best_feas_ci = -1e99
        last_bump_iter = -10**18

        with open(best_ci_path, "w") as f:
            f.write(f"{best_ci}\n")
        with open(best_feas_ci_path, "w") as f:
            f.write(f"{best_feas_ci}\n")

        print(
            f"[INIT] seed={seed} max_steps={max_steps}\n"
            f"       run_dir={run_dir}\n"
            f"       ckpt_dir={ckpt_dir}\n"
            f"       P={P} (penalty period)\n"
            f"       early_abort: step>={EARLY_ABORT_STEP} and CI<{EARLY_ABORT_CI_THRESH}\n"
            f"       lr_bump_start={LR_BUMP_START_STEP}\n"
            f"       plateau_window={LR_PLATEAU_WINDOW} plateau_thresh={LR_PLATEAU_IMPROV_THRESH} "
            f"bump_ratio={LR_BUMP_RATIO} cooldown={LR_BUMP_COOLDOWN_STEPS}"
        )

    # -----------------------
    # Training loop
    # -----------------------
    for nitern in range(start_iter, max_steps):
        apply_penalty = (P == 0) or ((nitern % (2 * P)) >= P)

        loss, CI, ns_input, np_input, state_RS, state_PA = target(x, apply_penalty)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ci_val = float(CI.detach())
        ns_val = float(ns_input.detach())
        np_val = float(np_input.detach())

        ci_list.append(ci_val)
        ns_list.append(ns_val)
        np_list.append(np_val)

        # update best-so-far list
        if nitern == 0 and len(best_ci_so_far) == 0:
            best_ci_so_far.append(ci_val)
        else:
            prev_best = best_ci_so_far[-1] if len(best_ci_so_far) > 0 else -1e99
            best_ci_so_far.append(max(prev_best, ci_val))

        is_feasible = (
            ns_val <= ns_constraint + FEAS_TOL
            and np_val <= np_constraint + FEAS_TOL
        )

        # ---------- Early abort ----------
        if nitern >= EARLY_ABORT_STEP and ci_val < EARLY_ABORT_CI_THRESH:
            # Save a final "latest" snapshot so the run is inspectable
            payload = {
                "nitern": nitern,
                "x": x.detach(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ci_list": ci_list,
                "ns_list": ns_list,
                "np_list": np_list,
                "best_ci": best_ci,
                "best_feasible_ci": best_feas_ci,
                "best_ci_so_far": best_ci_so_far,
                "last_bump_iter": last_bump_iter,
                "aborted_early": True,
                "abort_reason": f"CI<{EARLY_ABORT_CI_THRESH} at step {nitern} (CI={ci_val})",
            }
            if save:
                _save_latest(run_dir, filepath_prefix, ckpt_latest_path, payload, ci_list, ns_list, np_list)

            lr = optimizer.param_groups[0]["lr"]
            elapsed = datetime.now() - time1
            print(
                f"[EARLY_ABORT] seed={seed} iter={nitern} "
                f"ci={ci_val:.8f} (<{EARLY_ABORT_CI_THRESH}) "
                f"ns={ns_val:.4f} np={np_val:.4f} feas={is_feasible} "
                f"lr={lr:.3e} elapsed={elapsed}\n"
                f"             saved_latest={save} ckpt={ckpt_latest_path}"
            )
            break

        # ---------- Best overall ----------
        if ci_val > best_ci:
            best_ci = ci_val
            with open(best_ci_path, "w") as f:
                f.write(f"{best_ci}\n")

            torch.save({
                "nitern": nitern,
                "x": x.detach(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ci_list": ci_list,
                "ns_list": ns_list,
                "np_list": np_list,
                "best_ci": best_ci,
                "best_feasible_ci": best_feas_ci,
                "best_ci_so_far": best_ci_so_far,
                "last_bump_iter": last_bump_iter,
            }, ckpt_best_path)

        # ---------- Best feasible ----------
        if is_feasible and ci_val > best_feas_ci:
            best_feas_ci = ci_val
            with open(best_feas_ci_path, "w") as f:
                f.write(f"{best_feas_ci}\n")

            np.save(rs_best_feas_path, state_RS.detach().numpy())
            np.save(pa_best_feas_path, state_PA.detach().numpy())
            np.save(par_best_feas_path, x.detach().numpy())

            torch.save({
                "nitern": nitern,
                "x": x.detach(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ci_list": ci_list,
                "ns_list": ns_list,
                "np_list": np_list,
                "best_ci": best_ci,
                "best_feasible_ci": best_feas_ci,
                "feasible": True,
                "best_ci_so_far": best_ci_so_far,
                "last_bump_iter": last_bump_iter,
            }, ckpt_best_feas_path)

        # ---------- LR bump on plateau (after updating best_ci_so_far) ----------
        last_bump_iter, did_bump, improv = _maybe_bump_lr(
            optimizer=optimizer,
            nitern=nitern,
            best_ci_so_far=best_ci_so_far,
            last_bump_iter=last_bump_iter
        )

        # ---------- Periodic save ----------
        if save and (nitern % save_frequency == 0 or nitern == max_steps - 1):
            payload = {
                "nitern": nitern,
                "x": x.detach(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ci_list": ci_list,
                "ns_list": ns_list,
                "np_list": np_list,
                "best_ci": best_ci,
                "best_feasible_ci": best_feas_ci,
                "best_ci_so_far": best_ci_so_far,
                "last_bump_iter": last_bump_iter,
            }
            _save_latest(run_dir, filepath_prefix, ckpt_latest_path, payload, ci_list, ns_list, np_list)

        # StepLR decay (unchanged schedule)
        if optimizer.param_groups[0]["lr"] > LR_MIN:
            scheduler.step()

        if disp and nitern > 1 and nitern % disp_frequency == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = datetime.now() - time1
            bump_msg = ""
            if nitern < LR_BUMP_START_STEP:
                bump_msg = f" [LR_BUMP_INACTIVE until {LR_BUMP_START_STEP}]"
            elif did_bump:
                bump_msg = f" [LR_BUMP] improv({LR_PLATEAU_WINDOW})={improv:.3e} < {LR_PLATEAU_IMPROV_THRESH:.3e}"
            print(
                f"seed={seed} iter={nitern} "
                f"ci={ci_val:.8f} ns={ns_val:.4f} np={np_val:.4f} "
                f"feas={is_feasible} penalty_on={apply_penalty} "
                f"lr={lr:.3e} elapsed={elapsed}{bump_msg}"
            )

    print(
        f"[DONE] seed={seed} "
        f"best_ci={best_ci:.8f} best_feasible_ci={best_feas_ci:.8f}\n"
        f"       run_dir={run_dir}\n"
        f"       ckpt_dir={ckpt_dir}"
    )


if __name__ == "__main__":
    main()

