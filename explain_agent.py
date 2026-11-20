# explain_agent.py : what does the RL agent actually do differently from a fixed PID?
# ---------------------------------------------------------------
# Usage:   .\run explain_agent.py
#
# 1. results/gain_schedule.png - one column per case study, the step test 0 -> 1:
#      top:    process output   (RL agent vs the best fixed PID)
#      middle: control input u
#      bottom: the gains the agent chooses, relative to the best fixed PID it started from
#              (Kp / Kp0 and tau_i / tau_i0; 1 = the fixed PID's value)
#    The agent shown per case is the improved_v6b seed with the best VALIDATION score.
# 2. results/control_effort.txt - how hard each controller works the actuator:
#      total variation TV = sum |u[k] - u[k-1]| (how much u moves) and peak |u|,
#      nominal and with sensor noise 0.01 (mean of 10 noise seeds). RL = mean of its 5 seeds.
# ---------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from cases import load_case
from core.tools import (Printer, completed, find_agents, fixed, load_agent, load_fixed_gains,
                        results_path, run_test, runs_folder, score, version_of)

CASES = ["cs1", "cs2", "cs3"]
TIME_SHOWN = {"cs1": 4.0, "cs2": 12.0, "cs3": 30.0}
RL_COLOUR, PID_COLOUR, RULE_COLOUR = "#2a78d6", "#eb6834", "#1baf7a"
show = Printer()


def best_validation_seed(cid):
    """The improved_v6b agent with the lowest validation score (never the test score)."""
    best = None
    for name, folder in find_agents(cid):
        if version_of(name) == "improved_v6b":
            last = open(os.path.join(folder, "validation.csv")).read().strip().splitlines()[-1]
            score_ = float(last.split(",")[2])                 # best_so_far = the kept checkpoint
            if best is None or score_ < best[0]:
                best = (score_, name)
    return best[1]


def effort(case, ctrl, noise):
    """(total variation of u, peak |u|) of the step test; noise runs averaged over 10 seeds."""
    seeds = range(10) if noise > 0 else [None]
    tv, peak = [], []
    for s in seeds:
        env, _ = run_test(case, ctrl, noise_std=noise, seed=s)
        u = env.u[case.WARMUP_STEPS - 1: env.k]
        tv.append(np.sum(np.abs(np.diff(u))))
        peak.append(np.max(np.abs(u)))
    return np.mean(tv), np.mean(peak)


fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex="col", constrained_layout=True)
show("Control effort on the step test 0 -> 1 (TV = total variation of u = sum |u[k] - u[k-1]|)")
show(f"{'':34s}{'TV nominal':>12s}{'TV noise 0.01':>15s}{'peak |u|':>10s}")
for col, cid in enumerate(CASES):
    case = load_case(cid)
    name = best_validation_seed(cid)
    agent = load_agent(case, name)
    g = load_fixed_gains(cid)["best_fixed_pid"]
    pid = fixed(g["Kp"], g["tau_i"], g["tau_d"])
    Kp0, ti0, _ = case.ANCHOR_PID

    env_rl, gains = run_test(case, agent)
    env_pid, _ = run_test(case, pid)
    t = np.arange(len(env_rl.r)) * case.DT
    ax_y, ax_u, ax_g = axes[:, col]
    ax_y.step(t, env_rl.r, "--", color="#52514e", lw=1, where="post", label="setpoint")
    ax_y.plot(t, env_pid.y, color=PID_COLOUR, lw=2, label="best fixed PID")
    ax_y.plot(t, env_rl.y, color=RL_COLOUR, lw=2, label=f"RL ({name})")
    ax_u.step(t, env_pid.u, color=PID_COLOUR, lw=2, where="post", label="best fixed PID")
    ax_u.step(t, env_rl.u, color=RL_COLOUR, lw=2, where="post", label="RL agent")
    t_moves = (case.WARMUP_STEPS - 1 + np.arange(len(gains)) * case.ACTION_REPEAT) * case.DT
    ax_g.axhline(1.0, color=PID_COLOUR, lw=2, label="best fixed PID (= 1)")
    ax_g.step(t_moves, gains[:, 0] / Kp0, color=RL_COLOUR, lw=2, where="post", label="RL: Kp / Kp0")
    ax_g.step(t_moves, gains[:, 1] / ti0, color=RL_COLOUR, lw=2, ls=":", where="post", label="RL: tau_i / tau_i0")
    ax_y.set_title(case.NAME.split("  ")[0], fontsize=10)
    for ax in (ax_y, ax_u, ax_g):
        ax.set_xlim(0.5 * case.WARMUP_STEPS * case.DT, TIME_SHOWN[cid])
        ax.grid(alpha=0.3)
    ax_g.set_xlabel("Time (s)" if case.DT < 1 else "Time (samples)")
    if col == 0:
        ax_y.set_ylabel("output y")
        ax_u.set_ylabel("control input u")
        ax_g.set_ylabel("gain / fixed-PID gain")
    for ax in (ax_y, ax_u, ax_g):
        ax.legend(fontsize=8, loc="upper right")

    # control effort: RL (all 5 seeds), best fixed PID, best published rule
    rules = []
    for rname, rg in case.CLASSICAL:
        e, _ = run_test(case, fixed(*rg))
        if completed(e):
            rules.append((score(e)[2], rname, rg))
    _, rule_name, rule_gains = min(rules)
    rl_agents = [load_agent(case, n) for n, _ in find_agents(cid) if version_of(n) == "improved_v6b"]
    rows = [("RL improved_v6b (5 seeds)", rl_agents), ("Best fixed PID", [pid]),
            (f"{rule_name}", [fixed(*rule_gains)])]
    show(f"--- {case.NAME}")
    for label, ctrls in rows:
        nominal = np.mean([effort(case, c, 0.0) for c in ctrls], axis=0)
        noisy = np.mean([effort(case, c, 0.01) for c in ctrls], axis=0)
        show(f"{label:34s}{nominal[0]:12.1f}{noisy[0]:15.1f}{nominal[1]:10.2f}")
fig.suptitle("What the agent does: it changes its gains during the response, mostly the integral time "
             "tau_i (bottom row: 1 = the fixed PID's gain)", fontsize=12)
fig.savefig(results_path("gain_schedule.png"), dpi=150)
show.save(results_path("control_effort.txt"))
