# plot_three.py : one figure with the step test of all three processes (easy, medium, hard)
# ---------------------------------------------------------------
# Usage:   .\run plot_three.py [folder with authors_cs1.npz / authors_cs3.npz]
#
# One panel per process: the 5 improved_v6b agents (thin lines), the best published tuning rule
# (smallest ISE of the case's CLASSICAL list), the best fixed PID (grid search) and - only if the
# optional folder is given - the authors' recorded agent response. Those .npz files hold the
# arrays r, y, u exported from the authors' results (matlab_matrix.mat, Code Ocean capsule
# 8627992); they are not part of this repository.
# Saved to results/three_processes.png.
# ---------------------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from cases import load_case
from core.tools import completed, find_agents, fixed, load_agent, load_fixed_gains, results_path, run_test, score, version_of

AUTHORS_DIR = sys.argv[1] if len(sys.argv) > 1 else None
PANELS = [("cs1", "Easy - CS1: 1/(s - 1)", 6.0),
          ("cs2", "Medium - CS2: e^(-0.4s)/(s - 1)", 20.0),
          ("cs3", "Hard - CS3: (0.77248s - 0.1727)/(3.1s - 1)", 60.0)]
Y_TOP = {"cs1": 1.7, "cs2": 2.9, "cs3": 2.5}           # top of each panel's y axis

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (cid, title, t_end) in zip(axes, PANELS):
    case = load_case(cid)
    t = np.arange(case.WARMUP_STEPS + case.HOLD_STEPS) * case.DT

    # best published rule = smallest ISE among the rules that complete the test
    rules = []
    for name, gains in case.CLASSICAL:
        env, _ = run_test(case, fixed(*gains))
        if completed(env):
            rules.append((score(env)[2], name, env))
    _, rule_name, rule_env = min(rules, key=lambda x: x[0])
    ax.plot(t, rule_env.y, color="tab:orange", lw=2, label=f"{rule_name}")

    g = load_fixed_gains(cid)["best_fixed_pid"]
    env, _ = run_test(case, fixed(g["Kp"], g["tau_i"], g["tau_d"]))
    ax.plot(t, env.y, color="tab:green", lw=2, ls="--", label="Best fixed PID (grid search)")

    if AUTHORS_DIR and os.path.exists(os.path.join(AUTHORS_DIR, f"authors_{cid}.npz")):
        y = np.load(os.path.join(AUTHORS_DIR, f"authors_{cid}.npz"))["y"]
        ax.plot(t[: len(y)], y, color="tab:red", lw=2, label="Authors' agent (their results)")

    agents = [name for name, _ in find_agents(cid) if version_of(name) == "improved_v6b"]
    for i, name in enumerate(agents):
        env, _ = run_test(case, load_agent(case, name))
        ax.plot(t, env.y, color="tab:blue", lw=1.2, alpha=0.8,
                label=f"RL improved_v6b ({len(agents)} seeds)" if i == 0 else None)

    ax.step(t, rule_env.r, "--", color="black", lw=1, where="post", label="Setpoint")
    ax.set_title(title)
    ax.set_xlim(0, t_end)
    ax.set_ylim(-0.1, Y_TOP[cid])
    ax.set_xlabel("Time (s)" if case.DT < 1 else "Time (samples)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("Process output y")
fig.tight_layout()
fig.savefig(results_path("three_processes.png"), dpi=150)
print("(saved to results/three_processes.png)")
