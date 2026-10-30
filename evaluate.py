# evaluate.py : every trained agent vs the reference controllers on the step test
# ---------------------------------------------------------------
# Usage:   .\run evaluate.py <case>          e.g.  .\run evaluate.py cs1
#
# Prints one row per controller and the average per version, then saves
#   results/<case>_evaluate.txt     the table
#   results/<case>_comparison.png   process output (top) and Kp chosen by each agent (bottom)
# Columns: overshoot %, settling time (2 % band), ISE, IAE, whether the run completed, and
# whether the row beats the authors' agent on all four metrics (if the case gives their numbers).
# The "Best fixed PID" row needs grid_search.py to have been run once.
# ---------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from cases import load_case
from core.tools import (Printer, beats, completed, find_agents, load_agent, load_fixed_gains,
                        reference_controllers, results_path, run_test, score, version_of)

HIDE = ["improved_v3"]        # versions left out of this table and plot (their agents stay in runs/)

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "cs1")
show = Printer()

# ===== 1. The controllers to test: (name, controller, version) =====
controllers = [(name, ctrl, "paper") for name, ctrl in reference_controllers(case)]
best = load_fixed_gains(case.ID)
if best:
    g = best["best_fixed_pid"]
    show(f"Best fixed PID (from grid_search.py): Kp = {g['Kp']:.2f}, tau_i = {g['tau_i']:.2f}, "
         f"tau_d = {g['tau_d']:.2f}")
else:
    show("(Run grid_search.py once to add the 'Best fixed PID' row.)")
for name, _ in find_agents(case.ID):
    if version_of(name) not in HIDE:
        controllers.append(("RL " + name, load_agent(case, name), version_of(name)))

# ===== 2. Run the step test once per controller =====
# results: name -> (finished world, gains used at every move)   (like a Java LinkedHashMap)
results = {name: run_test(case, ctrl, version) for name, ctrl, version in controllers}


# ===== 3. The table =====
def row(name, m, done, beat):
    """One table line: name, the four metrics, completed, beats authors."""
    return f"{name:32s} {m[0]:11.1f} {m[1]:10.2f} {m[2]:9.3f} {m[3]:9.3f} {done:>9s} {beat:>22s}"


show(f"Step test {case.START_SETPOINT} -> {case.TEST_SETPOINT} on {case.NAME}")
show(f"{'Controller':32s} {'Overshoot %':>11s} {'Settling s':>10s} {'ISE':>9s} {'IAE':>9s} "
     f"{'Completed':>9s} {'Beats authors on all 4':>22s}")
for name, (env, _) in results.items():
    m = score(env)
    show(row(name, m, "yes" if completed(env) else "no", beats(m, case.AUTHORS)))
if case.AUTHORS:
    show(row("Authors' agent (their files)", case.AUTHORS, "-", "-"))

# Average per version, for versions with 2 or more seeds.
groups = {}                                   # version -> list of metric arrays
for name, (env, _) in results.items():
    if name.startswith("RL "):
        groups.setdefault(version_of(name[3:]), []).append(score(env))
for version, scores in groups.items():
    if len(scores) > 1:
        m = np.mean(scores, axis=0)
        show(row(f"AVERAGE {version} ({len(scores)} seeds)", m, "-", beats(m, case.AUTHORS)))
show.save(results_path(f"{case.ID}_evaluate.txt"))

# ===== 4. The plot: process output (top) and Kp chosen by the RL agents (bottom) =====
COLOURS = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors[1::2])   # 20 different colours
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for i, (name, (env, gains)) in enumerate(results.items()):
    colour = COLOURS[i % len(COLOURS)]
    ax1.plot(np.arange(env.k + 1) * case.DT, env.y[: env.k + 1], lw=2, color=colour, label=name)
    if name.startswith("RL "):
        # the agent picks new gains at every move: first move at the last warm-up sample
        t_moves = (case.WARMUP_STEPS - 1 + np.arange(len(gains)) * case.ACTION_REPEAT) * case.DT
        ax2.step(t_moves, gains[:, 0], where="post", color=colour, label=name)
setpoints = next(iter(results.values()))[0].r                 # every step test has the same setpoints
ax1.step(np.arange(len(setpoints)) * case.DT, setpoints, "--", color="black", where="post", label="Setpoint")
ax1.set_ylabel("Process output y")
ax1.set_xlim(0.5, 8)
ax1.set_ylim(-0.2, 2.0)
ax2.set_ylabel("Kp chosen\n(same colours as above)")
ax2.set_xlabel("Time (s)")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
ax1.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))   # one legend, outside on the right
fig.tight_layout()
fig.savefig(results_path(f"{case.ID}_comparison.png"), dpi=150)
print(f"(plot saved to results/{case.ID}_comparison.png)")
