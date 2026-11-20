# stress_test.py : how far can the process change before each controller stops working?
# ---------------------------------------------------------------
# Usage:   .\run stress_test.py             (all three case studies, a few minutes)
#
# Compares the final RL agents (improved_v6b, 5 seeds) with the best fixed PID (grid search) and
# the best published tuning rule (smallest nominal ISE of the case's CLASSICAL list). No re-training.
#
# 1. OPERATING ENVELOPE: a 9 x 9 grid of process changes, step test 0 -> 1 in every cell.
#      CS1, CS3: unstable pole x 0.6 .. 1.4  and  process gain K x 0.6 .. 1.4
#      CS2:      dead time 0.2 .. 0.6 s      and  process gain K x 0.6 .. 1.4
#    A cell "works" if the run completes and settles inside the 2 % band before the test ends.
#    RL cell = average ISE of the 5 seeds; it works only if all 5 seeds work.
# 2. MONTE CARLO: 200 random processes (each parameter uniform in +-20 %; CS2 dead time 0.3, 0.4 or
#    0.5 s) with sensor noise 0.01. Draw i is run by RL seed i mod 5, so every controller gets
#    200 runs. Reported: median and 90th-percentile ISE, runs that ran away, runs that never settled.
#
# Saves results/stress_test.txt, results/operating_envelope.png, results/monte_carlo.png.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cases import load_case
from core.tools import (Printer, completed, find_agents, fixed, load_agent, load_fixed_gains,
                        results_path, run_test, score, version_of)

CASES = ["cs1", "cs2", "cs3"]
FACTORS = np.linspace(0.6, 1.4, 9)
DELAYS = np.linspace(0.2, 0.6, 9).round(2)        # CS2 dead time in seconds (whole samples of 0.1 s)
POLE = {"cs1": "a", "cs2": "a", "cs3": "pole"}     # name of the pole parameter in each Process
N_DRAWS = 200
COLOURS = {"RL improved_v6b": "#2a78d6", "Best fixed PID": "#eb6834", "Best published rule": "#1baf7a"}
show = Printer()


def controllers(case):
    """{name: [controller, ...]}: the 5 RL seeds, the best fixed PID, the best published rule."""
    rules = []
    for name, gains in case.CLASSICAL:
        env, _ = run_test(case, fixed(*gains))
        if completed(env):
            rules.append((score(env)[2], name, gains))
    _, rule_name, rule_gains = min(rules)
    g = load_fixed_gains(case.ID)["best_fixed_pid"]
    agents = [load_agent(case, n) for n, _ in find_agents(case.ID) if version_of(n) == "improved_v6b"]
    return {"RL improved_v6b": agents, "Best fixed PID": [fixed(g["Kp"], g["tau_i"], g["tau_d"])],
            "Best published rule": [fixed(*rule_gains)]}, rule_name


def run(case, ctrl, params, noise=0.0, seed=None):
    """(ISE, ran away?, never settled?) of one step test."""
    env, _ = run_test(case, ctrl, noise_std=noise, process_params=params, seed=seed)
    if not completed(env):
        return np.inf, True, True
    m = score(env)
    return m[2], False, m[1] >= case.HOLD_STEPS * case.DT - 1e-9


def grid_params(cid, i, j):
    """Process changes of grid cell (row i = gain, column j = pole or dead time)."""
    p = {"K": FACTORS[i]}
    if cid == "cs2":
        p["delay"] = DELAYS[j]
    else:
        p[POLE[cid]] = FACTORS[j]
    return p


def random_params(cid, rng):
    p = {"K": rng.uniform(0.8, 1.2), POLE[cid]: rng.uniform(0.8, 1.2)}
    if cid == "cs2":
        p["delay"] = rng.choice([0.3, 0.4, 0.5])
    return p


envelope, monte = {}, {}
for cid in CASES:
    case = load_case(cid)
    ctrls, rule_name = controllers(case)
    show(f"=== {case.NAME}   (best published rule: {rule_name})")

    # ----- 1. operating envelope -----
    for name, group in ctrls.items():
        ise = np.zeros((9, 9))
        each = np.zeros((len(group), 9, 9), dtype=bool)          # works, per controller (RL: per seed)
        for i in range(9):
            for j in range(9):
                results = [run(case, c, grid_params(cid, i, j)) for c in group]
                ise[i, j] = np.mean([r[0] for r in results])
                each[:, i, j] = [not (r[1] or r[2]) for r in results]
        works = each.all(axis=0)
        per_seed = each.sum(axis=(1, 2)).mean()                  # average over the seeds (fixed: = works)
        envelope[(cid, name)] = (ise, works, per_seed)
        show(f"  envelope  {name:22s} works in {works.sum():2d} of 81 cells (every seed); "
             f"average per seed {per_seed:.1f} of 81")

    rl_ise, rl_ok = envelope[(cid, "RL improved_v6b")][:2]
    for other in ("Best fixed PID", "Best published rule"):
        o_ise, o_ok = envelope[(cid, other)][:2]
        both = rl_ok & o_ok
        show(f"  envelope  cells where both work: {both.sum()}; RL has the lower ISE in "
             f"{np.sum(rl_ise[both] < o_ise[both])} of them (vs {other})")

    # ----- 2. Monte Carlo -----
    rng = np.random.default_rng(0)
    draws = [random_params(cid, rng) for _ in range(N_DRAWS)]
    show(f"  Monte Carlo, {N_DRAWS} random processes, noise 0.01:   median ISE   90th pct ISE   ran away   never settled")
    for name, group in ctrls.items():
        res = [run(case, group[k % len(group)], p, noise=0.01, seed=k) for k, p in enumerate(draws)]
        ise = np.array([r[0] for r in res])
        monte[(cid, name)] = ise
        show(f"  {name:52s} {np.median(ise):10.3f} {np.percentile(ise, 90):14.3f} "
             f"{sum(r[1] for r in res):10d} {sum(r[2] for r in res):15d}")
show.save(results_path("stress_test.txt"))

# ----- figure 1: operating envelope (rows = processes, columns = controllers) -----
names = list(COLOURS)
fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
for r, cid in enumerate(CASES):
    finite = np.concatenate([envelope[(cid, n)][0][envelope[(cid, n)][1]] for n in names])   # working cells
    norm = LogNorm(vmin=finite.min(), vmax=finite.max())
    for c, name in enumerate(names):
        ax = axes[r, c]
        ise, works, per_seed = envelope[(cid, name)]
        shown = np.ma.masked_where(~works, ise)
        im = ax.imshow(shown, origin="lower", cmap="Blues_r", norm=norm, aspect="auto")
        ax.set_facecolor("#e8e8e6")                              # cells that fail: light gray + x
        for i, j in zip(*np.where(~works)):
            ax.text(j, i, "x", ha="center", va="center", color="#52514e", fontsize=9)
        ax.plot(4, 4, marker="o", ms=9, mfc="none", mec="#0b0b0b", mew=1.5)   # the nominal process
        xs = DELAYS if cid == "cs2" else FACTORS
        ax.set_xticks(range(0, 9, 2), [f"{v:g}" for v in xs[::2]])
        ax.set_yticks(range(0, 9, 2), [f"{v:g}" for v in FACTORS[::2]])
        detail = f" (all 5 seeds; per seed {per_seed:.1f})" if name.startswith("RL") else ""
        ax.set_title(f"{cid.upper()} - {name}\nworks in {works.sum()} of 81{detail}", fontsize=10)
        ax.set_xlabel("dead time (s)" if cid == "cs2" else "unstable pole x", fontsize=9)
        if c == 0:
            ax.set_ylabel("process gain K x", fontsize=9)
    fig.colorbar(im, ax=axes[r, :], label=f"{cid.upper()} ISE (darker = better)", shrink=0.9)
fig.suptitle("Operating envelope: step-test ISE when the process changes "
             "(x = ran away or never settled, circle = nominal process)", fontsize=12)
fig.savefig(results_path("operating_envelope.png"), dpi=150)

# ----- figure 2: Monte Carlo ISE distributions -----
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
for ax, cid in zip(axes, CASES):
    data = [monte[(cid, n)] for n in names]
    finite = [d[np.isfinite(d)] for d in data]
    box = ax.boxplot(finite, patch_artist=True, widths=0.55, showfliers=True,
                     medianprops=dict(color="#0b0b0b", lw=1.5),
                     flierprops=dict(marker="o", ms=4, mfc="none", mec="#52514e"))
    for patch, n in zip(box["boxes"], names):
        patch.set(facecolor=COLOURS[n], alpha=0.85, edgecolor="#0b0b0b")
    for k, d in enumerate(data):
        if not np.all(np.isfinite(d)):
            ax.text(k + 1, ax.get_ylim()[1], f"ran away: {np.sum(~np.isfinite(d))}", ha="center",
                    va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks([1, 2, 3], ["RL\nimproved_v6b", "Best fixed\nPID", "Best published\nrule"])
    ax.set_yscale("log")
    ax.set_title(f"{cid.upper()}: {N_DRAWS} random processes (+-20 %), noise 0.01", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Step-test ISE (log scale, lower = better)")
fig.savefig(results_path("monte_carlo.png"), dpi=150)
print("(figures saved to results/operating_envelope.png and results/monte_carlo.png)")
