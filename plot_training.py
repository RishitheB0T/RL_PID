# plot_training.py : validation score during training, round 1 (improved_v6) vs round 2 (improved_v6b)
# ---------------------------------------------------------------
# Usage:   .\run plot_training.py
#
# Reads runs/<case>/improved_v6*_seed*/validation.csv (written by train.py --select every 50,000
# steps). Score = average ratio to the anchor PID on the validation tests: 1 = as good as the best
# fixed PID the agent starts from, below 1 = better. Thin lines = seeds, thick line = their mean.
# Shows why v6b works: with the default exploration noise (v6) the agents drift ABOVE 1.
# Saved to results/training_curves.png.
# ---------------------------------------------------------------
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from core.tools import results_path, runs_folder

STYLE = {"improved_v6": ("#8a8986", "round 1: improved_v6 (exploration std 1)"),
         "improved_v6b": ("#2a78d6", "round 2: improved_v6b (exploration std 0.2)")}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, constrained_layout=True)
for ax, cid in zip(axes, ["cs1", "cs2", "cs3"]):
    for version, (colour, label) in STYLE.items():
        curves = []
        for path in sorted(glob.glob(os.path.join(runs_folder(cid), f"{version}_seed*", "validation.csv"))):
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            curves.append(data[:, 1])
            ax.plot(data[:, 0] / 1e6, data[:, 1], color=colour, lw=0.8, alpha=0.5)
        if curves:
            n = min(len(c) for c in curves)
            steps = np.arange(1, n + 1) * 0.05
            ax.plot(steps, np.mean([c[:n] for c in curves], axis=0), color=colour, lw=2.5,
                    label=f"{label}, {len(curves)} seeds")
    ax.axhline(1.0, color="#eb6834", lw=2, ls="--", label="best fixed PID (start point) = 1")
    ax.set_title(cid.upper(), fontsize=11)
    ax.set_xlabel("training steps (millions)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 2.0)
axes[0].set_ylabel("validation score (lower = better)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Training: small exploration keeps the agent improving on the PID it starts from", fontsize=12)
fig.savefig(results_path("training_curves.png"), dpi=150)
print("(saved to results/training_curves.png)")
