# plot_all_agents.py : every trained agent on the step test, all in ONE graph
# ---------------------------------------------------------------
# Usage:   .\run plot_all_agents.py <case>          e.g.  .\run plot_all_agents.py cs1
# Colour = version, line style = seed. Saved to results/<case>_all_agents.png and shown.
# ---------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from cases import load_case
from core.tools import completed, find_agents, load_agent, results_path, run_test, version_of
from core.versions import VERSIONS

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "cs1")

COLOURS = dict(zip(VERSIONS, plt.cm.tab10.colors))     # one colour per version
STYLES = {"0": "-", "1": "--", "2": ":"}               # one line style per seed

fig, ax = plt.subplots(figsize=(11, 6))
env = None
for name, _ in find_agents(case.ID):
    version, seed = name.split("_seed")
    env, _ = run_test(case, load_agent(case, name), version)
    label = f"{version} seed {seed}" + ("" if completed(env) else "  (ran away)")
    ax.plot(np.arange(env.k + 1) * case.DT, env.y[: env.k + 1], color=COLOURS[version],
            ls=STYLES.get(seed, "-"), lw=1.6, label=label)
    print(f"{name:26s} peak output {env.y[: env.k + 1].max():9.3f}   completed {completed(env)}")
if env is None:
    raise SystemExit(f"No trained agents in runs/{case.ID}/ yet - train one with train.py first.")

ax.step(np.arange(len(env.r)) * case.DT, env.r, "--", color="black", where="post", lw=1.2, label="Setpoint")
ax.set_xlim(0.5, 8)
ax.set_ylim(-0.2, 2.0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Process output y")
ax.set_title(f"Every trained agent on the step test ({case.NAME})")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout()
fig.savefig(results_path(f"{case.ID}_all_agents.png"), dpi=150)
print(f"(plot saved to results/{case.ID}_all_agents.png)")
plt.show()
