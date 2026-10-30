# robustness.py : how well does each controller cope with sensor noise and a wrong process model?
# ---------------------------------------------------------------
# Usage:   .\run robustness.py <case>          e.g.  .\run robustness.py cs1
#
# Evaluation only: nothing is trained or re-tuned. The scenarios come from the case file
# (ROBUSTNESS). Noise scenarios are repeated with 10 noise seeds and averaged; every controller
# gets the same 10 noise sequences. Scores always use the TRUE output.
# Prints one table per scenario, then group averages; saved to results/<case>_robustness.txt.
# ---------------------------------------------------------------
import sys
import numpy as np
from cases import load_case
from core.tools import (Printer, completed, find_agents, load_agent, reference_controllers,
                        results_path, run_test, score, version_of)

NOISE_SEEDS = list(range(10))     # 0, 1, ..., 9
SUMMARY_GROUPS = ["paper", "improved_v2", "improved_v3", "improved_v4n2",
                  "Classical PI", "Best fixed PID", "Best fixed 2-DOF PID"]

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "cs1")
show = Printer()

# ===== 1. Controllers: (name, controller, version, summary group) =====
# A fixed controller is its own group; an agent belongs to the group of its version.
controllers = [(name, ctrl, "paper", name) for name, ctrl in reference_controllers(case, include_2dof=True)]
for name, _ in find_agents(case.ID):
    controllers.append(("RL " + name, load_agent(case, name), version_of(name), version_of(name)))


def run_scenario(controller, version, noise_std, params):
    """All runs of one scenario: (mean metrics, runs completed, number of runs)."""
    seeds = NOISE_SEEDS if noise_std > 0 else [None]      # no noise: one run is enough
    runs = [run_test(case, controller, version, noise_std, params, seed)[0] for seed in seeds]
    metrics = np.mean([score(env) for env in runs], axis=0)
    return metrics, sum(completed(env) for env in runs), len(runs)


# ===== 2. One table per scenario =====
table = {}                                    # (scenario, controller) -> (metrics, done, runs)
show(f"Robustness test on {case.NAME}, step test {case.START_SETPOINT} -> {case.TEST_SETPOINT}")
for scenario, noise_std, params in case.ROBUSTNESS:
    how = f"mean of {len(NOISE_SEEDS)} noise seeds" if noise_std > 0 else "single run"
    show()
    show(f"=== {scenario}  (sensor noise std {noise_std}, process changes {params or 'none'}, {how}) ===")
    show(f"{'Controller':32s} {'Overshoot %':>11s} {'Settling s':>10s} {'ISE':>9s} {'IAE':>9s} {'Completed':>10s}")
    for name, ctrl, version, _ in controllers:
        m, done, runs = run_scenario(ctrl, version, noise_std, params)
        table[(scenario, name)] = (m, done, runs)
        status = "yes" if done == runs else ("no" if runs == 1 else f"no ({done}/{runs})")
        show(f"{name:32s} {m[0]:11.1f} {m[1]:10.2f} {m[2]:9.3f} {m[3]:9.3f} {status:>10s}")

# ===== 3. Summary: group averages, one small table per metric =====
scenarios = [s[0] for s in case.ROBUSTNESS]
show()
show("=== Summary: group averages (RL groups = mean of their seeds) ===")
for col, title, fmt in [(0, "Overshoot %", "{:11.1f}"), (1, "Settling s", "{:11.2f}"),
                        (2, "ISE", "{:11.3f}"), (3, "IAE", "{:11.3f}")]:
    show()
    show(f"{title:22s}" + "".join(f"{s:>11s}" for s in scenarios))
    for group in SUMMARY_GROUPS:
        members = [name for name, _, _, g in controllers if g == group]
        if members:
            cells = [fmt.format(np.mean([table[(s, n)][0][col] for n in members])) for s in scenarios]
            show(f"{group:22s}" + "".join(cells))
show()
show(f"{'Completed runs':22s}" + "".join(f"{s:>11s}" for s in scenarios))
for group in SUMMARY_GROUPS:
    members = [name for name, _, _, g in controllers if g == group]
    if members:
        cells = [f"{sum(table[(s, n)][1] for n in members)}/{sum(table[(s, n)][2] for n in members)}"
                 for s in scenarios]
        show(f"{group:22s}" + "".join(f"{c:>11s}" for c in cells))
show()
show("Settling 20.00 s = still outside the 2 % band at the end of the test.")
show.save(results_path(f"{case.ID}_robustness.txt"))
