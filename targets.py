# targets.py : did the agents beat the numbers they must beat?  (pass / fail table)
# ---------------------------------------------------------------
# Usage:   .\run targets.py <case> [version]        e.g.  .\run targets.py cs3 improved_v6
#
# The targets (fixed BEFORE training, see THREE_PROCESS_PLAN.md):
#   A. every classical tuning rule of the case file (CLASSICAL): beaten on ALL FOUR metrics
#      (each metric smaller than the best value any classical rule reaches)
#   B. the authors' agent (AUTHORS, if the case has one): beaten on ALL FOUR metrics
#   C. the best fixed PID (grid_search.py, tuned on this very test): beaten on ISE and IAE
# Every seed is checked, and the average over the seeds. Saved to results/<case>_targets.txt.
# ---------------------------------------------------------------
import sys
import numpy as np
from cases import load_case
from core.tools import (Printer, completed, find_agents, fixed, load_agent, load_fixed_gains,
                        results_path, run_test, score, version_of)

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "cs1")
VERSION = sys.argv[2] if len(sys.argv) > 2 else "improved_v6"
show = Printer()


def measured(controller):
    env, _ = run_test(case, controller)
    return score(env) if completed(env) else np.full(4, np.inf)


def line(name, m, verdicts=""):
    return f"{name:30s} {m[0]:9.1f} {m[1]:9.2f} {m[2]:9.3f} {m[3]:9.3f}   {verdicts}"


header = f"{'':30s} {'Overshoot':>9s} {'Settling':>9s} {'ISE':>9s} {'IAE':>9s}"

# ----- the targets -----
show(f"Targets on the {case.NAME} step test {case.START_SETPOINT} -> {case.TEST_SETPOINT}")
show(header)
classical = []
for name, gains in getattr(case, "CLASSICAL", []):
    m = measured(fixed(*gains))
    classical.append(m)
    show(line(name, m))
target_a = np.min(classical, axis=0)
show(line("A. best classical value", target_a, "(must beat all 4)"))
target_b = np.array(case.AUTHORS) if case.AUTHORS else None
if target_b is not None:
    show(line("B. authors' agent", target_b, "(must beat all 4)"))
g = load_fixed_gains(case.ID)["best_fixed_pid"]
target_c = measured(fixed(g["Kp"], g["tau_i"], g["tau_d"]))
show(line("C. best fixed PID", target_c, "(must beat ISE and IAE)"))


# ----- the agents -----
def verdict(m):
    """'A yes  B yes  C yes' for one row of metrics."""
    parts = [f"A {'yes' if np.all(m < target_a) else 'NO '}"]
    if target_b is not None:
        parts.append(f"B {'yes' if np.all(m < target_b) else 'NO '}")
    parts.append(f"C {'yes' if np.all(m[2:] < target_c[2:]) else 'NO '}")
    return "  ".join(parts)


show("")
show(f"{VERSION} agents (best checkpoint on the validation tests):")
show(header + "   targets passed")
rows = []
for name, _ in find_agents(case.ID):
    if version_of(name) == VERSION:
        m = measured(load_agent(case, name))
        rows.append(m)
        show(line(name, m, verdict(m)))
if rows:
    m = np.mean(rows, axis=0)
    show(line(f"AVERAGE ({len(rows)} seeds)", m, verdict(m)))
    show(f"Seeds passing every target: {sum(verdict(r).count('NO') == 0 for r in rows)} of {len(rows)}")
else:
    show(f"(no trained {VERSION} agents in runs/{case.ID}/)")
show.save(results_path(f"{case.ID}_targets.txt"))
