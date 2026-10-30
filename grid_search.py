# grid_search.py : the best FIXED gains for the step test (the baselines the agents are compared with)
# ---------------------------------------------------------------
# Usage:   .\run grid_search.py <case>          e.g.  .\run grid_search.py cs1
#
# Tries every combination of
#     Kp    = 0, 0.5, ..., 20        (41 values)
#     tau_i = 0, 0.05, ..., 2        (41 values)
#     tau_d = 0, 0.05, ..., 2        (41 values)
#     b     = 0, 0.05, ..., 1        (21 values, setpoint weight; b = 1 is the normal PID)
# on the step test and keeps the smallest ISE (runs that ran away are skipped):
#     "Best fixed PID"        best with b = 1
#     "Best fixed 2-DOF PID"  best over all b
# The result is saved to results/<case>_fixed_gains.json; evaluate.py and robustness.py read it.
#
# Speed trick: the same PID and process classes are run on numpy ARRAYS of gains, so all
# 68,921 gain sets of one b value are simulated side by side (a few seconds in total).
# ---------------------------------------------------------------
import json, sys
import numpy as np
from cases import load_case
from core.env import RLPIDEnv
from core.pid import PID
from core.tools import Printer, fixed, results_path, run_test, score, completed

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "cs1")
show = Printer()


def test_many(Kp, tau_i, tau_d, b):
    """The step test for many gain sets at once (arrays). Returns the ISE of each (inf = ran away)."""
    env = RLPIDEnv(case, test=True)
    env.reset()
    r = env.r                                     # the step test's setpoints
    Ki, Kd = Kp / (tau_i + 0.01), Kp * tau_d      # same formulas as the world
    pid = PID(case.DT, case.U_LIMIT)
    process = case.Process()
    y = process.reset()
    squared_error = np.zeros_like(Kp)
    ran_away = np.zeros(Kp.shape, dtype=bool)
    for k in range(case.WARMUP_STEPS - 1, len(r) - 1):   # the same samples as the world
        u = pid.control(r[k], y, Kp, Ki, Kd, b)
        y = process.step(u)
        squared_error += (r[k + 1] - y) ** 2
        ran_away |= np.abs(y) > case.Y_LIMIT
    ise = squared_error * case.DT
    ise[ran_away] = np.inf                        # a run that ran away can never be "best"
    return ise


# ----- the grid: every combination of Kp, tau_i, tau_d (np.meshgrid = 3 nested loops) -----
GRID = dict(Kp=np.linspace(0.0, 20.0, 41), tau_i=np.linspace(0.0, 2.0, 41),
            tau_d=np.linspace(0.0, 2.0, 41), b=np.linspace(0.0, 1.0, 21))
kp, ti, td = (g.ravel() for g in np.meshgrid(GRID["Kp"], GRID["tau_i"], GRID["tau_d"], indexing="ij"))

best = {}
for b in GRID["b"]:
    ise = test_many(kp, ti, td, b)
    i = np.argmin(ise)                            # position of the smallest ISE
    gains = dict(Kp=float(kp[i]), tau_i=float(ti[i]), tau_d=float(td[i]), b=float(b), ISE=float(ise[i]))
    if b == 1.0:
        best["best_fixed_pid"] = gains            # the normal PID (b = 1)
    if "best_fixed_2dof" not in best or gains["ISE"] < best["best_fixed_2dof"]["ISE"]:
        best["best_fixed_2dof"] = gains           # best over every b

# ----- show the winners, tested once more in the real world -----
show(f"Grid search on the {case.NAME} step test (smallest ISE): "
     f"{len(kp)} gain sets x {len(GRID['b'])} values of b")
for key, title in [("best_fixed_pid", "Best fixed PID"), ("best_fixed_2dof", "Best fixed 2-DOF PID")]:
    g = best[key]
    env, _ = run_test(case, fixed(g["Kp"], g["tau_i"], g["tau_d"], g["b"]))
    m = score(env)
    show(f"  {title:20s}: Kp = {g['Kp']:.2f}, tau_i = {g['tau_i']:.2f}, tau_d = {g['tau_d']:.2f}, "
         f"b = {g['b']:.2f}  ->  overshoot {m[0]:.1f} %, settling {m[1]:.2f} s, ISE {m[2]:.3f}, "
         f"IAE {m[3]:.3f}, completed {'yes' if completed(env) else 'no'}")

with open(results_path(f"{case.ID}_fixed_gains.json"), "w") as f:
    json.dump(best, f, indent=2)
show.save(results_path(f"{case.ID}_grid_search.txt"))
