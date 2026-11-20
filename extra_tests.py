# extra_tests.py : two more tests the agents were never trained or selected on
# ---------------------------------------------------------------
# Usage:   .\run extra_tests.py <case>          e.g.  .\run extra_tests.py cs3
#
# 1. LOAD DISTURBANCE. Step test 0 -> 1; once it has settled (100 samples after the step) a
#    constant extra input d enters the process (u + d), as when a load changes. d = 20 % of the
#    steady-state control input of the anchor PID (CS1/CS2: 0.2, CS3: about 2.6).
#    Measured after d starts: peak deviation |r - y|, recovery time (last sample outside the
#    2 % band, counted from the disturbance) and IAE.
# 2. NEW SETPOINT CHANGES: 0 -> 0.25 (small), 0 -> 1.9 (large), 1.5 -> 0.5 (a step DOWN).
#    The last step is scored with the four usual metrics.
# Agents are averaged per version (all seeds); a run that runs away counts as "ran away".
# Saved to results/<case>_extra_tests.txt.
# ---------------------------------------------------------------
import sys
import numpy as np
from cases import load_case
from core.env import RLPIDEnv
from core.tools import (Printer, completed, find_agents, fixed, load_agent, load_fixed_gains,
                        results_path, run_test, version_of)
from core.validation import last_step_metrics

case = load_case(sys.argv[1] if len(sys.argv) > 1 else "cs1")
show = Printer()
GROUPS = ["improved_v4n2", "improved_v5", "improved_v6b"]      # agent versions shown (if they exist)
NEW_STEPS = [("0 -> 0.25", [0.25]), ("0 -> 1.9", [1.9]), ("1.5 -> 0.5 (down)", [1.5, 0.5])]
DISTURBANCE_AT = 100                                          # samples after the setpoint step


class LoadDisturbance:
    """Wraps the case's process: from sample `start` on, the input is u + amount."""

    def __init__(self, process, start, amount):
        self.process, self.start, self.amount, self.n = process, start, amount, 0

    def reset(self):
        self.n = 0
        return self.process.reset()

    def step(self, u):
        self.n += 1
        return self.process.step(u + (self.amount if self.n > self.start else 0.0))


def disturbance_run(controller, amount):
    """Step test with a load disturbance. Returns [peak deviation, recovery time, IAE after d] or None."""
    env = RLPIDEnv(case, test=True)
    env.process = LoadDisturbance(env.process, DISTURBANCE_AT, amount)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step_gains(*controller(obs))
        done = terminated or truncated
    if not completed(env):
        return None
    first = case.WARMUP_STEPS + DISTURBANCE_AT                 # first output that feels d
    e = env.r[first:] - env.y[first:]
    outside = np.where(np.abs(e) > 0.02 * abs(case.TEST_SETPOINT - case.START_SETPOINT))[0]
    recovery = (outside[-1] + 1) * case.DT if len(outside) else 0.0
    return np.array([np.max(np.abs(e)), recovery, np.sum(np.abs(e)) * case.DT])


def step_run(controller, levels):
    env, _ = run_test(case, controller, levels=levels)
    return last_step_metrics(env) if completed(env) else None


# ----- the controllers: (group name, [controllers]) -----
groups = []
for version in GROUPS:
    agents = [load_agent(case, name) for name, _ in find_agents(case.ID) if version_of(name) == version]
    if agents:
        groups.append((f"{version} ({len(agents)} seeds)", agents))
for name, gains in getattr(case, "CLASSICAL", []):
    groups.append((name, [fixed(*gains)]))
g = load_fixed_gains(case.ID)["best_fixed_pid"]
groups.append(("Best fixed PID", [fixed(g["Kp"], g["tau_i"], g["tau_d"])]))

# disturbance size: 20 % of the anchor PID's steady-state control input at the test setpoint
env, _ = run_test(case, fixed(*case.ANCHOR_PID))
amount = 0.2 * abs(env.u[env.k - 1])


def table(title, columns, run):
    """One table: a row per group, averaged over its controllers ('ran away: n' if any failed)."""
    show("")
    show(title)
    show(f"{'Controller':28s}" + "".join(f"{c:>13s}" for c in columns))
    for name, controllers in groups:
        results = [run(c) for c in controllers]
        good = [r for r in results if r is not None]
        failed = len(results) - len(good)
        cells = "".join(f"{v:13.3f}" for v in np.mean(good, axis=0)) if good else "".join(f"{'-':>13s}" for _ in columns)
        show(f"{name:28s}{cells}" + (f"   (ran away: {failed} of {len(results)})" if failed else ""))


show(f"Extra tests on {case.NAME}")
table(f"1. Load disturbance d = {amount:.3f} on the input, {DISTURBANCE_AT} samples after the step 0 -> 1",
      ["peak dev.", "recovery", "IAE after d"], lambda c: disturbance_run(c, amount))
for title, levels in NEW_STEPS:
    table(f"2. Setpoint change {title}", ["overshoot %", "settling", "ISE", "IAE"],
          lambda c, levels=levels: step_run(c, levels))
show.save(results_path(f"{case.ID}_extra_tests.txt"))
