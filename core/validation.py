# core/validation.py : choosing the best checkpoint WITHOUT looking at the final test
# ---------------------------------------------------------------
# During training (train.py --select) the agent is tried every 50,000 steps on three VALIDATION
# step tests that are not the final 0 -> 1 test:
#     0 -> 0.5,    0 -> 1.5,    0 -> 0.8 -> 1.6   (only the last step is scored)
# Each is scored with the same four metrics as the final test, divided by what the case's
# anchor PID (its best fixed PID) gets on that same test:  1.0 = as good as the anchor PID,
# below 1.0 = better. The selection score is the average of these 12 ratios (3 tests x 4 metrics);
# the checkpoint with the lowest score is kept. A test that runs away scores infinity.
# Like choosing a model on a validation set in machine learning: the final test stays unseen.
# ---------------------------------------------------------------
import numpy as np
from core.tools import completed, fixed, run_test

VALIDATION_LEVELS = [[0.5], [1.5], [0.8, 1.6]]


def last_step_metrics(env):
    """[overshoot %, settling time, ISE, IAE] of the LAST setpoint step of a finished test.
    For the 0 -> 1 step test this is exactly core.tools.score (tested in tests/test_v6.py)."""
    c = env.case
    start = len(env.r) - env.hold_steps              # first sample of the last setpoint
    before, after = env.r[start - 1], env.r[start]
    size = after - before
    y = env.y[start: env.k + 1]
    e = env.r[start: env.k + 1] - y
    outside = np.where(np.abs(e) > 0.02 * abs(size))[0]
    peak = y.max() if size > 0 else y.min()
    overshoot = max(0.0, (peak - after) / size * 100)
    settling = (outside[-1] + 1) * c.DT if len(outside) else 0.0
    return np.array([overshoot, settling, np.sum(e ** 2) * c.DT, np.sum(np.abs(e)) * c.DT])


def validation_metrics(case, controller):
    """The metrics of every validation test ([3 x 4] array); a run-away test gives infinities."""
    rows = []
    for levels in VALIDATION_LEVELS:
        env, _ = run_test(case, controller, levels=levels)
        rows.append(last_step_metrics(env) if completed(env) else np.full(4, np.inf))
    return np.array(rows)


def anchor_metrics(case):
    """What the anchor PID (the case's best fixed PID) gets on the validation tests."""
    return validation_metrics(case, fixed(*case.ANCHOR_PID))


def selection_score(metrics, anchor, dt):
    """Average ratio to the anchor PID over all tests and metrics (lower = better).
    1 % is added to both overshoots and one sample (dt) to both settling times, so 0 never divides."""
    offset = np.array([1.0, dt, 0.0, 0.0])
    return float(np.mean((np.array(metrics) + offset) / (np.array(anchor) + offset)))
