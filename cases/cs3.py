# cases/cs3.py : Case Study 3 - the authors' third case study (the "hard" process)
# ---------------------------------------------------------------
# Transfer function in the authors' code (capsule 8627992, cs3.py):
#     G(s) = (0.77248 s - 0.1727) / (3.1 s - 1)      sample time 1 (minutes in their plots)
# Their simulation turns it into state space (python-control tf2ss + c2d) and computes the
# output as y = C x only: the direct term D (= 0.77248 / 3.1) is never added. The process they
# actually simulate - and on which their published results were obtained - is therefore the
# strictly proper part:
#     G_sim(s) = c / (s - a),   a = 1/3.1 = 0.3226 (unstable),   c = 0.02467
# This file simulates exactly that, so our results are comparable with the authors' CS3 agent.
# Checked: replaying the authors' recorded gains through this file reproduces their saved
# test trajectory (see REPORT.md).
# ---------------------------------------------------------------
import numpy as np
from core.tuning_rules import classical_set

NAME = "CS3: G(s) = (0.77248s - 0.1727)/(3.1s - 1)  (authors' CS3)"

# The authors' transfer function: numerator and denominator coefficients
NUM = (0.77248, -0.1727)
DEN = (3.1, -1.0)

# ===== Timing and limits (as the authors' cs3.py) =====
DT = 1.0                  # sample time
U_LIMIT = 100.0           # control input (and PID integral) kept inside -100..+100
Y_LIMIT = 50.0            # the authors' early stopping: |output| > 50 ends a training run
ACTION_REPEAT = 2

# ===== Setpoints (as the authors) =====
WARMUP_STEPS = 10
HOLD_STEPS = 200
START_SETPOINT = 0.0
TEST_SETPOINT = 1.0
TRAIN_RANGE = (0.0, 2.0)
TRAIN_STEPS = 3

# ===== Gain ranges for improved_v5 (set from the fixed-gain check, see REPORT.md) =====
GAIN_RANGES = None

# ===== Grid for grid_search.py (41 values each) =====
# The best fixed PIDs lie near Kp 40, tau_i 3.5, tau_d 0.1 (outside the default grid Kp <= 20, tau_i <= 2).
GRID = dict(Kp=np.linspace(0.0, 100.0, 41), tau_i=np.linspace(0.0, 10.0, 41),
            tau_d=np.linspace(0.0, 1.0, 41), b=np.linspace(0.0, 1.0, 21))

# ===== Reference controllers =====
# Published rules for K e^(-theta s)/(T s - 1) (core/tuning_rules.py): c/(s - a) = K/(T s - 1) with
# T = 3.1 and K = c*T = 0.0765; theta = DT/2 = 0.5 (the usual allowance for the sample-and-hold).
CLASSICAL = classical_set(K=(NUM[1] - NUM[0] / DEN[0] * DEN[1]) / DEN[0] * 3.1, T=3.1, theta=0.5)
AUTHORS = (99.7, 58.0, 11.266, 17.785)   # authors' CS3 agent, scored from their matlab_matrix.mat

# ===== Robustness scenarios: (name, sensor-noise std, changed process parameters) =====
ROBUSTNESS = [
    ("nominal",    0.0,  {}),
    ("noise 0.01", 0.01, {}),
    ("noise 0.05", 0.05, {}),
    ("pole -20%",  0.0,  {"pole": 0.8}),     # less unstable
    ("pole +20%",  0.0,  {"pole": 1.2}),     # more unstable
    ("K = 0.8",    0.0,  {"K": 0.8}),
    ("K = 1.2",    0.0,  {"K": 1.2}),
]


# ===== The process =====
class Process:
    """The strictly proper part of the authors' G(s), c / (s - a), exact for a constant input per sample:
           y[k+1] = A * y[k] + B * u[k],   A = e^(a*DT),   B = c * (e^(a*DT) - 1) / a
       pole and K scale a and c for the robustness test (1.0 = the authors' process)."""

    def __init__(self, pole=1.0, K=1.0):
        d = NUM[0] / DEN[0]                           # the direct term the authors' code leaves out
        a = -DEN[1] / DEN[0] * pole                   # pole of the process (+0.3226)
        c = (NUM[1] - d * DEN[1]) / DEN[0] * K        # gain of the strictly proper part (0.02467)
        self.A = np.exp(a * DT)
        self.B = c * (np.exp(a * DT) - 1.0) / a

    def reset(self):
        self.y = 0.0
        return self.y

    def step(self, u):
        self.y = self.A * self.y + self.B * u
        return self.y

    def bump(self, amount):
        self.y = self.y + amount
        return self.y
