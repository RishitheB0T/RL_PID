# cases/cs2.py : Case Study 2 - unstable first-order process WITH DEAD TIME, G(s) = e^(-0.4s) / (s - 1)
# ---------------------------------------------------------------
# The same unstable pole as CS1 (s = +1), plus a 0.4 s delay between the control input and the
# process: a widely used benchmark for unstable processes with dead time. The delay makes it much
# harder - a plain P-controller is only stable for roughly 1 < Kp < 3.3 (CS1: up to about 20).
# Same layout as cases/cs1.py; everything else (rewards, PID, training) is unchanged.
# ---------------------------------------------------------------
import numpy as np

NAME = "CS2: G(s) = e^(-0.4s)/(s - 1)"

# ===== Timing and limits =====
DT = 0.1                  # sample time in seconds (the 0.4 s delay is exactly 4 samples)
U_LIMIT = 100.0           # control input (and PID integral) kept inside -100..+100
Y_LIMIT = 100.0           # |output| > 100 = "ran away": the run is stopped
ACTION_REPEAT = 2         # the agent chooses new gains every 2 samples (0.2 s)

# ===== Setpoints (as CS1) =====
WARMUP_STEPS = 10
HOLD_STEPS = 200
START_SETPOINT = 0.0
TEST_SETPOINT = 1.0
TRAIN_RANGE = (0.0, 2.0)
TRAIN_STEPS = 3

# ===== Gain ranges for improved_v5: (lowest Kp, tau_i, tau_d), (highest Kp, tau_i, tau_d) =====
# From a fixed-gain check on the step test: stable PIDs need Kp 1.2-3.2; the best ones
# (ISE within 1.5x of the best, 1.929) lie at Kp 1.8-2.9, tau_i 0.75-8, tau_d 0.10-0.46.
# The CS1 ranges (tau_i <= 0.5) contain NO stable fixed PID for this process.
GAIN_RANGES = (np.array([0.0, 0.5, 0.0]), np.array([5.0, 4.0, 0.5]))

# ===== Reference controllers =====
CLASSICAL_PI = None       # no classical tuning entered: the grid-search PID (grid_search.py) is the baseline
AUTHORS = None            # the paper gives no numbers for this process

# ===== Robustness scenarios: (name, sensor-noise std, changed process parameters) =====
ROBUSTNESS = [
    ("nominal",     0.0,  {}),
    ("noise 0.01",  0.01, {}),
    ("noise 0.05",  0.05, {}),
    ("delay 0.3",   0.0,  {"delay": 0.3}),    # dead time 25 % shorter than the model
    ("delay 0.5",   0.0,  {"delay": 0.5}),    # dead time 25 % longer than the model
    ("K = 0.8",     0.0,  {"K": 0.8}),
    ("K = 1.2",     0.0,  {"K": 1.2}),
]


# ===== The process =====
class Process:
    """G(s) = K e^(-delay*s) / (s - a), simulated exactly for a constant control input per sample:
           y[k+1] = A * y[k] + B * u[k - d],   A = e^(a*DT),   B = K * (e^(a*DT) - 1) / a,
       where d = delay / DT samples. Nominal CS2: a = 1, K = 1, delay = 0.4 s (d = 4)."""

    def __init__(self, a=1.0, K=1.0, delay=0.4):
        self.A = np.exp(a * DT)
        self.B = K * (np.exp(a * DT) - 1.0) / a
        self.d = int(round(delay / DT))          # delay in whole samples

    def reset(self):
        """Start at rest: output 0 and no control input on its way yet."""
        self.y = 0.0
        self.pipe = [0.0] * self.d               # control inputs still "travelling" through the delay
        return self.y

    def step(self, u):
        """One sample of DT seconds: u enters the delay line, the input from d samples ago acts."""
        self.pipe.append(u)
        u_delayed = self.pipe.pop(0)             # first in, first out (Java: a queue)
        self.y = self.A * self.y + self.B * u_delayed
        return self.y

    def bump(self, amount):
        """A sudden disturbance on the output (used by live_demo.py)."""
        self.y = self.y + amount
        return self.y
