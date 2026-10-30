# cases/cs1.py : Case Study 1 - the open-loop unstable process G(s) = 1/(s - 1)
# ---------------------------------------------------------------
# Everything that belongs to ONE case study is in this file: the process, its sample time and
# limits, the setpoints, the reference controllers and the robustness scenarios.
# cases/cs2.py has exactly the same layout.
# ---------------------------------------------------------------
import numpy as np

NAME = "CS1: G(s) = 1/(s - 1)"

# ===== Timing and limits =====
DT = 0.1                  # sample time in seconds
U_LIMIT = 100.0           # control input (and PID integral) kept inside -100..+100
Y_LIMIT = 100.0           # |output| > 100 = "ran away": the run is stopped
ACTION_REPEAT = 2         # the agent chooses new gains every 2 samples (0.2 s)

# ===== Setpoints =====
WARMUP_STEPS = 10         # samples at START_SETPOINT before the first change (1 s)
HOLD_STEPS = 200          # each setpoint is held for 200 samples (20 s)
START_SETPOINT = 0.0      # setpoint during the warm-up
TEST_SETPOINT = 1.0       # the step test: START_SETPOINT -> TEST_SETPOINT
TRAIN_RANGE = (0.0, 2.0)  # training: random setpoints from this range ...
TRAIN_STEPS = 3           # ... this many per training run (3 x 200 samples)

# ===== Reference controllers =====
CLASSICAL_PI = (3.0, 1.25, 0.0)          # classical PI: Kp, tau_i, tau_d
AUTHORS = (57.8, 1.70, 0.142, 0.266)     # authors' agent: overshoot %, settling s, ISE, IAE

# ===== Robustness scenarios: (name, sensor-noise std, changed process parameters) =====
ROBUSTNESS = [
    ("nominal",    0.0,  {}),
    ("noise 0.01", 0.01, {}),
    ("noise 0.05", 0.05, {}),
    ("a = 0.8",    0.0,  {"a": 0.8}),      # less unstable than the model
    ("a = 1.2",    0.0,  {"a": 1.2}),      # more unstable than the model
    ("K = 0.8",    0.0,  {"K": 0.8}),      # control input 20 % weaker
    ("K = 1.2",    0.0,  {"K": 1.2}),      # control input 20 % stronger
]


# ===== The process =====
class Process:
    """G(s) = K / (s - a), simulated exactly for a constant control input over each sample:
           y[k+1] = A * y[k] + B * u[k],   A = e^(a*DT),   B = K * (e^(a*DT) - 1) / a
       The nominal CS1 process is a = 1, K = 1, i.e. 1/(s - 1): A = 1.105, B = 0.105."""

    def __init__(self, a=1.0, K=1.0):
        self.A = np.exp(a * DT)
        self.B = K * (np.exp(a * DT) - 1.0) / a

    def reset(self):
        """Start at rest. Returns the output at time 0."""
        self.y = 0.0
        return self.y

    def step(self, u):
        """One sample of DT seconds with control input u. Returns the new output."""
        self.y = self.A * self.y + self.B * u
        return self.y

    def bump(self, amount):
        """A sudden disturbance on the output (used by live_demo.py). Returns the new output."""
        self.y = self.y + amount
        return self.y
