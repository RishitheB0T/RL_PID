# cases/cs2.py : Case Study 2 - TEMPLATE, to be filled in
# ---------------------------------------------------------------
# Same layout as cases/cs1.py. Fill in every line marked TODO, then all scripts work for CS2:
#     .\run train.py cs2 improved_v4n2 0
#     .\run grid_search.py cs2
#     .\run evaluate.py cs2
# Until the required blanks are filled in, the scripts stop and list what is still missing.
# Lines without TODO are general settings, copied from CS1; change them only if CS2 needs it.
# ---------------------------------------------------------------
import numpy as np

NAME = None               # TODO: e.g. "CS2: G(s) = ..."

# ===== Timing and limits =====
DT = None                 # TODO: sample time in seconds                     (CS1: 0.1)
U_LIMIT = None            # TODO: control input kept inside -U_LIMIT..+U_LIMIT (CS1: 100.0)
Y_LIMIT = None            # TODO: |output| above this = "ran away"           (CS1: 100.0)
ACTION_REPEAT = 2         # the agent chooses new gains every 2 samples (as CS1)

# ===== Setpoints =====
WARMUP_STEPS = 10         # samples at START_SETPOINT before the first change (as CS1)
HOLD_STEPS = 200          # each setpoint is held for 200 samples (as CS1)
START_SETPOINT = None     # TODO: setpoint during the warm-up                (CS1: 0.0)
TEST_SETPOINT = None      # TODO: setpoint after the test step               (CS1: 1.0)
TRAIN_RANGE = None        # TODO: (lowest, highest) random training setpoint (CS1: (0.0, 2.0))
TRAIN_STEPS = 3           # random setpoints per training run (as CS1)

# ===== Reference controllers (optional: leave None to skip) =====
CLASSICAL_PI = None       # TODO (optional): (Kp, tau_i, tau_d) of a classical controller
AUTHORS = None            # TODO (optional): authors' (overshoot %, settling s, ISE, IAE) for CS2

# ===== Robustness scenarios: (name, sensor-noise std, changed process parameters) =====
ROBUSTNESS = [
    ("nominal",    0.0,  {}),
    ("noise 0.01", 0.01, {}),
    ("noise 0.05", 0.05, {}),
    # TODO (optional): model-mismatch scenarios, using the parameter names of Process.__init__,
    #                  e.g. ("K = 1.2", 0.0, {"K": 1.2}),
]


# ===== The process =====
class Process:
    """TODO: describe your CS2 process here (transfer function or equations)."""

    def __init__(self):
        # TODO: put the model parameters in the brackets WITH their nominal values,
        #       e.g.  def __init__(self, K=1.0, tau=2.0):
        #       and store or pre-compute what step() needs, e.g.  self.K = K
        pass

    def reset(self):
        """Start at rest. Return the output y at time 0."""
        # TODO: set the starting state (e.g. self.y = 0.0) and return the output.
        raise NotImplementedError("Fill in Process.reset() in cases/cs2.py")

    def step(self, u):
        """Simulate ONE sample of DT seconds with control input u. Return the new output y."""
        # TODO: update the state with u and return the new output.
        #       Write it with numpy (np.exp, +, *, ...): grid_search.py then also works.
        raise NotImplementedError("Fill in Process.step() in cases/cs2.py")

    def bump(self, amount):
        """(Only for live_demo.py) A sudden disturbance on the output. Return the new output."""
        # TODO (optional): e.g. self.y = self.y + amount; return self.y
        raise NotImplementedError("Fill in Process.bump() in cases/cs2.py")


# ---------------------------------------------------------------
# HOW TO WRITE step() - two examples (not used, for reference only)
#
# (a) First-order process G(s) = K / (T*s - 1), exact for a constant u over one sample:
#         def __init__(self, K=1.0, T=1.0):
#             self.A = np.exp(DT / T)
#             self.B = K * (np.exp(DT / T) - 1.0)
#         def step(self, u):
#             self.y = self.A * self.y + self.B * u
#             return self.y
#
# (b) Nonlinear process written as dx/dt = f(x, u): split the sample into small Euler steps:
#         def step(self, u):
#             for _ in range(10):
#                 self.x = self.x + (DT / 10) * f(self.x, u)
#             return output_of(self.x)
#
# A time delay of d samples can be added by keeping the last d control inputs in a list
# and feeding the oldest one to the process.
# ---------------------------------------------------------------
