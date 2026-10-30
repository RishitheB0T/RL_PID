# core/pid.py : the PID controller used in every case study
# ---------------------------------------------------------------
# Every sample the controller computes the full control input u ("positional" PID):
#
#   P = Kp * (b * r - y)               r = setpoint, y = measured output
#                                      b = setpoint weight (1 = normal PID; only improved_v3 uses b < 1)
#   I = running sum of Ki * e * dt     e = r - y; kept inside -limit..+limit (anti-windup)
#   D = -Kd * (y - y_prev) / dt        on the MEASUREMENT, so a setpoint jump gives no derivative kick
#
#   u = P + I + D, then kept inside -limit..+limit (the actuator limit)
#
# It works with plain numbers AND with numpy arrays of gains: grid_search.py uses that to run
# tens of thousands of fixed-gain PIDs side by side.
# ---------------------------------------------------------------
import numpy as np


class PID:
    def __init__(self, dt, limit):
        # Constructor (Java: public PID(double dt, double limit)).
        self.dt = dt             # sample time in seconds
        self.limit = limit       # u and the integral are always kept inside -limit..+limit
        self.reset()

    def reset(self):
        """Forget the past: empty integral, no previous measurement."""
        self.integral = 0.0
        self.y_prev = None       # None = Java's null: there is no previous measurement yet

    def control(self, r, y, Kp, Ki, Kd, b=1.0):
        """One sample: setpoint r, measured output y and the gains -> control input u."""
        e = r - y                                              # error
        dy = 0.0 if self.y_prev is None else y - self.y_prev   # change of the measurement
        self.integral += Ki * e * self.dt                      # I part grows with the error
        self.integral = np.clip(self.integral, -self.limit, self.limit)   # anti-windup cap
        u = Kp * (b * r - y) + self.integral - Kd * dy / self.dt         # P + I + D
        self.y_prev = y                                        # remember for the next sample
        return np.clip(u, -self.limit, self.limit)             # actuator limit
