# core/tuning_rules.py : published (classical) PID tuning rules - the baselines that must be beaten
# ---------------------------------------------------------------
# Source: I. Kaya, "Simple and Optimal PI/PID Tuning Formulae for Unstable Time Delay Processes",
#         ELECO 2017 (10th Int. Conf. on Electrical and Electronics Engineering), Turkey.
#         Tables 1 and 2, equations (6), (7), (9), (10), (11).
# The rules are for the unstable first-order-plus-dead-time model (UFOPDT)
#         G(s) = K e^(-theta s) / (T s - 1)
# and the ideal controller  Kc (1 + 1/(Ti s) + Td s), with constants fitted for minimum ISTE and
# minimum IST2E (time-weighted squared error) of a set-point step, valid for 0.1 <= theta/T <= 0.9.
# Checked: tests/test_tuning_rules.py reproduces the paper's own worked examples (Tables 3 and 4).
#
# How a rule becomes our gains: our PID uses Ki = Kp/(tau_i + 0.01) and Kd = Kp*tau_d (core/pid.py),
# so tau_i = Ti - 0.01 gives exactly the rule's Ki = Kc/Ti.
# ---------------------------------------------------------------
import numpy as np

# PI controller, eq. (6) K*Kc = a1 e^(b1 tn) + c1 e^(d1 tn),  eq. (7) Ti/T = a2 e^(b2 tn) + c2 e^(d2 tn)
PI_CONSTANTS = {     # (range of tn = theta/T): {criterion: (a1, b1, c1, d1, a2, b2, c2, d2)}
    "low":  {"ISTE":  (19.81, -16.36, 5.321, -2.298, 0.592, 5.095, -0.672, -8.667),
             "IST2E": (18.26, -16.24, 4.711, -2.233, 0.465, 5.615, -0.573, -7.039)},
    "high": {"ISTE":  (5.298, -4.148, 1.279, -0.338, 0.237, 6.754, 1.48e-8, 25.92),
             "IST2E": (5.046, -4.207, 1.048, -0.125, 0.178, 7.348, 1.08e-8, 26.48)},
}
# PID controller, eq. (9) K*Kc = a + b tn^c,  eq. (10) Ti/T = w e^(x tn) + y e^(z tn),  eq. (11) Td/T = k tn + m
PID_CONSTANTS = {    # {criterion: (a, b, c, w, x, y, z, k, m)}
    "low":  {"ISTE":  (0.208, 1.166, -0.980, 0.576, 2.644, -0.546, -2.989, 0.494, -0.00402),
             "IST2E": (0.162, 1.160, -1.001, 0.608, 2.782, -0.608, -1.537, 0.436, -0.00348)},
    "high": {"ISTE":  (0.343, 1.048, -1.047, 0.504, 2.806, -8.78e7, -45.030, 0.514, -0.0129),
             "IST2E": (0.395, 0.969, -1.110, 0.455, 3.140, -9.50e4, -30.320, 0.470, -0.0199)},
}


def kaya_2017(K, T, theta, controller="PID", criterion="ISTE"):
    """Kaya (2017) settings for K e^(-theta s)/(T s - 1). Returns the paper's (Kc, Ti, Td)."""
    tn = theta / T                                   # normalised dead time
    if not 0.1 <= tn <= 0.9:
        raise ValueError(f"theta/T = {tn:.3f} is outside the rule's range 0.1 .. 0.9")
    part = "low" if tn <= 0.450 else "high"          # the paper's two ranges: 0.1-0.450 and 0.451-0.9
    if controller == "PI":
        a1, b1, c1, d1, a2, b2, c2, d2 = PI_CONSTANTS[part][criterion]
        Kc = (a1 * np.exp(b1 * tn) + c1 * np.exp(d1 * tn)) / K
        Ti = (a2 * np.exp(b2 * tn) + c2 * np.exp(d2 * tn)) * T
        return Kc, Ti, 0.0
    a, b, c, w, x, y, z, k, m = PID_CONSTANTS[part][criterion]
    Kc = (a + b * tn ** c) / K
    Ti = (w * np.exp(x * tn) + y * np.exp(z * tn)) * T
    Td = (k * tn + m) * T
    return Kc, Ti, Td


def classical_set(K, T, theta):
    """All four Kaya (2017) rules as our gains: [(name, (Kp, tau_i, tau_d)), ...]."""
    rules = []
    for controller in ("PI", "PID"):
        for criterion in ("ISTE", "IST2E"):
            Kc, Ti, Td = kaya_2017(K, T, theta, controller, criterion)
            rules.append((f"Kaya 2017 {controller} ({criterion})", (Kc, Ti - 0.01, Td)))
    return rules
