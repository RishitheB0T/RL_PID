# core/versions.py : the reward VERSIONS - what the agent is trained to want
# ---------------------------------------------------------------
# A "version" = reward function + crash penalty + gain ranges (+ optional setpoint weight b).
# All eleven versions of this project are in ONE table at the bottom of this file, in the order
# they were tried. "improved_v4n2" is the final one for CS1; "improved_v5" is its general form;
# "improved_v6" starts from the best fixed PID of each case study (the three-process study).
#
# The agent never outputs gains directly. It outputs "sliders" between -1 and +1, which are
# stretched into the gain ranges of its version:  -1 -> lowest value,  +1 -> highest value.
# ---------------------------------------------------------------
import numpy as np
from dataclasses import dataclass

# ===== 1. Reward functions (higher reward = better) =====
#   e  = setpoint - output (the error)
#   ov = how far the output is PAST the setpoint, in the direction of the last setpoint change
#        (0 if it is not past it)
ERROR_SCALE = 10.0        # our error weight (the authors used 0.01)
OVERSHOOT_SCALE = 100.0   # extra weight on overshoot (improved_v4 onwards)
PENALTY_CAP = 5000.0      # the penalty of one sample never exceeds this (the authors used 5)
BAND_BONUS = 1.0          # improved_v6: +1 per sample inside the 2 % band of the last setpoint step


def bonus(e):
    """A bonus of about +2 while the output is within 0.01 of the setpoint (the authors' idea)."""
    return (2.0 - abs(e)) if abs(e) <= 0.01 else 0.0


def reward_paper(e, ov):
    """The authors' reward: tiny penalty 0.01*e^2 (capped at 5) + bonus. ov is not used."""
    return -min(0.01 * e ** 2, 5.0) + bonus(e)


def reward_squared(e, ov):
    """Our reward: penalty 10*e^2 (capped at 5000) + bonus. ov is not used."""
    return -min(ERROR_SCALE * e ** 2, PENALTY_CAP) + bonus(e)


def reward_overshoot(e, ov):
    """Our reward with an overshoot penalty: 10*e^2 + 100*ov^2 (capped at 5000) + bonus."""
    return -min(ERROR_SCALE * e ** 2 + OVERSHOOT_SCALE * ov ** 2, PENALTY_CAP) + bonus(e)


# ===== 2. Gain ranges: (lowest Kp, tau_i, tau_d), (highest Kp, tau_i, tau_d) =====
WIDE = (np.array([0.0, 0.0, 0.0]), np.array([20.0, 2.0, 2.0]))       # the authors' ranges
NARROW = (np.array([0.0, 0.05, 0.0]), np.array([20.0, 1.0, 0.2]))    # tau_i <= 1, tau_d <= 0.2
NARROW2 = (np.array([0.0, 0.05, 0.0]), np.array([20.0, 0.5, 0.2]))   # tau_i <= 0.5

# "anchor" ranges (improved_v6): centred on the case's best fixed PID (ANCHOR_PID in the case file),
# so slider 0 gives exactly that PID and the untrained agent already starts as a good controller:
#     Kp    = Kp0 * 2^a       (a = -1 .. +1  ->  Kp0/2 .. 2*Kp0)
#     tau_i = tau_i0 * 3^a    (tau_i0/3 .. 3*tau_i0)
#     tau_d = tau_d0 + a * span, never below 0   (span = ANCHOR_TD_SPAN in the case file)
ANCHOR_KP_FACTOR = 2.0
ANCHOR_TI_FACTOR = 3.0


# ===== 3. What one version is made of =====
# @dataclass writes the constructor for us (like a Java "record"): Version(reward, crash, ranges, ...).
@dataclass(frozen=True)
class Version:
    reward: object            # the reward function of one sample
    crash: str                # crash penalty: "small" = -20 (authors), "worst" = -5000 per step left
    ranges: object            # (low, high) gain ranges, see above - or "case": the case file's GAIN_RANGES
                              # - or "anchor": around the case file's ANCHOR_PID (see above)
    weight_b: bool = False    # True = the agent has a 4th slider for the setpoint weight b
    band_bonus: float = 0.0   # reward added per sample inside the 2 % band (improved_v6)
    log_std_init: float = None  # PPO's starting exploration noise, log of its std (None = SB3 default 0,
                                # std 1); improved_v6b: -1.6, std 0.2 = small tries around the anchor PID
    note: str = ""            # one-line description

    @property
    def n_sliders(self):
        """How many numbers the agent outputs: 3 gains, plus b for improved_v3."""
        return 4 if self.weight_b else 3

    def to_gains(self, action, ranges):
        """Sliders (-1..+1) -> (Kp, tau_i, tau_d, b), stretched into ranges = (low, high)."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)     # keep sliders inside -1..+1
        if self.ranges == "anchor":                                  # improved_v6: around the anchor PID
            (Kp0, tau_i0, tau_d0), span = ranges
            return (Kp0 * ANCHOR_KP_FACTOR ** a[0], tau_i0 * ANCHOR_TI_FACTOR ** a[1],
                    max(0.0, tau_d0 + a[2] * span), 1.0)
        low, high = ranges
        Kp, tau_i, tau_d = low + (a[:3] + 1.0) / 2.0 * (high - low)  # -1 -> low, +1 -> high
        b = (a[3] + 1.0) / 2.0 if self.weight_b else 1.0             # -1 -> 0, +1 -> 1
        return Kp, tau_i, tau_d, b


def gain_ranges(version_name, case):
    """The (low, high) gain ranges a version uses for a case study."""
    ranges = VERSIONS[version_name].ranges
    if ranges == "anchor":                              # ((Kp0, tau_i0, tau_d0), tau_d span) from the case file
        if getattr(case, "ANCHOR_PID", None) is None:
            raise SystemExit(f"{version_name} needs ANCHOR_PID and ANCHOR_TD_SPAN in cases/{case.ID}.py")
        return tuple(map(float, case.ANCHOR_PID)), float(case.ANCHOR_TD_SPAN)
    if isinstance(ranges, str):                         # "case": the case file chooses its own ranges
        if getattr(case, "GAIN_RANGES", None) is None:
            raise SystemExit(f"{version_name} needs GAIN_RANGES in cases/{case.ID}.py")
        return case.GAIN_RANGES
    return ranges


# ===== 4. The versions (name -> Version), in the order they were tried =====
# improved_v5 applies the idea behind improved_v4n2 to any process: the gain ranges are chosen
# per case study from a fixed-gain check (GAIN_RANGES in the case file). For CS1 they are exactly
# the ranges of improved_v4n2, so on CS1 improved_v5 and improved_v4n2 are the same version.
VERSIONS = {
    "paper":         Version(reward_paper,     "small", WIDE,    note="the authors' reward (reproduction)"),
    "improved":      Version(reward_squared,   "small", WIDE,    note="error penalty 1000x larger, cap 5000"),
    "improved_v2":   Version(reward_squared,   "worst", WIDE,    note="+ crashing costs the worst score for every step left"),
    "improved_v3":   Version(reward_squared,   "worst", WIDE,    weight_b=True,
                             note="+ setpoint weight b as a 4th slider"),
    "improved_v4":   Version(reward_overshoot, "worst", WIDE,    note="v2 + extra penalty for overshoot"),
    "improved_v4n":  Version(reward_overshoot, "worst", NARROW,  note="v4 + narrower gains (tau_i 0.05-1.0, tau_d 0-0.2)"),
    "improved_v4n2": Version(reward_overshoot, "worst", NARROW2, note="v4n + tau_i 0.05-0.5 (final for CS1)"),
    "improved_v5":   Version(reward_overshoot, "worst", "case",  note="v4 + gain ranges chosen per case study"),
    "improved_v6":   Version(reward_overshoot, "worst", "anchor", band_bonus=BAND_BONUS,
                             note="v5 + gains anchored on the best fixed PID + bonus inside the 2 % band"),
    "improved_v6b":  Version(reward_overshoot, "worst", "anchor", band_bonus=BAND_BONUS, log_std_init=-1.6,
                             note="v6 + small starting exploration (std 0.2 instead of 1)"),
    "improved_v6c":  Version(reward_overshoot, "worst", "anchor", band_bonus=5 * BAND_BONUS, log_std_init=-1.6,
                             note="v6b + 5x band bonus (pushes settling time; tried on CS1)"),
}
