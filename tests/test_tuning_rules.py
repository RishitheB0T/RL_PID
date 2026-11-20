# tests/test_tuning_rules.py : the published tuning rules reproduce the paper's own examples
# ---------------------------------------------------------------
# Kaya (ELECO 2017), Table 3: G(s) = e^(-0.2s)/(s - 1);  Table 4: G(s) = 4 e^(-2s)/(4s - 1).
# The paper prints its results to 4 decimals; our formulas agree to within 0.5 %
# (largest difference 0.4 %, Ti of example 2: the paper's constants are rounded to 3-4 digits).
# ---------------------------------------------------------------
import unittest
from core.tuning_rules import kaya_2017, classical_set

# (K, T, theta, controller, criterion) -> the paper's (Kc, Ti, Td)
PAPER = [
    ((1, 1, 0.2, "PI", "ISTE"), (4.1118, 1.5207, 0.0)),
    ((1, 1, 0.2, "PID", "ISTE"), (5.8524, 0.6767, 0.0949)),
    ((1, 1, 0.2, "PI", "IST2E"), (3.7235, 1.2890, 0.0)),
    ((1, 1, 0.2, "PID", "IST2E"), (5.9715, 0.6140, 0.0836)),
    ((4, 4, 2.0, "PI", "ISTE"), (0.4364, 27.7618, 0.0)),
    ((4, 4, 2.0, "PID", "ISTE"), (0.6270, 8.1408, 0.9772)),
    ((4, 4, 2.0, "PI", "IST2E"), (0.4000, 28.0047, 0.0)),
    ((4, 4, 2.0, "PID", "IST2E"), (0.6217, 8.6164, 0.8608)),
]


class TestKaya2017(unittest.TestCase):
    def test_paper_examples(self):
        for args, expected in PAPER:
            for ours, paper in zip(kaya_2017(*args), expected):
                tolerance = 0.005 * abs(paper) + 1e-4          # 0.5 % (the paper's rounding)
                self.assertAlmostEqual(ours, paper, delta=tolerance, msg=str(args))

    def test_outside_range_is_refused(self):
        with self.assertRaises(ValueError):
            kaya_2017(1, 1, 0.05)

    def test_gains_keep_the_rule_integral_gain(self):
        """Our Ki = Kp/(tau_i + 0.01) must equal the rule's Kc/Ti."""
        for name, (Kp, tau_i, tau_d) in classical_set(1, 1, 0.2):
            Kc, Ti, Td = kaya_2017(1, 1, 0.2, *name.replace("Kaya 2017 ", "").replace("(", "").replace(")", "").split())
            self.assertAlmostEqual(Kp / (tau_i + 0.01), Kc / Ti, places=12)


if __name__ == "__main__":
    unittest.main()
