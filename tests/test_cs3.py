# tests/test_cs3.py : checks of case study 3 (the authors' CS3 process)
# ---------------------------------------------------------------
# Run from the project folder:   .\run -m unittest discover tests
# ---------------------------------------------------------------
import unittest
import numpy as np
from cases import load_case

CS3 = load_case("cs3")


class TestCaseStudy3(unittest.TestCase):
    def test_strictly_proper_part(self):
        """(0.77248 s - 0.1727)/(3.1 s - 1) = d + c/(s - a) with a = 1/3.1 and c = 0.02467."""
        a = 1.0 / 3.1
        d = 0.77248 / 3.1
        c = (-0.1727 + d) / 3.1
        self.assertAlmostEqual(c, 0.024673, places=6)
        process = CS3.Process()
        self.assertAlmostEqual(process.A, np.exp(a), places=12)
        self.assertAlmostEqual(process.B, c * (np.exp(a) - 1.0) / a, places=12)

    def test_first_sample_and_no_direct_term(self):
        """From rest, u = 1 for one sample gives y = B (the authors' code adds no D*u term)."""
        process = CS3.Process()
        process.reset()
        self.assertAlmostEqual(process.step(1.0), process.B, places=12)

    def test_unstable_without_control(self):
        """With u = 0, any output grows by e^(1/3.1) = 1.381 per sample."""
        process = CS3.Process()
        process.reset()
        process.bump(1.0)
        self.assertAlmostEqual(process.step(0.0), np.exp(1 / 3.1), places=12)


if __name__ == "__main__":
    unittest.main()
