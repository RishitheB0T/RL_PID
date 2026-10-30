# tests/test_cs1.py : automatic checks that the code still gives the measured results
# ---------------------------------------------------------------
# Run from the project folder:   .\run -m unittest discover tests
# (unittest is Python's built-in test library, like JUnit in Java: every method whose name
#  starts with "test" is one check; self.assert... fails the check if the condition is wrong.)
# ---------------------------------------------------------------
import unittest
import numpy as np
from cases import load_case
from core.pid import PID
from core.tools import fixed, load_agent, run_test, score, completed, beats
from core.versions import VERSIONS, gain_ranges

CS1 = load_case("cs1")


class TestBuildingBlocks(unittest.TestCase):
    def test_process_is_exact_discretisation(self):
        """1/(s - 1) with DT = 0.1: from rest, a push of 1 gives y = e^0.1 - 1 after one sample."""
        process = CS1.Process()
        process.reset()
        self.assertAlmostEqual(process.step(1.0), np.exp(0.1) - 1.0, places=12)

    def test_pid_first_sample(self):
        """Error 1, Kp = 2, Ki = 5: u = 2*1 + 5*1*0.1 = 2.5 (no D part on the first sample)."""
        pid = PID(dt=0.1, limit=100.0)
        self.assertAlmostEqual(pid.control(r=1.0, y=0.0, Kp=2.0, Ki=5.0, Kd=3.0), 2.5, places=12)

    def test_pid_respects_limit(self):
        pid = PID(dt=0.1, limit=100.0)
        self.assertEqual(pid.control(r=1.0, y=0.0, Kp=1000.0, Ki=0.0, Kd=0.0), 100.0)

    def test_slider_to_gain_mapping(self):
        """-1 -> lowest, +1 -> highest value of the version's ranges."""
        final = VERSIONS["improved_v4n2"]
        ranges = gain_ranges("improved_v4n2", CS1)
        self.assertEqual(tuple(map(float, final.to_gains([-1, -1, -1], ranges))), (0.0, 0.05, 0.0, 1.0))
        self.assertEqual(tuple(map(float, final.to_gains([1, 1, 1], ranges))), (20.0, 0.5, 0.2, 1.0))

    def test_v5_uses_the_case_ranges(self):
        """improved_v5 on CS1 has exactly the ranges of improved_v4n2."""
        for a, b in zip(gain_ranges("improved_v5", CS1), gain_ranges("improved_v4n2", CS1)):
            np.testing.assert_array_equal(a, b)


def as_printed(m):
    """The four metrics rounded exactly as the results table prints them."""
    return (round(m[0], 1), round(m[1], 2), round(m[2], 3), round(m[3], 3))


class TestMeasuredResults(unittest.TestCase):
    def test_classical_pi(self):
        env, _ = run_test(CS1, fixed(*CS1.CLASSICAL_PI))
        self.assertTrue(completed(env))
        self.assertEqual(as_printed(score(env)), (48.8, 4.00, 0.401, 0.914))

    def test_final_agent_beats_authors(self):
        env, _ = run_test(CS1, load_agent(CS1, "improved_v4n2_seed0"), "improved_v4n2")
        m = score(env)
        self.assertEqual(as_printed(m), (32.0, 1.30, 0.128, 0.255))
        self.assertEqual(beats(m, CS1.AUTHORS), "yes")


class TestCaseTemplate(unittest.TestCase):
    def test_blank_template_says_what_is_missing(self):
        with self.assertRaises(SystemExit) as stop:
            load_case("template")
        self.assertIn("Still blank", str(stop.exception))


if __name__ == "__main__":
    unittest.main()
