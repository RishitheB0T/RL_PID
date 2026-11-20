# tests/test_cs2_and_stm32.py : checks of case study 2 and of the C (STM32) controller
# ---------------------------------------------------------------
# Run from the project folder:   .\run -m unittest discover tests
# The C check is skipped automatically if gcc is not installed.
# ---------------------------------------------------------------
import os, shutil, subprocess, sys, unittest
import numpy as np
from cases import load_case
from core.versions import gain_ranges

CS2 = load_case("cs2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCaseStudy2(unittest.TestCase):
    def test_dead_time_is_four_samples(self):
        """A push of 1 changes nothing for 4 samples (0.4 s), then y = e^0.1 - 1."""
        process = CS2.Process()
        process.reset()
        outputs = [process.step(1.0) for _ in range(5)]
        self.assertEqual(outputs[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(outputs[4], np.exp(0.1) - 1.0, places=12)

    def test_v5_ranges_come_from_the_case(self):
        low, high = gain_ranges("improved_v5", CS2)
        self.assertEqual(list(low), [0.0, 0.5, 0.0])
        self.assertEqual(list(high), [5.0, 4.0, 0.5])


@unittest.skipIf(shutil.which("gcc") is None, "gcc is not installed")
class TestCController(unittest.TestCase):
    """The C controller of the firmware gives the same step-test metrics as Python.
    verify_on_pc.py rewrites stm32/firmware/Inc/agent_params.h, so the firmware's default agent
    (CS1 improved_v6b seed 2) is checked LAST (unittest runs the tests in name order)."""

    def check(self, case_id, agent):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "stm32", "verify_on_pc.py"),
                                 case_id, agent], capture_output=True, text=True, cwd=ROOT)
        self.assertIn("RESULT: PASS", result.stdout, result.stdout + result.stderr)

    def test_1_linear_ranges_cs1_v4n2(self):
        self.check("cs1", "improved_v4n2_seed0")

    def test_2_anchor_mapping_cs3_v6b(self):
        self.check("cs3", "improved_v6b_seed3")

    def test_3_firmware_default_cs1_v6b(self):
        self.check("cs1", "improved_v6b_seed2")


if __name__ == "__main__":
    unittest.main()
