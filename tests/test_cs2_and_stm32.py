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
    def test_c_matches_python(self):
        """The C controller of the firmware gives the same step-test metrics as Python."""
        result = subprocess.run([sys.executable, os.path.join(ROOT, "stm32", "verify_on_pc.py"),
                                 "cs1", "improved_v4n2_seed0"], capture_output=True, text=True, cwd=ROOT)
        self.assertIn("RESULT: PASS", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
