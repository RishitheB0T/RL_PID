# tests/test_hil.py : hardware in the loop without a board - process on the PC, controller in C
# ---------------------------------------------------------------
# Runs stm32/pc_plant.py --sim: the firmware's controller code (rl_pid.c + hil.c) is compiled on
# the PC and driven through the same text protocol the STM32 uses; the process runs in Python.
# The step test must give exactly the Python numbers. Skipped if gcc is not installed.
# ---------------------------------------------------------------
import os, shutil, subprocess, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipIf(shutil.which("gcc") is None, "gcc is not installed")
class TestHardwareInTheLoop(unittest.TestCase):
    def test_step_test_through_the_protocol(self):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "stm32", "pc_plant.py"), "test", "--sim"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertIn("RESULT: PASS", result.stdout, result.stdout + result.stderr)

    def test_protocol_answers(self):
        """The controller answers I, Z and S, and reports an unknown command as an error."""
        exe = os.path.join(ROOT, "stm32", "build", "host_hil.exe" if os.name == "nt" else "host_hil")
        if not os.path.exists(exe):
            self.test_step_test_through_the_protocol()          # builds it
        out = subprocess.run([exe], input="I\nZ 0\nS 1 0\nX\n", capture_output=True, text=True).stdout.split("\n")
        self.assertEqual(out[0], "READY")
        self.assertTrue(out[1].startswith("I cs"), out[1])
        self.assertEqual(out[2], "Z OK")
        self.assertTrue(out[3].startswith("U ") and len(out[3].split()) == 6, out[3])
        self.assertTrue(out[4].startswith("E "), out[4])


if __name__ == "__main__":
    unittest.main()
