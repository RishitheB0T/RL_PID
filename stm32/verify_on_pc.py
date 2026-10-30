# stm32/verify_on_pc.py : check the C controller against Python, on the PC
# ---------------------------------------------------------------
# Usage (from the project folder):
#     .\run stm32\verify_on_pc.py cs1 improved_v4n2_seed0
#
# 1. exports the agent to stm32/firmware/Inc/agent_params.h
# 2. compiles the firmware's controller code (rl_pid.c) with the PC's gcc
# 3. runs the step test in C and in Python and compares the outputs and the four metrics
# The C code uses single precision (float), like the Cortex-M4 FPU; Python uses double, so tiny
# differences are expected. The check passes if the metrics agree to the printed precision.
# ---------------------------------------------------------------
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from export_agent import export
from cases import load_case
from core.tools import load_agent, run_test, score, version_of

case_id = sys.argv[1] if len(sys.argv) > 1 else "cs1"
agent = sys.argv[2] if len(sys.argv) > 2 else "improved_v4n2_seed0"

# 1. export
export(case_id, agent)

# 2. compile with the PC's gcc
build = os.path.join(HERE, "build")
os.makedirs(build, exist_ok=True)
exe = os.path.join(build, "host_test.exe" if os.name == "nt" else "host_test")
cmd = ["gcc", "-O2", "-std=c99", "-Wall", "-I", os.path.join(HERE, "firmware", "Inc"),
       os.path.join(HERE, "host_test", "host_main.c"), os.path.join(HERE, "firmware", "Src", "rl_pid.c"),
       "-o", exe, "-lm"]
subprocess.run(cmd, check=True)

# 3. run C and Python
lines = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.splitlines()
c_metrics = [float(v) for v in next(l for l in lines if l.startswith("metrics")).split()[1:5]]
c_y = np.array([float(l.split()[2]) for l in lines if l.startswith("y ")])

case = load_case(case_id)
env, _ = run_test(case, load_agent(case, agent), version_of(agent))
py_metrics = score(env)
py_y = env.y[: len(c_y)]
n = env.k + 1

print(f"\nStep test of {case_id}/{agent}:       overshoot %   settling s      ISE      IAE")
print(f"  Python (double)                  {py_metrics[0]:11.1f} {py_metrics[1]:10.2f} {py_metrics[2]:8.3f} {py_metrics[3]:8.3f}")
print(f"  C on the PC (float, as the STM32) {c_metrics[0]:10.1f} {c_metrics[1]:10.2f} {c_metrics[2]:8.3f} {c_metrics[3]:8.3f}")
print(f"  largest output difference over the test: {np.max(np.abs(c_y[:n] - py_y[:n])):.2e}")
same = (round(c_metrics[0], 1) == round(py_metrics[0], 1) and round(c_metrics[1], 2) == round(py_metrics[1], 2)
        and round(c_metrics[2], 3) == round(py_metrics[2], 3) and round(c_metrics[3], 3) == round(py_metrics[3], 3))
print("RESULT:", "PASS - C and Python agree to the printed precision" if same else "DIFFERENT - see the numbers above")
