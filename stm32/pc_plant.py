# stm32/pc_plant.py : the PROCESS runs here on the PC; the STM32 is the controller (hardware in the loop)
# ---------------------------------------------------------------
# The board (firmware built with  stm32\build.bat pc) holds the trained RL agent and the PID. Every
# sample this script sends it the setpoint and the measured output, the board answers with the
# control input u, and this script moves the process model (cases/csN.py, the same model as in
# training) one sample forward with that u. The board never sees the process - only r and y.
#
# Usage (from the project folder):
#   .\run stm32\pc_plant.py test                     step test 0 -> 1, compared with Python
#   .\run stm32\pc_plant.py live                     real time, setpoint changes, live plot
#   .\run stm32\pc_plant.py live --bump 12:0.3       ... with a disturbance of +0.3 on y at t = 12 s
#   .\run stm32\pc_plant.py live --param K=1.2 --noise 0.01     a different process / sensor noise
# Options: --port COM7 (default: the ST-LINK port, found automatically)
#          --sim       no board: the firmware's C controller compiled on this PC (same protocol)
#          --seconds 60  --hold 8  --setpoints 1,0.5,1.5,1   (live)
# The live run is saved to stm32/logs/hil_<date>_<time>.csv and .png.
# Protocol: see stm32/firmware/Inc/hil.h.
# ---------------------------------------------------------------
import argparse, os, subprocess, sys, time
from datetime import datetime
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cases import load_case
from core.tools import load_agent, run_test, score


# ===== the two ways to reach the controller =====
class BoardLink:
    """The STM32 over its serial port (ST-LINK virtual COM port, 115200 baud)."""

    def __init__(self, port):
        import serial                                   # pyserial (requirements.txt)
        self.port = serial.Serial(port or find_port(), 115200, timeout=3)
        time.sleep(0.2)
        self.port.reset_input_buffer()                  # drop the start-up text of the board

    def send(self, text):
        self.port.write((text + "\n").encode())

    def receive(self):
        line = self.port.readline().decode(errors="replace").strip()
        if not line:
            raise SystemExit("No answer from the board. Is the 'pc' firmware flashed (build.bat pc)?")
        return line


class SimLink:
    """No board: the firmware's controller code (rl_pid.c + hil.c) compiled with the PC's gcc."""

    def __init__(self):
        exe = os.path.join(HERE, "build", "host_hil.exe" if os.name == "nt" else "host_hil")
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        src = os.path.join(HERE, "firmware", "Src")
        subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-I", os.path.join(HERE, "firmware", "Inc"),
                        os.path.join(HERE, "host_test", "host_hil.c"), os.path.join(src, "rl_pid.c"),
                        os.path.join(src, "hil.c"), "-o", exe, "-lm"], check=True)
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def send(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def receive(self):
        return self.proc.stdout.readline().strip()


def find_port():
    """The ST-LINK virtual COM port (USB vendor 0x0483 = STMicroelectronics); never guesses another."""
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == 0x0483 or "STLink" in (p.description or "") or "STMicro" in (p.manufacturer or ""):
            return p.device
    names = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none"
    raise SystemExit(f"No ST-LINK COM port found (ports: {names}). Plug in the board or give --port.")


class Controller:
    """Speaks the protocol of hil.h over a link."""

    def __init__(self, link):
        self.link = link

    def ask(self, text, answer):
        self.link.send(text)
        for _ in range(50):                               # skip any other text (e.g. start-up lines)
            line = self.link.receive()
            if line.startswith(answer + " ") or line == answer:
                return line.split()[1:]
            if line.startswith("E "):
                raise SystemExit(f"Controller error: {line}")
        raise SystemExit(f"No '{answer}' answer to '{text}'")

    def info(self):
        agent, dt, u_lim, y_lim, repeat = self.ask("I", "I")
        return agent, float(dt)

    def reset(self, y0):
        self.ask(f"Z {y0:.9g}", "Z")

    def step(self, r, y):
        u, Kp, tau_i, tau_d, cycles = self.ask(f"S {r:.9g} {y:.9g}", "U")
        return float(u), (float(Kp), float(tau_i), float(tau_d)), int(cycles)


# ===== the step test: exactly the loop of core/env.py, with the controller on the board =====
def step_test(ctrl, case, params, noise, seed=0):
    r = np.concatenate([np.full(case.WARMUP_STEPS, case.START_SETPOINT),
                        np.full(case.HOLD_STEPS, case.TEST_SETPOINT)])
    process = case.Process(**params)
    y = np.full(len(r), process.reset())
    rng = np.random.default_rng(seed)
    seen = lambda i: y[i] + (rng.normal(0.0, noise) if noise > 0 else 0.0)
    k = case.WARMUP_STEPS - 1                             # "now" = the last warm-up sample
    ctrl.reset(y[k - 1])
    cycles = []
    while True:
        u, _, c = ctrl.step(r[k], seen(k))
        cycles.append(c)
        y[k + 1] = process.step(u)
        k += 1
        if abs(y[k]) > case.Y_LIMIT or k >= len(r) - 1:
            break
    return score(SimpleNamespace(case=case, y=y, r=r, k=k)), abs(y[k]) <= case.Y_LIMIT, cycles


def run_test_mode(ctrl, case, agent, args, params):
    m, done, cycles = step_test(ctrl, case, params, args.noise)
    print(f"Step test of {agent}, process on the PC{'':5s}overshoot  settling      ISE      IAE")
    label = f"controller on the {'PC (--sim)' if args.sim else 'STM32'}"
    print(f"  {label:44s}{m[0]:9.1f} %{m[1]:9.2f}{m[2]:9.3f}{m[3]:9.3f}   completed {'yes' if done else 'no'}")
    if cycles and max(cycles) > 0:
        c = np.array(cycles)
        print(f"  board time per controller step: mean {c.mean():.0f} cycles, max {c.max()} cycles "
              f"(at 168 MHz: {c.max() / 168:.1f} us)")
    if params or args.noise:
        return                                            # a changed process: nothing to compare with
    env, _ = run_test(case, load_agent(case, agent.split("/")[1]))
    p = score(env)
    print(f"  {'Python (the same agent, all in Python)':44s}{p[0]:9.1f} %{p[1]:9.2f}{p[2]:9.3f}{p[3]:9.3f}")
    same = all(round(a, n) == round(b, n) for a, b, n in zip(m, p, (1, 2, 3, 3)))
    print("RESULT:", "PASS - the loop through the controller gives the Python numbers" if same
          else "DIFFERENT - see the numbers above")


# ===== live: real time, setpoint changes, live plot =====
def run_live(ctrl, case, agent, args, params):
    import matplotlib.pyplot as plt
    dt = case.DT
    setpoints = [float(v) for v in args.setpoints.split(",")]
    hold = args.hold if args.hold else 50 * dt
    bump_t, bump = (float(v) for v in args.bump.split(":")) if args.bump else (None, 0.0)
    n = int(round(args.seconds / dt))
    process = case.Process(**params)
    y = process.reset()
    ctrl.reset(y)
    rng = np.random.default_rng(0)
    log = []                                             # t, r, y, u, Kp, tau_i, tau_d, cycles

    plt.ion()
    fig, (ax_y, ax_u, ax_g) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle(f"Controller on the {'PC (--sim)' if args.sim else 'STM32'}: {agent}   |   process on the PC: {case.NAME}")
    start = time.perf_counter()
    for k in range(n):
        t = k * dt
        r = setpoints[min(int(t // hold), len(setpoints) - 1)]
        if bump_t is not None and abs(t - bump_t) < dt / 2:
            y = process.bump(bump)                       # a sudden disturbance on the output
        u, gains, cycles = ctrl.step(r, y + (rng.normal(0.0, args.noise) if args.noise else 0.0))
        log.append((t, r, y, u, *gains, cycles))
        y = process.step(u)
        if abs(y) > case.Y_LIMIT:
            print(f"The output ran away at t = {t:.1f}: stopping.")
            break
        if k % max(1, int(0.5 / dt)) == 0 or k == n - 1:  # redraw about twice a second
            a = np.array(log)
            for ax in (ax_y, ax_u, ax_g):
                ax.clear()
                ax.grid(alpha=0.3)
            ax_y.step(a[:, 0], a[:, 1], "--", color="#52514e", where="post", label="setpoint")
            ax_y.plot(a[:, 0], a[:, 2], color="#2a78d6", lw=2, label="output y")
            ax_u.step(a[:, 0], a[:, 3], color="#eb6834", where="post", label="control input u")
            ax_g.step(a[:, 0], a[:, 4], color="#2a78d6", where="post", label="Kp")
            ax_g.step(a[:, 0], a[:, 5], color="#1baf7a", where="post", label="tau_i")
            ax_g.step(a[:, 0], a[:, 6], color="#6250d6", where="post", label="tau_d")
            for ax in (ax_y, ax_u, ax_g):
                ax.legend(fontsize=8, loc="upper right")
            ax_g.set_xlabel("time (s)" if dt < 1 else "time (samples)")
            plt.pause(0.001)
        wait = start + (k + 1) * dt - time.perf_counter()  # keep real time: one sample every dt
        if wait > 0:
            time.sleep(wait)

    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    name = os.path.join(HERE, "logs", f"hil_{datetime.now():%Y%m%d_%H%M%S}")
    np.savetxt(name + ".csv", np.array(log), delimiter=",", fmt="%.6g",
               header="t,setpoint,output,control_input,Kp,tau_i,tau_d,controller_cycles", comments="")
    fig.savefig(name + ".png", dpi=120)
    print(f"Saved {name}.csv and .png")
    plt.ioff()
    plt.show()


# ===== command line =====
parser = argparse.ArgumentParser(description="Process on the PC, RL-PID controller on the STM32.")
parser.add_argument("mode", choices=["test", "live"], nargs="?", default="test")
parser.add_argument("--port", help="serial port of the board (default: the ST-LINK port)")
parser.add_argument("--sim", action="store_true", help="no board: the C controller compiled on this PC")
parser.add_argument("--seconds", type=float, default=40.0, help="live: length of the run")
parser.add_argument("--hold", type=float, default=0.0, help="live: seconds per setpoint (default 50 samples)")
parser.add_argument("--setpoints", default="1,0.5,1.5,1", help="live: setpoints, one after another")
parser.add_argument("--bump", help="live: TIME:AMOUNT, a sudden disturbance on the output")
parser.add_argument("--noise", type=float, default=0.0, help="sensor noise (standard deviation)")
parser.add_argument("--param", action="append", default=[], help="process change, e.g. K=1.2 (repeatable)")
args = parser.parse_args()

controller = Controller(SimLink() if args.sim else BoardLink(args.port))
agent, dt = controller.info()                            # e.g. "cs1/improved_v6b_seed2"
case = load_case(agent.split("/")[0])
if abs(dt - case.DT) > 1e-6:
    raise SystemExit(f"The controller runs at dt = {dt}, but {case.ID} has dt = {case.DT}")
params = {k: float(v) for k, v in (p.split("=") for p in args.param)}
print(f"Controller: {agent} ({'compiled on this PC' if args.sim else 'on the STM32'}); "
      f"process on the PC: {case.NAME} {params or ''}")
(run_test_mode if args.mode == "test" else run_live)(controller, case, agent, args, params)
