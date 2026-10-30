# live_demo.py : watch a trained agent control the process live - or record it as a GIF
# ---------------------------------------------------------------
# Usage:   .\run live_demo.py <case> <agent> [--gif file.gif]
# Example: .\run live_demo.py cs1 improved_v4n2_seed0                  (live window)
#          .\run live_demo.py cs1 improved_v4n2_seed0 --gif demo.gif   (30 s recording)
#
# Keys in the live window:
#   Up / Down  setpoint up / down by a quarter of the test step (kept inside the training range)
#   d          disturbance: a sudden bump on the output - watch the agent recover
#   space      pause / resume          n   start again
#
# Every decision you see is made live by the trained network; nothing is replayed.
# ---------------------------------------------------------------
import itertools, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from cases import load_case
from core.env import RLPIDEnv
from core.tools import load_agent, version_of
from core.versions import VERSIONS

# ===== 1. What to show =====
if len(sys.argv) < 3:
    raise SystemExit("Usage: .\\run live_demo.py <case> <agent> [--gif file.gif]   "
                     "e.g.  cs1 improved_v4n2_seed0")
case = load_case(sys.argv[1])
AGENT = sys.argv[2]
GIF = sys.argv[sys.argv.index("--gif") + 1] if "--gif" in sys.argv else None
VERSION = version_of(AGENT)
controller = load_agent(case, AGENT)

STEP = case.TEST_SETPOINT - case.START_SETPOINT          # size of the test step (1 for CS1)
N = 300 if GIF else 3000                                 # samples: 30 s for the GIF, 300 s live
SCRIPT = {80: case.START_SETPOINT + 0.5 * STEP,          # GIF only: at these samples the setpoint
          150: case.START_SETPOINT + 1.5 * STEP,         # changes, and at the last one the output
          220: "bump"}                                   # gets a sudden bump


# ===== 2. One run: the world, what the agent sees, and what happened so far =====
class LiveRun:
    def __init__(self):
        self.env = RLPIDEnv(case, VERSION, test=True, hold_steps=N - case.WARMUP_STEPS)
        self.obs, _ = self.env.reset(seed=0)
        self.log = []                 # (time, Kp, tau_i, tau_d, b) of every move
        self.done_events = set()      # scripted GIF events that already happened
        self.paused = False
        self.finished = False

    def set_setpoint(self, value):
        """New setpoint from the next sample on (kept inside the training range)."""
        lo, hi = case.TRAIN_RANGE
        self.env.r[self.env.k + 1:] = min(hi, max(lo, value))
        self.obs = self.env.observation()

    def bump(self):
        """A sudden disturbance on the output."""
        self.env.kick(0.3 * STEP)
        self.obs = self.env.observation()

    def move(self):
        """The agent chooses gains, the world runs one move with them."""
        if GIF:
            for sample, event in SCRIPT.items():
                if self.env.k >= sample and sample not in self.done_events:
                    self.bump() if event == "bump" else self.set_setpoint(event)
                    self.done_events.add(sample)
        gains = controller(self.obs)
        self.log.append((self.env.k * case.DT,) + tuple(gains))
        self.obs, _, terminated, truncated, _ = self.env.step_gains(*gains)
        self.finished = terminated or truncated


run = LiveRun()

# ===== 3. The window: process output, control input, PID gains =====
low, high = VERSIONS[VERSION].ranges
fig, (ax_y, ax_u, ax_k) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True,
                                       gridspec_kw={"height_ratios": [1.4, 1, 1]})
line_r, = ax_y.step([], [], "k--", where="post", lw=1.2, label="Setpoint r(t)")
line_y, = ax_y.plot([], [], color="#CC5A30", lw=2.4, label="Process output y(t)")
line_u, = ax_u.step([], [], color="#1F3A5F", where="post", lw=1.6, label="Control input u(t)")
line_kp, = ax_k.step([], [], color="#2E6FB7", where="post", lw=2, label="Kp")
ax_k2 = ax_k.twinx()                                     # second y-axis for tau_i
line_ti, = ax_k2.step([], [], color="#D98E1F", where="post", lw=2, label="τi")
info = ax_y.text(0.01, 0.97, "", transform=ax_y.transAxes, va="top", fontsize=10.5,
                 bbox=dict(boxstyle="round", fc="white", ec="#CBD5E1"))
lo, hi = case.TRAIN_RANGE
ax_y.set_ylim(lo - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))
ax_k.set_ylim(low[0], high[0])
ax_k2.set_ylim(-0.02, high[1] * 1.1)
ax_y.set_ylabel("Process output y")
ax_u.set_ylabel("Control input u")
ax_k.set_ylabel("Proportional gain Kp", color="#2E6FB7")
ax_k2.set_ylabel("Integral time τi (s)", color="#D98E1F")
ax_k.set_xlabel("Time (s)")
ax_y.legend(loc="upper right")
ax_u.legend(loc="upper right")
for ax in (ax_y, ax_u, ax_k):
    ax.grid(alpha=0.3)
fig.suptitle(f"RL-tuned PID ({AGENT}) controlling {case.NAME}", fontweight="bold")


def frame(_):
    """Called by the animation every 200 ms: one move, then redraw."""
    if not run.paused and not run.finished:
        run.move()
    env, k = run.env, run.env.k
    t = np.arange(k + 1) * case.DT
    line_r.set_data(t, env.r[:k + 1])
    line_y.set_data(t, env.y[:k + 1])
    line_u.set_data(t[:-1], env.u[:k])
    ax_u.relim()
    ax_u.autoscale_view(scalex=False)
    if run.log:
        g = np.array(run.log)
        line_kp.set_data(g[:, 0], g[:, 1])
        line_ti.set_data(g[:, 0], g[:, 2])
        b_text = f"   b {g[-1, 4]:.2f}" if VERSIONS[VERSION].weight_b else ""
        info.set_text(f"t = {k * case.DT:5.1f} s   setpoint r = {env.r[k]:.2f}   output y = {env.y[k]:.3f}\n"
                      f"PID gains chosen by the agent: Kp {g[-1, 1]:.2f}   τi {g[-1, 2]:.3f}   "
                      f"τd {g[-1, 3]:.3f}{b_text}"
                      + ("\n[finished - press n]" if run.finished and not GIF else ""))
    now = k * case.DT
    ax_y.set_xlim(max(0, now - 15), max(15, now))         # a scrolling 15-second window
    return line_r, line_y, line_u, line_kp, line_ti, info


def on_key(event):
    global run
    if event.key == "up":
        run.set_setpoint(run.env.r[run.env.k] + 0.25 * STEP)
    elif event.key == "down":
        run.set_setpoint(run.env.r[run.env.k] - 0.25 * STEP)
    elif event.key == "d":
        run.bump()
    elif event.key == " ":
        run.paused = not run.paused
    elif event.key == "n":
        run = LiveRun()


fig.canvas.mpl_connect("key_press_event", on_key)

# ===== 4. Record a GIF or show the live window =====
if GIF:
    anim = FuncAnimation(fig, frame, frames=range(N // case.ACTION_REPEAT), interval=200, blit=False)
    anim.save(GIF, writer=PillowWriter(fps=10), dpi=80)   # 10 frames/s = twice real speed for CS1
    print(f"Saved {GIF}")
else:
    anim = FuncAnimation(fig, frame, frames=itertools.count(), interval=200, blit=False,
                         cache_frame_data=False)
    plt.show()
