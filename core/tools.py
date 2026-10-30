# core/tools.py : small helpers shared by all scripts
# ---------------------------------------------------------------
# - where things are stored (runs/, results/)
# - loading a trained agent, making a fixed-gain controller
# - running the step test and scoring it (overshoot, settling, ISE, IAE)
# - Printer: prints a line AND keeps it, so the whole output can be saved to results/
#
# A "controller" here is a function:  observation -> (Kp, tau_i, tau_d, b)
# ---------------------------------------------------------------
import glob, json, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from core.env import RLPIDEnv
from core.versions import VERSIONS, gain_ranges

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the project folder


# ===== where things are stored =====
def runs_folder(case_id):
    """runs/cs1/ : one sub-folder per trained agent, e.g. runs/cs1/improved_v4n2_seed0/"""
    return os.path.join(ROOT, "runs", case_id)


def results_path(filename):
    """results/<filename> : every table and plot the scripts produce."""
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    return os.path.join(ROOT, "results", filename)


def find_agents(case_id):
    """[(name, folder), ...] for every trained agent of a case study, sorted by name."""
    folders = sorted(glob.glob(os.path.join(runs_folder(case_id), "*_seed*")))
    return [(os.path.basename(f), f) for f in folders if os.path.exists(os.path.join(f, "model.zip"))]


def version_of(agent_name):
    """'improved_v4n2_seed0' -> 'improved_v4n2'"""
    return agent_name.split("_seed")[0]


# ===== controllers =====
def load_agent(case, agent_name):
    """A trained agent as a controller. Its saved input scaling is loaded and frozen."""
    folder = os.path.join(runs_folder(case.ID), agent_name)
    if not os.path.exists(os.path.join(folder, "model.zip")):
        names = ", ".join(name for name, _ in find_agents(case.ID)) or "none yet"
        raise SystemExit(f"No trained agent '{agent_name}' in runs/{case.ID}/. Available: {names}")
    version = version_of(agent_name)
    model = PPO.load(os.path.join(folder, "model"))                       # the trained network
    norm = VecNormalize.load(os.path.join(folder, "vecnormalize.pkl"),    # its input scaling
                             DummyVecEnv([lambda: RLPIDEnv(case, version)]))
    norm.training = False                                                 # freeze the scaling
    ranges = gain_ranges(version, case)

    def controller(obs):
        action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]   # best action, no noise
        return VERSIONS[version].to_gains(action, ranges)
    return controller


def fixed(Kp, tau_i, tau_d, b=1.0):
    """A fixed-gain controller: always the same gains, whatever it sees."""
    return lambda obs: (Kp, tau_i, tau_d, b)


def load_fixed_gains(case_id):
    """The gains found by grid_search.py (None if it has not been run yet)."""
    path = results_path(f"{case_id}_fixed_gains.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def reference_controllers(case, include_2dof=False):
    """The fixed-gain controllers the agents are compared with: [(name, controller), ...].
    The classical PI comes from the case file; the best fixed PID (and, if asked, the best fixed
    2-DOF PID) from grid_search.py - they are simply left out if grid_search.py was not run."""
    refs = []
    if case.CLASSICAL_PI:
        refs.append(("Classical PI", fixed(*case.CLASSICAL_PI)))
    best = load_fixed_gains(case.ID)
    if best:
        wanted = [("best_fixed_pid", "Best fixed PID")]
        if include_2dof:
            wanted.append(("best_fixed_2dof", "Best fixed 2-DOF PID"))
        for key, title in wanted:
            g = best[key]
            refs.append((title, fixed(g["Kp"], g["tau_i"], g["tau_d"], g["b"])))
    return refs


# ===== the step test =====
def run_test(case, controller, version="paper", noise_std=0.0, process_params=None, seed=None,
             hold_steps=None):
    """Play the step test once. Returns the finished world and the gains used at every move."""
    env = RLPIDEnv(case, version, test=True, noise_std=noise_std, process_params=process_params,
                   hold_steps=hold_steps)
    obs, _ = env.reset(seed=seed)          # the seed only matters when there is noise
    gains = []
    while True:
        g = controller(obs)
        gains.append(g)
        obs, _, terminated, truncated, _ = env.step_gains(*g)
        if terminated or truncated:
            return env, np.array(gains, dtype=float)


def score(env):
    """[overshoot %, settling time s (2 % band), ISE, IAE], measured from the setpoint step on."""
    c = env.case
    start = c.WARMUP_STEPS
    y = env.y[start: env.k + 1]                      # true output after the step
    e = env.r[start: env.k + 1] - y                  # error after the step
    size = c.TEST_SETPOINT - c.START_SETPOINT        # size of the step
    outside = np.where(np.abs(e) > 0.02 * abs(size))[0]       # samples outside the 2 % band
    peak = y.max() if size > 0 else y.min()          # furthest point in the step's direction
    overshoot = max(0.0, (peak - c.TEST_SETPOINT) / size * 100)
    settling = (outside[-1] + 1) * c.DT if len(outside) else 0.0   # last sample outside the band + 1
    return np.array([overshoot, settling, np.sum(e ** 2) * c.DT, np.sum(np.abs(e)) * c.DT])


def completed(env):
    """True if the test run reached its end (it did not run away)."""
    return env.k == len(env.r) - 1


def beats(metrics, target):
    """'yes' if all four metrics are smaller than the target's, else 'no' ('-' if no target)."""
    if target is None:
        return "-"
    return "yes" if np.all(np.asarray(metrics) < np.asarray(target)) else "no"


# ===== printing =====
class Printer:
    """show("text") prints a line and keeps it; show.save(path) writes all kept lines to a file."""
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"(saved to {os.path.relpath(path, ROOT)})")
