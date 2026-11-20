# train.py : train one PPO agent
# ---------------------------------------------------------------
# Usage:   .\run train.py <case> <version> <seed> [steps] [--select]
# Example: .\run train.py cs1 improved_v4n2 0          (500,000 steps = the authors' number)
#          .\run train.py cs3 improved_v6 0 1000000 --select
#
# Saves the agent to runs/<case>/<version>_seed<seed>/ :
#   model.zip         the trained network
#   vecnormalize.pkl  its input scaling (needed to use the agent later)
#   0.monitor.csv     score and length of every training run
#   log.txt           the training progress table (ep_len_mean, ep_rew_mean, ...)
#
# --select (used for improved_v6): every 50,000 steps the agent is tried on the VALIDATION tests
# (core/validation.py - never the final step test) and the best checkpoint so far is saved as
# model.zip + vecnormalize.pkl. The network at the very end is kept as final_model.zip +
# final_vecnormalize.pkl, and every validation result is written to validation.csv.
#
# PPO settings = Stable-Baselines3 defaults (as in the authors' code): 2 hidden layers of 64,
# tanh, learning rate 3e-4, 2048 steps per update, batch 64, 10 epochs, gamma 0.99.
# ---------------------------------------------------------------
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from cases import load_case
from core.env import RLPIDEnv
from core.tools import runs_folder
from core.validation import anchor_metrics, selection_score, validation_metrics
from core.versions import VERSIONS, gain_ranges

# ----- command line: case, version, seed, steps, --select (sys.argv = Java's String[] args) -----
args = [a for a in sys.argv[1:] if a != "--select"]
SELECT = "--select" in sys.argv
if len(args) < 3:
    raise SystemExit("Usage: .\\run train.py <case> <version> <seed> [steps] [--select]   e.g.  cs1 improved_v4n2 0")
CASE = load_case(args[0])
VERSION = args[1]
SEED = int(args[2])
STEPS = int(args[3]) if len(args) > 3 else 500_000
VALIDATE_EVERY = 50_000
if VERSION not in VERSIONS:
    raise SystemExit(f"Unknown version '{VERSION}'. Choose one of: {', '.join(VERSIONS)}")

OUT = os.path.join(runs_folder(CASE.ID), f"{VERSION}_seed{SEED}")
if os.path.exists(os.path.join(OUT, "model.zip")):
    raise SystemExit(f"{OUT} already has a trained agent - delete that folder first to retrain.")


class KeepBestOnValidation(BaseCallback):
    """Every VALIDATE_EVERY steps: score the agent on the validation tests; save it if it is the best so far."""

    def __init__(self):
        super().__init__()
        self.anchor = anchor_metrics(CASE)       # the anchor PID's results: the yardstick (score 1.0)
        self.best = float("inf")
        self.ranges = gain_ranges(VERSION, CASE)
        with open(os.path.join(OUT, "validation.csv"), "w") as f:
            f.write("steps,score,best_so_far\n")

    def controller(self, obs):
        """The agent as it is right now (no exploration noise, input scaling not updated)."""
        action = self.model.predict(self.training_env.normalize_obs(obs), deterministic=True)[0]
        return VERSIONS[VERSION].to_gains(action, self.ranges)

    def _on_step(self):
        if self.num_timesteps % VALIDATE_EVERY == 0:
            score = selection_score(validation_metrics(CASE, self.controller), self.anchor, CASE.DT)
            if score < self.best:
                self.best = score
                self.model.save(os.path.join(OUT, "model"))
                self.training_env.save(os.path.join(OUT, "vecnormalize.pkl"))
            with open(os.path.join(OUT, "validation.csv"), "a") as f:
                f.write(f"{self.num_timesteps},{score:.6f},{self.best:.6f}\n")
            print(f"validation at {self.num_timesteps} steps: score {score:.4f} (best {self.best:.4f})")
        return True                              # True = keep training


# ----- the practice world, wrapped for PPO -----
# make_vec_env builds the world, seeds it and logs every run to OUT/0.monitor.csv.
env = make_vec_env(lambda: RLPIDEnv(CASE, VERSION), n_envs=1, seed=SEED, monitor_dir=OUT)
# VecNormalize rescales what the agent sees and the rewards to about mean 0, std 1 (the authors' choice).
env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

# ----- the agent and its training -----
model = PPO("MlpPolicy", env, seed=SEED, verbose=1)
model.set_logger(configure(OUT, ["stdout", "log"]))     # progress on screen AND in OUT/log.txt
model.learn(total_timesteps=STEPS, callback=KeepBestOnValidation() if SELECT else None)

final = "final_" if SELECT else ""                      # with --select, model.zip is the best checkpoint
model.save(os.path.join(OUT, f"{final}model"))
env.save(os.path.join(OUT, f"{final}vecnormalize.pkl"))
print(f"Saved to {OUT}")
