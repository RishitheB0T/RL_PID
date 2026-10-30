# train.py : train one PPO agent
# ---------------------------------------------------------------
# Usage:   .\run train.py <case> <version> <seed> [steps]
# Example: .\run train.py cs1 improved_v4n2 0          (500,000 steps = the authors' number)
#
# Saves the agent to runs/<case>/<version>_seed<seed>/ :
#   model.zip         the trained network
#   vecnormalize.pkl  its input scaling (needed to use the agent later)
#   0.monitor.csv     score and length of every training run
#   log.txt           the training progress table (ep_len_mean, ep_rew_mean, ...)
#
# PPO settings = Stable-Baselines3 defaults (as in the authors' code): 2 hidden layers of 64,
# tanh, learning rate 3e-4, 2048 steps per update, batch 64, 10 epochs, gamma 0.99.
# ---------------------------------------------------------------
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from cases import load_case
from core.env import RLPIDEnv
from core.tools import runs_folder
from core.versions import VERSIONS

# ----- command line: case, version, seed, steps (sys.argv = Java's String[] args) -----
if len(sys.argv) < 4:
    raise SystemExit("Usage: .\\run train.py <case> <version> <seed> [steps]   e.g.  cs1 improved_v4n2 0")
CASE = load_case(sys.argv[1])
VERSION = sys.argv[2]
SEED = int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 500_000
if VERSION not in VERSIONS:
    raise SystemExit(f"Unknown version '{VERSION}'. Choose one of: {', '.join(VERSIONS)}")

OUT = os.path.join(runs_folder(CASE.ID), f"{VERSION}_seed{SEED}")
if os.path.exists(os.path.join(OUT, "model.zip")):
    raise SystemExit(f"{OUT} already has a trained agent - delete that folder first to retrain.")

# ----- the practice world, wrapped for PPO -----
# make_vec_env builds the world, seeds it and logs every run to OUT/0.monitor.csv.
env = make_vec_env(lambda: RLPIDEnv(CASE, VERSION), n_envs=1, seed=SEED, monitor_dir=OUT)
# VecNormalize rescales what the agent sees and the rewards to about mean 0, std 1 (the authors' choice).
env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

# ----- the agent and its training -----
model = PPO("MlpPolicy", env, seed=SEED, verbose=1)
model.set_logger(configure(OUT, ["stdout", "log"]))     # progress on screen AND in OUT/log.txt
model.learn(total_timesteps=STEPS)

model.save(os.path.join(OUT, "model"))
env.save(os.path.join(OUT, "vecnormalize.pkl"))
print(f"Saved to {OUT}")
