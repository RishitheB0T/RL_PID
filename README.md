# RL_PID — Reinforcement-learning-tuned PID control of an unstable process

A PPO agent adjusts the gains of a PID controller **online** to control the open-loop unstable
process **G(s) = 1/(s − 1)**. The project reproduces and improves the m-PPO RL-PID of
T. Shuprajhaa, S.K. Sujit and K. Srinivasan, *"Reinforcement learning based adaptive PID controller
design for control of linear/nonlinear unstable processes"*, Applied Soft Computing 128 (2022) 109450.

![Live demo: the trained agent tracks setpoint changes and rejects a disturbance](results/cs1_demo.gif)

## Result

The final version (**improved_v4n2**) beats the authors' trained agent on all four step-test
metrics, for each of its 3 training seeds and on average:

| Step test 0 → 1 | Overshoot | Settling (2 %) | ISE | IAE |
|---|---|---|---|---|
| Authors' agent | 57.8 % | 1.70 s | 0.142 | 0.266 |
| **improved_v4n2 (mean of 3 seeds)** | **32.0 %** | **1.37 s** | **0.127** | **0.256** |

Full results, robustness tests and the story of all seven reward versions: [REPORT.md](REPORT.md).

## How it works

```
             setpoint r                      control input u
  r(t) ──►  ┌──────────┐   e = r − y   ┌─────┐  ──────────►  ┌─────────────┐ ──► y(t)
            │ PPO agent│ ── Kp,τi,τd ─► │ PID │               │ G(s)=1/(s−1)│
            └──────────┘               └─────┘  ◄──────────  └─────────────┘
                 ▲          sees: r, y now, y one sample ago        │
                 └──────────────────────────────────────────────────┘
```

Every 0.2 s the agent chooses new PID gains; the PID then controls the process with them.
The agent was trained with a reward that punishes the tracking error, extra-punishes overshoot
and makes crashing (output running away) the worst possible outcome.

## Project structure

```
core/         pid.py         the PID controller
              versions.py    the reward versions (one table) and the slider -> gain mapping
              env.py         the training / test environment (works for every case study)
              tools.py       loading agents, the step test, scoring, saving results
cases/        cs1.py         case study 1: G(s) = 1/(s - 1)
              cs2.py         case study 2: G(s) = e^(-0.4s)/(s - 1)  (with dead time)
              template.py    blank template for a new case study
stm32/        the controller in C and the firmware for the STM32F4 Discovery (see stm32/README.md)
tests/        automatic checks (measured results, CS2 process, C code = Python)
runs/         the trained agents: runs/<case>/<version>_seed<N>/
results/      every table and plot the scripts produce
train.py  grid_search.py  evaluate.py  robustness.py  plot_all_agents.py  live_demo.py
```

## Setup (Windows, Python 3.14)

```
git clone https://github.com/RishitheB0T/RL_PID.git
cd RL_PID
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`run.bat` then runs every script with that environment.

## Usage (PowerShell, from the project folder)

```
.\run grid_search.py cs1                          # best fixed-gain PID (baseline), ~15 s
.\run evaluate.py cs1                             # results table + plot, ~10 s
.\run robustness.py cs1                           # sensor noise and model mismatch, ~15 s
.\run plot_all_agents.py cs1                      # all step responses in one graph
.\run live_demo.py cs1 improved_v4n2_seed0        # watch an agent live (keys: Up/Down, d, space, n)
.\run live_demo.py cs1 improved_v4n2_seed0 --gif demo.gif
.\run train.py cs1 improved_v4n2 3                # train a new agent, ~8 min
.\run -m unittest discover tests                  # check that everything still works
```

Use `cs2` instead of `cs1` for case study 2.

Reward versions: `paper`, `improved`, `improved_v2`, `improved_v3`, `improved_v4`,
`improved_v4n`, `improved_v4n2` (final for CS1) and `improved_v5` (the same idea with gain ranges
chosen per case study) — see `core/versions.py`.

## Running on an STM32F4 (ARM Cortex-M4)

The trained controller also runs on an **STM32F407G-DISC1** board: the neural network, the PID
and a simulated process all run on the microcontroller (a real process can be connected through
the DAC and ADC). The C code gives the same step-test results as Python. See
[stm32/README.md](stm32/README.md).

```
.\run stm32\export_agent.py cs1 improved_v4n2_seed0     # agent -> C header
.\run stm32\verify_on_pc.py cs1 improved_v4n2_seed0     # C code vs Python, on the PC
stm32\build.bat                                         # build the firmware
stm32\flash.bat                                         # write it to the board
```

## Adding another case study

Copy `cases/template.py` to `cases/cs3.py`. It has the same layout as `cases/cs1.py`, with the
process and its settings left blank. Fill in every line marked `TODO` (name, sample time, limits,
setpoints and the `Process` class), then run the same commands with `cs3`. Until then the scripts
list what is still missing.

## Author

Rishikesh Kallapally — B.Tech final-year project (Electronics and Instrumentation Engineering).
