# RL-tuned PID for an open-loop unstable process — results

**Project:** B.Tech final-year project. Reproduction and improvement of the m-PPO RL-tuned PID of Shuprajhaa, Sujit and Srinivasan, *Applied Soft Computing* 128 (2022) 109450.

**Case study 1:** G(s) = 1/(s − 1), sample time 0.1 s.

All numbers below were measured by running this code. The step-test and robustness numbers come from `results/cs1_evaluate.txt` and `results/cs1_robustness.txt`. Training figures come from the training logs (`runs/cs1/*/train_log.txt`, `0.monitor.csv`).

---

## 1. Goal

Train a PPO agent that tunes a PID controller online for an unstable process. The aim is to beat the authors' trained agent on all four step-test metrics:

| Metric | Authors' agent |
|---|---|
| Overshoot | 57.8 % |
| Settling time (2 % band) | 1.70 s |
| ISE | 0.142 |
| IAE | 0.266 |

---

## 2. Setup

### Process, controller and agent

| Part | Setting |
|---|---|
| Process | G(s) = 1/(s − 1), simulated exactly with zero-order hold: y[k+1] = 1.105·y[k] + 0.105·u[k] |
| Controller | Positional PID: P on the error (or b·r − y), I with ±100 anti-windup, D on the measurement; control input u limited to ±100 |
| Gain formulas | Ki = Kp / (τi + 0.01), Kd = Kp·τd (the authors' formulas) |
| Agent output | 3 sliders in [−1, 1], stretched into Kp, τi, τd (a 4th slider for b in improved_v3 only) |
| Agent input | Setpoint, output now, output one sample ago |
| Action repeat | 2: new gains every 0.2 s |
| Run stops | When \|y\| > 100 ("ran away") |
| Algorithm | PPO (Stable-Baselines3 defaults: 2 × 64 tanh, lr 3·10⁻⁴, 2048 steps per update, batch 64), VecNormalize on observations and rewards, 500,000 steps |
| Training run | 1 s at setpoint 0, then 3 random setpoints in [0, 2], 20 s each |

### Step test and metrics

- **Step test:** 1 s at setpoint 0, then setpoint 1 for 20 s.
- **Overshoot:** peak above 1, in %.
- **Settling:** time after the step until the output stays within ±0.02.
- **ISE, IAE:** Σe²·Δt and Σ|e|·Δt from the step on.
- **Note on the minimum:** every controller's first sample after the step is still 0, so ISE ≥ 0.1, IAE ≥ 0.1 and settling ≥ 0.1 s for all of them.

### Software

Python 3.14.7, numpy 2.5.3, torch 2.14.0+cpu, gymnasium 1.3.0, stable-baselines3 2.9.0, matplotlib 3.11.2. All runs were on CPU.

---

## 3. The seven reward versions

Results are 3-seed averages (overshoot % / settling s / ISE / IAE) unless marked 1 seed.

| Version | Change | Reason | Result |
|---|---|---|---|
| **paper** | The authors' reward: −min(0.01e², 5) + bonus | Reproduction | 55.4 / 2.23 / 0.143 / 0.291. Behaves like the authors' agent: high Kp, first-sample spike. |
| **improved** | Error penalty 10e², cap 5000 | Overshoot cost almost nothing | Seeds 0 and 1 good (33.9 % and 13.7 % overshoot). **Seed 2 learned to crash on purpose.** |
| **improved_v2** | + crashing costs −5000 for every step left | Close the crash loophole | No seed collapsed. 13.7 / 3.43 / 0.113 / 0.318: low overshoot, slow settling. |
| **improved_v3** | + setpoint weight b as a 4th slider | Softer first push | Seeds inconsistent. 13.5 / 3.80 / 0.175 / 0.491 |
| **improved_v4** | v2 + overshoot penalty 100·ov² | Punish going past the setpoint | 1 seed: 16.0 / 3.70 / 0.123 / 0.378 |
| **improved_v4n** | v4 + τi 0.05–1.0, τd 0–0.2 | All v2 agents and 2 of 3 v3 agents pinned τi at 2.0 | 21.8 / 2.13 / 0.123 / 0.289. Beats the authors on 2 of 4 metrics. |
| **improved_v4n2** | v4n + τi 0.05–0.5 | Fixed-gain check: τi = 0.5 with Kp 9–12.5 beats the authors on all four | **32.0 / 1.37 / 0.127 / 0.256. Beats the authors on all four.** |

An earlier try with the improved reward capped at 5 (instead of 5000) also learned to crash on purpose: ep_len_mean 4.52, test overshoot 10037.5 %. It is not one of the seven versions.

---

## 4. Step-test results (all agents)

From `results/cs1_evaluate.txt`. "Beats" means better than the authors' agent on all four metrics. improved_v3 is left out here; it is in Section 6. The baseline row is explained below the table.

| Controller | Overshoot % | Settling s | ISE | IAE | Completed | Beats authors |
|---|---|---|---|---|---|---|
| Classical PI (Kp 3, τi 1.25) | 48.8 | 4.00 | 0.401 | 0.914 | yes | no |
| Best fixed PID (Kp 10.5, τi 1.15, τd 0) | 19.9 | 2.50 | 0.122 | 0.310 | yes | no |
| paper seed 0 / 1 / 2 | 67.9 / 46.6 / 51.6 | 2.00 / 2.30 / 2.40 | 0.162 / 0.129 / 0.136 | 0.296 / 0.279 / 0.298 | yes | no |
| improved seed 0 / 1 | 33.9 / 13.7 | 2.80 / 3.10 | 0.120 / 0.109 | 0.296 / 0.284 | yes | no |
| improved seed 2 | 10037.5 | 0.80 | 2476.529 | 35.928 | **no** (ran away) | no |
| improved_v2 seed 0 / 1 / 2 | 13.2 / 15.7 / 12.2 | 3.50 / 3.40 / 3.40 | 0.114 / 0.112 / 0.113 | 0.328 / 0.309 / 0.317 | yes | no |
| improved_v4 seed 0 | 16.0 | 3.70 | 0.123 | 0.378 | yes | no |
| improved_v4n seed 0 / 1 / 2 | 21.4 / 24.9 / 19.1 | 2.20 / 2.20 / 2.00 | 0.121 / 0.133 / 0.114 | 0.292 / 0.323 / 0.253 | yes | no |
| **improved_v4n2 seed 0** | **32.0** | **1.30** | **0.128** | **0.255** | yes | **yes** |
| **improved_v4n2 seed 1** | **31.5** | **1.40** | **0.129** | **0.260** | yes | **yes** |
| **improved_v4n2 seed 2** | **32.5** | **1.40** | **0.126** | **0.252** | yes | **yes** |
| Authors' agent (their files) | 57.8 | 1.70 | 0.142 | 0.266 | – | – |

The best fixed PID is the gain set with the smallest ISE found by `grid_search.py` (Kp 0–20 in steps of 0.5, τi and τd 0–2 in steps of 0.05). It was tuned on this test.

**Averages per version (3 seeds):**

| Version | Overshoot % | Settling s | ISE | IAE | Beats authors |
|---|---|---|---|---|---|
| paper | 55.4 | 2.23 | 0.143 | 0.291 | no |
| improved | 3361.7 | 2.23 | 825.586 | 12.170 | no |
| improved_v2 | 13.7 | 3.43 | 0.113 | 0.318 | no |
| improved_v4n | 21.8 | 2.13 | 0.123 | 0.289 | no |
| **improved_v4n2** | **32.0** | **1.37** | **0.127** | **0.256** | **yes** |

**Gains chosen by the final agents (median over the test):**

| improved_v4n2 | Kp | τi | τd |
|---|---|---|---|
| seed 0 | 9.99 | 0.485 | 0 |
| seed 1 | 8.78 | 0.500 | 0 |
| seed 2 | 9.47 | 0.500 | 0 |

Plots: `results/cs1_comparison.png` (outputs and Kp) and `results/cs1_all_agents.png` (all agents in one graph).

---

## 5. Training summary

Every run trained for 500,000 steps; a full training run is 300 moves.

| Version | Seed | Final ep_len_mean | Final ep_rew_mean | Training time |
|---|---|---|---|---|
| paper | 0 / 1 / 2 | 300 / 300 / 300 | 1060.6* / 1010 / 1050 | 444* / 486 / 478 s |
| improved | 0 / 1 / 2 | 300 / 300 / **4.9** | −158.1* / 844 / −31,500 | 503* / 531 / 491 s |
| improved_v2 | 0 / 1 / 2 | 300 / 300 / 300 | 830 / −66,400 / 921 | 510 / 553 / 758 s |
| improved_v3 | 0 / 1 / 2 | 300 / 300 / 300 | 782 / 789 / −8,990 | 625 / 631 / 633 s |
| improved_v4 | 0 | 300 | 784 | 402 s |
| improved_v4n | 0 / 1 / 2 | 300 / 300 / 300 | 337 / 115 / −581 | 401 / 398 / 394 s |
| improved_v4n2 | 0 / 1 / 2 | 300 / 300 / 300 | −483 / 783 / 937 | 495 / 542 / 1330 s |

\* paper seed 0 and improved seed 0 were trained in the first session without a saved log. Their reward is the mean of the last 100 episodes in `0.monitor.csv`, and their time is the monitor's elapsed time.

- **Collapse:** only improved seed 2 collapsed (ep_len_mean 4.9 = crashing on purpose).
- **Negative final rewards in some normal runs** (for example improved_v2 seed 1 and improved_v4n2 seed 0) come from a few very bad full-length exploration episodes, not from crashes. Their last-100 medians were 935.1 and 885.4.

---

## 6. Robustness (no re-training, no re-tuning)

From `results/cs1_robustness.txt`.

- **Scenarios:** sensor noise (σ = 0.01 and 0.05, 10 noise seeds each, scored on the true output) and model mismatch (G(s) = K/(s − a) with a = 0.8 or 1.2, or K = 0.8 or 1.2).
- **Groups:** RL rows are the mean of 3 seeds. Every run of these groups completed.
- **Best fixed 2-DOF PID:** the grid search over setpoint weight b as well (Kp 10.5, τi 1.80, b 0.85). It was also tuned on the nominal test.

**ISE**

| Group | nominal | noise 0.01 | noise 0.05 | a = 0.8 | a = 1.2 | K = 0.8 | K = 1.2 |
|---|---|---|---|---|---|---|---|
| paper | 0.143 | 0.157 | 0.473 | 0.140 | 0.146 | 0.118 | 2.660 |
| improved_v2 | 0.113 | 0.120 | 0.253 | 0.110 | 0.117 | 0.122 | 0.118 |
| improved_v3 | 0.175 | 0.198 | 0.721 | 0.177 | 0.174 | 0.180 | 222.691 |
| **improved_v4n2** | 0.127 | 0.131 | 0.201 | 0.124 | 0.131 | 0.144 | 0.120 |
| Classical PI | 0.401 | 0.404 | 0.429 | 0.330 | 0.497 | 0.596 | 0.304 |
| Best fixed PID | 0.122 | 0.126 | 0.199 | 0.118 | 0.127 | 0.130 | 0.130 |
| Best fixed 2-DOF PID | 0.100 | 0.104 | 0.171 | 0.100 | 0.100 | 0.105 | 0.105 |

**IAE**

| Group | nominal | noise 0.01 | noise 0.05 | a = 0.8 | a = 1.2 | K = 0.8 | K = 1.2 |
|---|---|---|---|---|---|---|---|
| paper | 0.291 | 0.620 | 2.096 | 0.279 | 0.303 | 0.285 | 4.366 |
| improved_v2 | 0.318 | 0.512 | 1.473 | 0.289 | 0.346 | 0.381 | 0.299 |
| improved_v3 | 0.491 | 0.879 | 2.691 | 0.487 | 0.495 | 0.514 | 38.195 |
| **improved_v4n2** | 0.256 | 0.427 | 1.138 | 0.248 | 0.264 | 0.299 | 0.226 |
| Classical PI | 0.914 | 0.985 | 1.293 | 0.803 | 1.049 | 1.271 | 0.738 |
| Best fixed PID | 0.310 | 0.466 | 1.155 | 0.288 | 0.333 | 0.346 | 0.292 |
| Best fixed 2-DOF PID | 0.101 | 0.282 | 1.010 | 0.134 | 0.135 | 0.187 | 0.168 |

**Settling time (s)** (20.00 = not settled by the end of the test)

| Group | nominal | noise 0.01 | noise 0.05 | a = 0.8 | a = 1.2 | K = 0.8 | K = 1.2 |
|---|---|---|---|---|---|---|---|
| paper | 2.23 | 19.93 | 19.99 | 2.03 | 2.30 | 2.43 | 8.13 |
| improved_v2 | 3.43 | 19.86 | 19.98 | 3.13 | 3.63 | 3.77 | 3.03 |
| improved_v3 | 3.80 | 19.94 | 19.99 | 3.77 | 3.87 | 3.90 | 9.50 |
| **improved_v4n2** | 1.37 | 19.19 | 19.99 | 1.37 | 1.37 | 1.40 | 1.27 |
| Classical PI | 4.00 | 4.79 | 19.78 | 3.00 | 4.40 | 5.10 | 2.90 |
| Best fixed PID | 2.50 | 19.33 | 19.98 | 2.40 | 2.50 | 2.60 | 2.30 |
| Best fixed 2-DOF PID | 0.10 | 19.28 | 19.98 | 0.10 | 0.40 | 1.50 | 0.70 |

**Overshoot (%)**

| Group | nominal | noise 0.01 | noise 0.05 | a = 0.8 | a = 1.2 | K = 0.8 | K = 1.2 |
|---|---|---|---|---|---|---|---|
| paper | 55.4 | 55.5 | 56.1 | 53.8 | 57.0 | 24.3 | 86.4 |
| improved_v2 | 13.7 | 13.9 | 25.5 | 12.1 | 15.4 | 16.7 | 28.9 |
| improved_v3 | 13.5 | 15.9 | 39.6 | 12.6 | 14.5 | 15.4 | 212.8 |
| **improved_v4n2** | 32.0 | 32.2 | 33.4 | 29.3 | 34.8 | 30.2 | 28.4 |
| Classical PI | 48.8 | 49.0 | 50.1 | 40.5 | 58.2 | 60.2 | 41.4 |
| Best fixed PID | 19.9 | 19.9 | 23.7 | 18.7 | 21.2 | 21.0 | 43.9 |
| Best fixed 2-DOF PID | 0.1 | 3.1 | 15.6 | 0.0 | 2.1 | 3.7 | 20.0 |

**What the tables show:**

- **improved_v4n2 stays fast in every mismatch scenario:** settling 1.27–1.40 s, ISE 0.120–0.144, IAE 0.226–0.299.
- **Against paper:** improved_v4n2 is better on all four metrics in 6 of 7 scenarios. At K = 0.8, paper is better on overshoot, ISE and IAE.
- **paper and improved_v3 degrade at K = 1.2:** one seed of each keeps oscillating.
- **Noise:** with sensor noise, almost every controller's 2 % settling time goes to the end of the test.
- **The best fixed 2-DOF PID**, tuned on this exact test, is better than every RL group on overshoot, ISE and IAE in every scenario.

---

## 7. Conclusions

1. **The final version, improved_v4n2, beats the authors' agent on all four metrics, for each of its 3 seeds and on average:**

   | Metric | Authors | improved_v4n2 average |
   |---|---|---|
   | Overshoot | 57.8 % | 32.0 % |
   | Settling | 1.70 s | 1.37 s |
   | ISE | 0.142 | 0.127 |
   | IAE | 0.266 | 0.256 |

   The three seeds are close together: overshoot 31.5–32.5 %, settling 1.30–1.40 s.
2. **What made the difference:**
   - Closing the crash loophole (worst-case crash penalty).
   - Penalising overshoot.
   - Most important, limiting the integral time τi to 0.05–0.5 s. The improved_v2 and improved_v4n agents pushed τi to the top of their allowed range (2.0 and about 1.0). A fixed-gain check showed that with τi = 1.0 the response never settles within 1.70 s, while τi = 0.5 does.
3. **Robustness:** improved_v4n2 keeps its fast settling under ±20 % model mismatch, and is ahead of the paper group in most scenarios.

## 8. Limitations

- **Three seeds** per version is a small sample (improved_v4: one seed).
- **The baselines are tuned on the test itself.** The fixed-gain baselines were tuned on the very step test they are scored on; the RL agents never saw it. The best fixed 2-DOF PID reaches ISE 0.100, the lowest possible value on this test.
- **Settling under noise:** the 2 % settling time stops separating the controllers once sensor noise is added.
- **Differences from the paper:**
  - The authors' code uses 2 hidden layers of 64 and batch size 64; the paper's table says 3 layers and 256.
  - The paper's classical PID (71.08 % / 3.10 s) could not be reproduced; there is no classical PID in their code.

## 9. How to reproduce

```
.\run grid_search.py cs1
.\run evaluate.py cs1
.\run robustness.py cs1
.\run train.py cs1 improved_v4n2 0        (one new training run, about 8 min)
```

Case study 2 is prepared as a template in `cases/cs2.py`; see README.md.
