# Deploying the RL-tuned PID on an STM32F4 (ARM Cortex-M4)

Target board: **STM32F407G-DISC1** (STM32F4 Discovery, STM32F407VG, Cortex-M4F at 168 MHz,
1 MB flash, 192 KB RAM).

The trained agent runs **on the microcontroller itself**: every 0.2 s the neural network
(3 → 64 → 64 → 3, tanh) chooses the PID gains, and every 0.1 s the PID computes the control input.
The process is simulated on the chip, or a real process is connected through the DAC and ADC, or
(the edge-controller setup, Section 6) the process runs on the PC and the board only receives the
measured output and sends back the control input.

## Files

| File | What it does |
|---|---|
| `export_agent.py` | Turns a trained agent into `firmware/Inc/agent_params.h`: weights, input scaling, gain ranges, process model and the Python step-test results |
| `firmware/Inc/rl_pid.h`, `firmware/Src/rl_pid.c` | The controller in plain C: network, slider → gain mapping, PID, simulated process, step test. No hardware code, so it also runs on a PC |
| `firmware/Src/main.c` | The STM32 program: clock (168 MHz), UART, LEDs, button, step test, live control loop |
| `firmware/Src/syscalls.c` | Minimal C-library hooks for a chip without an operating system |
| `firmware/Inc/stm32f4xx_hal_conf.h` | Which parts of ST's HAL library are used |
| `host_test/host_main.c`, `verify_on_pc.py` | Compile the same C code on the PC and compare it with Python |
| `build.bat`, `flash.bat` | Build the firmware and write it to the board, without any IDE project |
| `simulate.bat` | Run the built firmware on a simulated STM32F4 Discovery (Renode), no board needed |
| `firmware/Inc/hil.h`, `firmware/Src/hil.c` | Hardware in the loop: the text protocol through which the PC sends (setpoint, output) and the board answers with the control input |
| `pc_plant.py` | The process on the PC; the board is the controller (step test vs Python, live run with plot) |
| `host_test/host_hil.c` | The board's controller + protocol compiled on the PC: `pc_plant.py --sim` works without a board |

## 1. Choose the agent (on the PC)

From the project folder:

```
.\run stm32\export_agent.py cs1 improved_v6b_seed2
```

Any trained agent of a first-order case study can be exported (CS1, CS2 with its dead time,
CS3), with either gain mapping: the linear ranges of the older versions or the improved_v6/v6b
mapping around the anchor PID. The exported file in the repository is the CS1 agent
`improved_v6b_seed2` (the CS1 seed with the best validation score; for CS2 and CS3 these are
`improved_v6b_seed3`). The timing follows the case: CS1/CS2 sample every 0.1 s, CS3 every 1 s.

## 2. Check the C code against Python (on the PC)

```
.\run stm32\verify_on_pc.py cs1 improved_v6b_seed2
```

This needs `gcc` on the PATH (for example MinGW). Measured results (overshoot / settling / ISE / IAE;
C single precision as on the Cortex-M4, Python double precision):

| Agent | Python | C code | Largest output difference |
|---|---|---|---|
| CS1 improved_v6b seed 2 | 15.1 % / 0.90 s / 0.106 / 0.164 | 15.1 % / 0.90 s / 0.106 / 0.164 | 8.5 × 10⁻⁸ |
| CS2 improved_v6b seed 3 | 83.2 % / 5.70 s / 1.214 / 1.960 | 83.2 % / 5.70 s / 1.214 / 1.960 | 2.8 × 10⁻⁷ |
| CS3 improved_v6b seed 3 | 25.1 % / 7.00 / 1.122 / 1.721 | 25.1 % / 7.00 / 1.122 / 1.721 | 1.4 × 10⁻⁷ |
| CS1 improved_v4n2 seed 0 (earlier final agent) | 32.0 % / 1.30 s / 0.128 / 0.255 | 32.0 % / 1.30 s / 0.128 / 0.255 | 7.6 × 10⁻⁸ |

## 3. Build the firmware

Requirements, both free from st.com:
- **STM32CubeIDE**, installed in `C:\ST` (only its ARM compiler and programmer are used).
- **The STM32CubeF4 package**, in `%USERPROFILE%\STM32Cube\Repository\STM32Cube_FW_F4_V1.28.3`. STM32CubeIDE downloads it the first time you create an STM32F4 project.

```
cd stm32
.\build.bat            (process simulated on the chip)
.\build.bat real       (real process through the DAC and ADC)
.\build.bat pc         (the board is only the controller; the process runs on the PC, Section 6)
```

Output: `build\firmware\rlpid.elf` and `rlpid.bin`. With the improved_v6b agent the build uses
44,584 bytes of code + 468 bytes of initialised data (about 45 KB of flash) and 3,664 bytes of RAM;
no compiler warnings.

## 4. Flash and run

Connect the board's mini-USB (ST-LINK) port, then:

```
.\flash.bat
```

Open the board's virtual COM port at **115200 baud, 8N1**, for example with PuTTY or the
STM32CubeIDE terminal. It is on USART2: PA2 = TX, PA3 = RX. If no COM port appears, connect a
USB-to-serial adapter to PA2/PA3 and GND. After a reset the board prints:

```
=== RL-tuned PID on STM32F407 (168 MHz) ===
agent cs1/improved_v6b_seed2 - CS1: G(s) = 1/(s - 1)
step test on the chip: overshoot ... settling ... ISE ... IAE ... completed yes
measured in Python   : overshoot 15.1 %  settling 0.90 s  ISE 0.106  IAE 0.164  completed yes
one network decision: ... cycles = ... us
live control (simulated process), blue button = next setpoint
t_s,setpoint,output,control_input,Kp,tau_i,tau_d
0.0,1.000,...
```

From the CSV lines, the live response can be plotted in Excel or Python.

| LED / button | Meaning |
|---|---|
| Green (PD12) | Blinks with every sample: the controller is running |
| Orange (PD13) | Step test running |
| Blue (PD15) | Output within 2 % of the setpoint |
| Red (PD14) | The output ran away; the simulated run restarts |
| Blue USER button (PA0) | Next setpoint: 1.0 → 0.5 → 1.5 → 0.0 → 1.0 … |

## 5. A real process (optional)

Build with `.\build.bat real`. Connections:

| Pin | Use |
|---|---|
| PA4 (DAC) | Control input u: −100 … +100 → 0 … 3.3 V |
| PA1 (ADC) | Measured output y: 0 … 3.3 V → −0.5 … 2.5 |

Adapt the scaling (`REAL_Y_AT_0V`, `REAL_Y_AT_3V3` in `main.c`, and the DAC mapping) to your
circuit and sensor. Keep the signals inside 0 … 3.3 V.

## 6. The STM32 as the controller of a process on the PC (hardware in the loop)

Here the board is **only the controller**, like an edge controller next to a plant. It holds the
trained RL agent and the PID, and it never sees the process: every sample it gets the setpoint r
and the measured output y and answers with the control input u. The process runs on the PC in
`pc_plant.py`, with the same model the agent was trained on (`cases/csN.py`).

```
 PC (pc_plant.py)                                   STM32F407 (build.bat pc)
 ┌──────────────────────────────┐   "S r y"   ┌──────────────────────────────────┐
 │ process model  y[k+1] = f(u) │ ──────────► │ RL network -> Kp, tau_i, tau_d     │
 │ setpoints, disturbances,     │             │ PID -> u                           │
 │ noise, live plot, CSV log    │ ◄────────── │ (every ACTION_REPEAT samples the  │
 └──────────────────────────────┘ "U u Kp ..." │  network chooses new gains)       │
                                              └──────────────────────────────────┘
```

**Protocol** (`firmware/Inc/hil.h`), one text line per message at 115200 baud. You can also try it
by hand in a serial terminal.

| PC sends | Board answers |
|---|---|
| `I` | `I <agent> <dt> <u_limit> <y_limit> <action_repeat>` |
| `Z <y0>` | `Z OK` (reset: empty integral, previous output y0) |
| `S <r> <y>` | `U <u> <Kp> <tau_i> <tau_d> <cycles>` (cycles = CPU cycles of this controller step) |

**Run it:**

```
.\run stm32\export_agent.py cs1 improved_v6b_seed2     (choose the agent; CS2/CS3: improved_v6b_seed3)
stm32\build.bat pc
stm32\flash.bat
.\run stm32\pc_plant.py test                            step test 0 -> 1 through the board, compared with Python
.\run stm32\pc_plant.py live                            real time, setpoints 1 -> 0.5 -> 1.5 -> 1, live plot
.\run stm32\pc_plant.py live --bump 12:0.3              ... plus a sudden disturbance at t = 12 s
.\run stm32\pc_plant.py live --param K=1.2 --noise 0.01 ... a different process and sensor noise
```

- `pc_plant.py` reads the agent's name from the board and picks the matching process model.
- The live run is saved to `stm32/logs/hil_<date>_<time>.csv` and `.png`.
- The PC keeps the time (one sample every dt). The board answers within its controller time, which `test` reports in CPU cycles.
- The ST-LINK COM port is found automatically (USB vendor 0x0483). The script never picks another port, for example the Bluetooth ports on this PC; `--port COMx` sets it by hand.

**Without a board:** `--sim` compiles the board's controller code (`rl_pid.c` + `hil.c`) on the PC
and talks to it through the same protocol (`host_test/host_hil.c`). Measured with `--sim`, the step
test through the protocol gives exactly the Python numbers for all three processes:

| Agent | Through the protocol (C controller) | Python |
|---|---|---|
| CS1 improved_v6b seed 2 | 15.1 % / 0.90 s / 0.106 / 0.164 | 15.1 % / 0.90 s / 0.106 / 0.164 |
| CS2 improved_v6b seed 3 | 83.2 % / 5.70 s / 1.214 / 1.960 | 83.2 % / 5.70 s / 1.214 / 1.960 |
| CS3 improved_v6b seed 3 | 25.1 % / 7.00 / 1.122 / 1.721 | 25.1 % / 7.00 / 1.122 / 1.721 |

The firmware in `pc` mode builds with no warnings: 51,180 bytes of code, 3,920 bytes of RAM.
**Not yet run with the real board** (none was connected); the first `pc_plant.py test` on the board is that check.

## 7. Run it without a board: simulated STM32 (Renode)

[Renode](https://renode.io) (free, open source, by Antmicro) simulates the STM32F4 Discovery
board and runs the **same `rlpid.elf`** that goes on the real board. The network, the PID and the
simulated process run inside the simulated Cortex-M4, and the serial output appears in a window.

1. Download `renode-1.17.0.windows-portable.zip` from https://github.com/renode/renode/releases
   and unpack it into `rlpid_step1\tools\renode\`. Nothing needs installing.
2. Build, then simulate:

```
.\build.bat
.\simulate.bat            (window with the serial output, live)
.\simulate.bat log 10     (no window: 10 s of board time, output saved to build\renode_uart.txt)
```

The step-test line must match the Python numbers, as on the real board. The timing in Renode is
simulated, so its "network decision" cycle count is not a hardware measurement.

**Status:** `simulate.bat` has not yet been run on this PC, so the first run is its test. If the
output stops after the first line, the clock setup (`clock_168mhz` in `main.c`) is the likely place:
it waits for the crystal and the PLL, which the simulated clock controller must report as ready.

## 8. Which setup to use

| Setup | What runs where | What you need | Use it for |
|---|---|---|---|
| **A. Process on the PC** (Section 6, `build.bat pc`) | Network and PID on the STM32 (the edge controller); the process in Python on the PC | the board, `pyserial` (in requirements.txt) | **The main demo:** the STM32 controls a process it cannot see, only through r, y and u, as with a real plant. Processes, noise and disturbances change on the PC without reflashing |
| **B. Real board, process on the chip** (default firmware) | Network, PID **and** process on the STM32 | the board and a serial terminal | Exact real-time timing with no PC in the loop |
| **C. Real board, real process** (`build.bat real`) | Network and PID on the STM32; a real circuit on the DAC/ADC | wiring, e.g. an op-amp model of the process | A physical experiment |
| **D. Renode** (Section 7) | Everything in a simulated STM32 | Renode (free) | Checking a build with no board |

Without a board, `pc_plant.py --sim` shows setup A with the board's controller code compiled on the PC.

**Recommended:** use A as the main demo. Its live plot shows the setpoint, output, control input and
the gains the board chooses.

Not recommended:
- QEMU: its STM32F4 support is limited.
- Proteus: paid, with limited STM32F4 support.
- STM32CubeIDE: it has no instruction-set simulator, only the debugger for a real board.

## Using STM32CubeIDE instead of build.bat

1. Create an STM32 project for the **STM32F407VGTx** (or the board STM32F407G-DISC1).
2. Copy `firmware/Inc/*.h` into `Core/Inc` and `firmware/Src/rl_pid.c` into `Core/Src`.
3. Call `rlpid_step_test()`, `rlpid_controller_step()` and `rlpid_plant_step()` from the
   generated `main.c` as in this `main.c`, with a loop of `RLPID_DT_MS` milliseconds (100 ms for
   CS1/CS2, 1000 ms for CS3) and UART output.
4. Enable the FPU (the default for this chip) and add `-u _printf_float` to the linker flags if
   floats are printed.
