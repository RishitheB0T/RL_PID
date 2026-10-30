# Deploying the RL-tuned PID on an STM32F4 (ARM Cortex-M4)

Target board: **STM32F407G-DISC1** (STM32F4 Discovery, STM32F407VG, Cortex-M4F at 168 MHz,
1 MB flash, 192 KB RAM).

The trained agent runs **on the microcontroller itself**: every 0.2 s the neural network
(3 → 64 → 64 → 3, tanh) chooses the PID gains, and every 0.1 s the PID computes the control input.
The process is simulated on the chip, or a real process is connected through the DAC and ADC.

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

## 1. Choose the agent (on the PC)

From the project folder:

```
.\run stm32\export_agent.py cs1 improved_v4n2_seed0
```

Any trained agent of a first-order case study can be exported, for example `cs2 improved_v5_seed0`.
The exported file in the repository is the CS1 final agent `improved_v4n2_seed0`.

## 2. Check the C code against Python (on the PC)

```
.\run stm32\verify_on_pc.py cs1 improved_v4n2_seed0
```

This needs `gcc` on the PATH (for example MinGW). Measured result for the CS1 final agent:

| Step test 0 → 1 | Overshoot | Settling | ISE | IAE |
|---|---|---|---|---|
| Python (double precision) | 32.0 % | 1.30 s | 0.128 | 0.255 |
| C code (single precision, as on the Cortex-M4) | 32.0 % | 1.30 s | 0.128 | 0.255 |

Largest difference in the output over the whole test: 7.6 × 10⁻⁸.

## 3. Build the firmware

Requirements, both free from st.com:
- **STM32CubeIDE**, installed in `C:\ST` (only its ARM compiler and programmer are used).
- **The STM32CubeF4 package**, in `%USERPROFILE%\STM32Cube\Repository\STM32Cube_FW_F4_V1.28.3`. STM32CubeIDE downloads it the first time you create an STM32F4 project.

```
cd stm32
.\build.bat            (process simulated on the chip)
.\build.bat real       (real process through the DAC and ADC)
```

Output: `build\firmware\rlpid.elf` and `rlpid.bin`. The build uses about 43 KB of flash and 4 KB of RAM.

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
agent cs1/improved_v4n2_seed0 - CS1: G(s) = 1/(s - 1)
step test on the chip: overshoot ... settling ... ISE ... IAE ... completed yes
measured in Python   : overshoot 32.0 %  settling 1.30 s  ISE 0.128  IAE 0.255  completed yes
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

## Using STM32CubeIDE instead of build.bat

1. Create an STM32 project for the **STM32F407VGTx** (or the board STM32F407G-DISC1).
2. Copy `firmware/Inc/*.h` into `Core/Inc` and `firmware/Src/rl_pid.c` into `Core/Src`.
3. Call `rlpid_step_test()`, `rlpid_controller_step()` and `rlpid_plant_step()` from the
   generated `main.c` as in this `main.c`, with a 100 ms loop and UART output.
4. Enable the FPU (the default for this chip) and add `-u _printf_float` to the linker flags if
   floats are printed.
